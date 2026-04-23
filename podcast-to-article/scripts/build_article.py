from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from article_builder import ArticleSection, VideoCandidate, build_outline_sections, render_article_markdown
    from normalize import Segment, merge_timed_segments, normalize_timed_content
    from oxylabs_client import OxylabsClient, OxylabsError
    from serpapi_client import SerpApiClient
    from utils import detect_input_type, extract_video_id, format_timestamp, load_local_env, parse_credentials, parse_serpapi_key, resolve_setting, slugify
else:
    from .article_builder import ArticleSection, VideoCandidate, build_outline_sections, render_article_markdown
    from .normalize import Segment, merge_timed_segments, normalize_timed_content
    from .oxylabs_client import OxylabsClient, OxylabsError
    from .serpapi_client import SerpApiClient
    from .utils import detect_input_type, extract_video_id, format_timestamp, load_local_env, parse_credentials, parse_serpapi_key, resolve_setting, slugify


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
DEFAULT_TEMPLATE = CODE_ROOT / "templates" / "article-template.md"


@dataclass
class ResolvedVideo:
    candidate: VideoCandidate
    metadata_payload: dict[str, Any]
    content_payload: dict[str, Any]
    source_kind: str
    origin: Optional[str] = None


def _metadata_chapters(raw_chapters: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_chapters, list):
        return []
    chapters: list[dict[str, Any]] = []
    for index, chapter in enumerate(raw_chapters):
        if not isinstance(chapter, dict):
            continue
        title = chapter.get("title") or chapter.get("chapter") or f"Chapter {index + 1}"
        start = chapter.get("start_time")
        if start is None:
            start = chapter.get("time_start")
        if start is None and chapter.get("start_ms") is not None:
            start = int(chapter["start_ms"]) // 1000
        if start is None:
            continue
        start_sec = parse_duration_seconds(start)
        if start_sec is None:
            continue
        chapters.append({"title": str(title).strip(), "start_time": start_sec})
    return chapters


def parse_duration_seconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        if ":" in stripped:
            parts = [int(part) for part in stripped.split(":") if part.isdigit()]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def parse_metadata(metadata_payload: dict[str, Any], video_id: str) -> dict[str, Any]:
    if "search_metadata" in metadata_payload and "results" not in metadata_payload:
        search_metadata = metadata_payload.get("search_metadata", {})
        channel = metadata_payload.get("channel")
        channel_name = channel.get("name") if isinstance(channel, dict) else channel
        title = metadata_payload.get("title") or f"Podcast episode {video_id}"
        return {
            "title": title,
            "channel": channel_name or "Unknown channel",
            "duration_sec": parse_duration_seconds(metadata_payload.get("duration") or metadata_payload.get("length")),
            "language": metadata_payload.get("language") or metadata_payload.get("search_parameters", {}).get("hl") or "unknown",
            "url": search_metadata.get("youtube_video_url") or f"https://www.youtube.com/watch?v={video_id}",
            "chapters": _metadata_chapters(metadata_payload.get("chapters")),
        }

    results = metadata_payload.get("results", [])
    first = results[0] if results else {}
    content = first.get("content", {})
    parsed = content.get("results", {}) if isinstance(content, dict) else {}
    title = parsed.get("title") or f"Podcast episode {video_id}"
    channel = parsed.get("uploader") or parsed.get("channel") or "Unknown channel"
    duration = parse_duration_seconds(parsed.get("duration"))
    language = parsed.get("language") or "unknown"
    url = f"https://www.youtube.com/watch?v={video_id}"
    chapters = _metadata_chapters(parsed.get("chapters"))
    return {
        "title": title,
        "channel": channel,
        "duration_sec": duration,
        "language": language,
        "url": url,
        "chapters": chapters,
    }


def _flatten_search_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    serpapi_results: list[dict[str, Any]] = []
    for key in ("video_results", "ads_results"):
        value = payload.get(key)
        if isinstance(value, list):
            serpapi_results.extend(item for item in value if isinstance(item, dict))
    if serpapi_results:
        return serpapi_results

    results = payload.get("results", [])
    if not results:
        return []
    content = results[0].get("content", {})
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            return []
        return [item for item in decoded if isinstance(item, dict)]
    if isinstance(content, dict) and isinstance(content.get("results"), list):
        return [item for item in content["results"] if isinstance(item, dict)]
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    return []


def _nested_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        if "simpleText" in value:
            text = str(value.get("simpleText", "")).strip()
            return text or None
        runs = value.get("runs")
        if isinstance(runs, list):
            text = "".join(str(run.get("text", "")) for run in runs).strip()
            return text or None
    return None


def _channel_name(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        name = value.get("name") or value.get("title")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def search_candidates(payload: dict[str, Any], query: str) -> list[VideoCandidate]:
    query_terms = {token for token in slugify(query).split("-") if token}
    candidates: list[VideoCandidate] = []
    for fallback_position, item in enumerate(_flatten_search_items(payload), start=1):
        navigation = item.get("navigationEndpoint", {})
        watch = navigation.get("watchEndpoint", {}) if isinstance(navigation, dict) else {}
        video_id = item.get("videoId") or item.get("video_id") or item.get("id") or watch.get("videoId")
        if not video_id and isinstance(item.get("link"), str):
            video_id = extract_video_id(item["link"])
        if not video_id:
            continue
        title = _nested_text(item.get("title")) or item.get("name") or f"Video {video_id}"
        channel = (
            _channel_name(item.get("channel"))
            or _channel_name(item.get("uploader"))
            or _nested_text(item.get("ownerText"))
            or _nested_text(item.get("shortBylineText"))
            or _nested_text(item.get("longBylineText"))
            or "Unknown channel"
        )
        url = item.get("url") or item.get("link") or f"https://www.youtube.com/watch?v={video_id}"
        duration_sec = parse_duration_seconds(
            item.get("durationSeconds") or item.get("duration") or item.get("length") or _nested_text(item.get("lengthText"))
        )
        title_terms = {token for token in slugify(title).split("-") if token}
        channel_terms = {token for token in slugify(channel).split("-") if token}
        title_overlap = len(query_terms & title_terms)
        channel_overlap = len(query_terms & channel_terms)
        duration_bonus = min((duration_sec or 0) / 3600.0, 2.0)
        position = item.get("position_on_page") or fallback_position
        position_bonus = max(0.0, (10.0 - float(position)) * 0.15) if isinstance(position, (int, float)) else 0.0
        score = title_overlap * 1.5 + channel_overlap * 2.5 + duration_bonus + position_bonus
        candidates.append(
            VideoCandidate(
                video_id=video_id,
                title=title,
                channel=channel,
                url=url,
                duration_sec=duration_sec,
                score=score,
                transcript_available=False,
                subtitles_available=False,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def resolve_single_video(
    raw_input: str,
    client: Any,
    *,
    language_code: str,
    search_source: str,
    search_limit: int,
) -> ResolvedVideo:
    input_type = detect_input_type(raw_input)
    if input_type in {"youtube_url", "video_id"}:
        video_id = extract_video_id(raw_input) or raw_input.strip()
        probe = client.fetch_best_timed_content(video_id, language_code=language_code)
        parsed = parse_metadata(probe.metadata, video_id)
        candidate = VideoCandidate(
            video_id=video_id,
            title=parsed["title"],
            channel=parsed["channel"],
            url=parsed["url"],
            duration_sec=parsed["duration_sec"],
            score=1.0,
            transcript_available=probe.source_kind == "transcript",
            subtitles_available=probe.source_kind == "subtitles",
        )
        return ResolvedVideo(candidate, probe.metadata, probe.content_payload, probe.source_kind, getattr(probe, "origin", None))

    search_payload = client.search(raw_input, source=search_source, subtitles=True)
    candidates = search_candidates(search_payload, raw_input)[:search_limit]
    if not candidates:
        raise OxylabsError(f"No YouTube candidates were found for query: {raw_input}")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            probe = client.fetch_best_timed_content(candidate.video_id, language_code=language_code)
            parsed = parse_metadata(probe.metadata, candidate.video_id)
            candidate.title = parsed["title"]
            candidate.channel = parsed["channel"]
            candidate.url = parsed["url"]
            candidate.duration_sec = parsed["duration_sec"]
            candidate.transcript_available = probe.source_kind == "transcript"
            candidate.subtitles_available = probe.source_kind == "subtitles"
            return ResolvedVideo(candidate, probe.metadata, probe.content_payload, probe.source_kind, getattr(probe, "origin", None))
        except Exception as exc:
            last_error = exc
            continue
    raise OxylabsError(
        f"Search results were found for '{raw_input}', but none produced usable transcript or subtitle content."
    ) from last_error


def load_template(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8")


def _runtime_client(provider: str, client: Optional[Any], credential_root: Path) -> Any:
    load_local_env(credential_root)
    if client is not None:
        return client
    if provider not in {"auto", "serpapi", "oxylabs"}:
        raise ValueError("provider must be one of: auto, serpapi, oxylabs")
    has_serpapi_key = resolve_setting(("SERPAPI_API_KEY", "SERPAPI_KEY"), start_path=credential_root) is not None
    if provider == "serpapi" or (provider == "auto" and has_serpapi_key):
        return SerpApiClient(parse_serpapi_key(credential_root))
    username, password = parse_credentials(credential_root)
    return OxylabsClient(username, password)


def _segment_end_sec(segment: Segment) -> int:
    return int(segment.end_sec if segment.end_sec is not None else segment.start_sec)


def _segment_words(segments: list[Segment]) -> int:
    return sum(len(segment.text.split()) for segment in segments)


def _chapter_for_segment(chapters: list[dict[str, Any]], start_sec: int) -> Optional[str]:
    active: Optional[str] = None
    for chapter in sorted(chapters, key=lambda item: int(item["start_time"])):
        if start_sec < int(chapter["start_time"]):
            break
        active = str(chapter["title"])
    return active


def _chapter_context(
    segments: list[Segment],
    chapters: list[dict[str, Any]],
    video_id: str,
) -> list[dict[str, Any]]:
    if not chapters:
        return []
    normalized = sorted(chapters, key=lambda item: int(item["start_time"]))
    chapter_payloads: list[dict[str, Any]] = []
    for index, chapter in enumerate(normalized):
        start_sec = int(chapter["start_time"])
        end_sec = int(normalized[index + 1]["start_time"]) if index + 1 < len(normalized) else None
        matching = [
            segment
            for segment in segments
            if segment.start_sec >= start_sec and (end_sec is None or segment.start_sec < end_sec)
        ]
        if not matching:
            continue
        chapter_end = _segment_end_sec(matching[-1])
        chapter_payloads.append(
            {
                "index": index,
                "title": str(chapter["title"]),
                "start_sec": start_sec,
                "end_sec": chapter_end,
                "timestamp": format_timestamp(start_sec),
                "url": f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s",
                "segment_count": len(matching),
                "word_count": _segment_words(matching),
                "text": "\n".join(segment.text.strip() for segment in matching if segment.text.strip()),
            }
        )
    return chapter_payloads


def _segments_payload(segments: list[Segment], chapters: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "start_sec": int(segment.start_sec),
            "end_sec": _segment_end_sec(segment),
            "timestamp": format_timestamp(segment.start_sec),
            "url": f"https://www.youtube.com/watch?v={video_id}&t={int(segment.start_sec)}s",
            "chapter": segment.label or _chapter_for_segment(chapters, int(segment.start_sec)),
            "text": segment.text,
        }
        for index, segment in enumerate(segments)
    ]


def _coverage_payload(segments: list[Segment], duration_sec: Optional[int]) -> dict[str, Any]:
    first_start = min((int(segment.start_sec) for segment in segments), default=0)
    last_end = max((_segment_end_sec(segment) for segment in segments), default=0)
    span_sec = max(last_end - first_start, 0)
    payload: dict[str, Any] = {
        "segments_count": len(segments),
        "words_count": _segment_words(segments),
        "first_start_sec": first_start,
        "last_end_sec": last_end,
        "span_sec": span_sec,
        "span_timestamp": format_timestamp(span_sec),
    }
    if duration_sec:
        payload["duration_sec"] = duration_sec
        payload["coverage_ratio"] = round(last_end / duration_sec, 4)
    return payload


def fetch_transcript_context(
    raw_input: str,
    *,
    output_dir: Path,
    language_code: str = "en",
    search_source: str = "youtube_search",
    search_limit: int = 5,
    provider: str = "serpapi",
    client: Optional[Any] = None,
) -> Path:
    credential_root = Path.cwd()
    runtime_client = _runtime_client(provider, client, credential_root)
    resolved = resolve_single_video(
        raw_input,
        runtime_client,
        language_code=language_code,
        search_source=search_source,
        search_limit=search_limit,
    )
    metadata = parse_metadata(resolved.metadata_payload, resolved.candidate.video_id)
    segments = normalize_timed_content(
        resolved.content_payload,
        video_id=resolved.candidate.video_id,
        source_kind=resolved.source_kind,
        language=metadata["language"],
    )
    segments = merge_timed_segments(segments)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{slugify(metadata['title'], fallback=resolved.candidate.video_id)}.transcript.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": raw_input,
        "provider": provider,
        "source_kind": resolved.source_kind,
        "origin": resolved.origin,
        "video": {
            "video_id": resolved.candidate.video_id,
            "title": metadata["title"],
            "channel": metadata["channel"],
            "duration_sec": metadata["duration_sec"],
            "language": metadata["language"],
            "url": metadata["url"],
            "chapters": metadata.get("chapters", []),
        },
        "coverage": _coverage_payload(segments, metadata["duration_sec"]),
        "chapters": _chapter_context(segments, metadata.get("chapters", []), resolved.candidate.video_id),
        "segments": _segments_payload(segments, metadata.get("chapters", []), resolved.candidate.video_id),
        "agent_instructions": {
            "purpose": "Use this complete transcript context as source material for writing the article.",
            "do_not_treat_as_article": True,
            "preserve_timestamp_links": True,
        },
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def build_article(
    raw_input: str,
    *,
    output_dir: Path,
    language_code: str = "en",
    mode: str = "single",
    search_source: str = "youtube_search",
    search_limit: int = 5,
    provider: str = "serpapi",
    template_path: Path = DEFAULT_TEMPLATE,
    client: Optional[Any] = None,
) -> Path:
    if mode != "single":
        raise NotImplementedError(
            "Aggregation mode is reserved for explicit multi-source requests and is not implemented in v1."
        )

    credential_root = Path.cwd()
    runtime_client = _runtime_client(provider, client, credential_root)

    resolved = resolve_single_video(
        raw_input,
        runtime_client,
        language_code=language_code,
        search_source=search_source,
        search_limit=search_limit,
    )
    metadata = parse_metadata(resolved.metadata_payload, resolved.candidate.video_id)
    segments = normalize_timed_content(
        resolved.content_payload,
        video_id=resolved.candidate.video_id,
        source_kind=resolved.source_kind,
        language=metadata["language"],
    )
    segments = merge_timed_segments(segments)
    sections: list[ArticleSection] = build_outline_sections(segments, chapters=metadata.get("chapters"))
    article_title = f"{metadata['title']} - Article"
    output_dir.mkdir(parents=True, exist_ok=True)
    template_text = load_template(template_path)
    markdown = render_article_markdown(
        title=article_title,
        source_title=metadata["title"],
        channel=metadata["channel"],
        video_url=metadata["url"],
        language=metadata["language"],
        sections=sections,
        template_text=template_text,
    )
    destination = output_dir / f"{slugify(metadata['title'])}.md"
    destination.write_text(markdown, encoding="utf-8")
    return destination


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a YouTube podcast URL, video ID, or search query into a Markdown article."
    )
    parser.add_argument("input", help="A YouTube URL, a video ID, or a search query.")
    parser.add_argument("--output-dir", default="articles", help="Directory where the Markdown article should be saved.")
    parser.add_argument("--language-code", default="en", help="Preferred transcript/subtitle language code.")
    parser.add_argument(
        "--provider",
        choices=["auto", "serpapi", "oxylabs"],
        default="serpapi",
        help="API provider. SerpApi is the default; use oxylabs to force the legacy path.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "aggregate"],
        default="single",
        help="Content generation mode. Aggregation mode is intentionally deferred in v1.",
    )
    parser.add_argument(
        "--search-source",
        choices=["youtube_search", "youtube_search_max"],
        default="youtube_search",
        help="Oxylabs search source to use for query-based resolution. Ignored by SerpApi.",
    )
    parser.add_argument("--search-limit", type=int, default=5, help="Maximum number of ranked search candidates to probe.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output_path = build_article(
            args.input,
            output_dir=(REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir),
            language_code=args.language_code,
            mode=args.mode,
            search_source=args.search_source,
            search_limit=args.search_limit,
            provider=args.provider,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
