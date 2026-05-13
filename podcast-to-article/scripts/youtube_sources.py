from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from normalize import Segment, merge_timed_segments, normalize_timed_content
    from serpapi_client import SerpApiClient, SerpApiError
    from utils import detect_input_type, extract_video_id, format_timestamp, load_local_env, parse_serpapi_key, slugify
else:
    from .normalize import Segment, merge_timed_segments, normalize_timed_content
    from .serpapi_client import SerpApiClient, SerpApiError
    from .utils import detect_input_type, extract_video_id, format_timestamp, load_local_env, parse_serpapi_key, slugify


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
SEARCH_QUERY_HASH_LENGTH = 8
SEARCH_MANIFEST_FILENAME = "search-manifest.json"
LATIN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}
PREFERRED_SOURCE_TERMS = {
    "conversation",
    "discussion",
    "fireside",
    "interview",
    "keynote",
    "lecture",
    "panel",
    "podcast",
    "talk",
    "对谈",
    "访谈",
    "采访",
    "播客",
    "演讲",
    "圆桌",
    "专访",
}
NOISY_SOURCE_TERMS = {
    "clip",
    "clips",
    "highlight",
    "highlights",
    "reaction",
    "reacts",
    "short",
    "shorts",
    "teaser",
    "trailer",
}
DIRECT_SOURCE_TERMS = {
    "conversation",
    "dialogue",
    "interview",
    "keynote",
    "speech",
    "talk",
    "with",
    "对话",
    "访谈",
    "采访",
    "演讲",
    "专访",
}
THIRD_PARTY_ANALYSIS_TERMS = {
    "analysis",
    "analyst",
    "battle",
    "breakthrough",
    "documentary",
    "explained",
    "explains",
    "race",
    "rivalry",
    "war",
    "winning",
    "解读",
    "分析",
    "纪录片",
    "赶超",
    "竞争",
}
CHINA_TOPIC_TERMS = {"china", "chinese", "中国", "国产", "华人"}
LEADER_TOPIC_TERMS = {
    "boss",
    "ceo",
    "entrepreneur",
    "founder",
    "leader",
    "leaders",
    "大佬",
    "企业家",
    "创始人",
    "领袖",
}
CHINESE_LEADER_SIGNAL_TERMS = {
    "01",
    "alibaba",
    "baichuan",
    "baidu",
    "bytedance",
    "deepseek",
    "huawei",
    "kimi",
    "minimax",
    "moonshot",
    "qwen",
    "sensetime",
    "tencent",
    "tsai",
    "wang",
    "xiaochuan",
    "zhipu",
    "zhang",
    "阿里",
    "百度",
    "百川",
    "蔡崇信",
    "大模型",
    "华为",
    "李开复",
    "梁建章",
    "梁文锋",
    "零一",
    "商汤",
    "深度求索",
    "腾讯",
    "王小川",
    "月之暗面",
    "张亚勤",
    "智谱",
    "周鸿祎",
    "字节",
}


@dataclass
class VideoCandidate:
    video_id: str
    title: str
    channel: str
    url: str
    duration_sec: Optional[int]
    score: float
    transcript_available: bool
    subtitles_available: bool
    views: Optional[int] = None
    published_date: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ResolvedVideo:
    candidate: VideoCandidate
    metadata_payload: dict[str, Any]
    content_payload: dict[str, Any]
    source_kind: str
    origin: Optional[str] = None


def build_runtime_client(client: Optional[Any], credential_root: Path) -> Any:
    load_local_env(credential_root)
    if client is not None:
        return client
    return SerpApiClient(parse_serpapi_key(credential_root))


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
        start_sec = parse_duration_seconds(start)
        if start_sec is not None:
            chapters.append({"title": str(title).strip(), "start_time": start_sec})
    return chapters


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
    return {
        "title": title,
        "channel": parsed.get("uploader") or parsed.get("channel") or "Unknown channel",
        "duration_sec": parse_duration_seconds(parsed.get("duration")),
        "language": parsed.get("language") or "unknown",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "chapters": _metadata_chapters(parsed.get("chapters")),
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
        except (OSError, json.JSONDecodeError):
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


def _text(value: Any) -> str:
    nested = _nested_text(value)
    if nested is not None:
        return nested
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip() if isinstance(value, str) else ""


def _search_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(r"[a-z0-9]+|[\u3400-\u9fff]+", value.lower()):
        text = match.group(0)
        if re.fullmatch(r"[a-z0-9]+", text):
            if len(text) > 1 and text not in LATIN_STOPWORDS:
                tokens.add(text)
            continue
        if len(text) <= 4:
            tokens.add(text)
        for size in (2, 3, 4):
            if len(text) >= size:
                tokens.update(text[index : index + size] for index in range(len(text) - size + 1))
    return tokens


def _term_overlap_score(query_terms: set[str], value: str, weight: float) -> float:
    if not query_terms or not value:
        return 0.0
    return len(query_terms & _search_tokens(value)) * weight


def _parse_views(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None

    stripped = value.strip().lower().replace(",", "")
    multiplier = 1.0
    if "亿" in stripped:
        multiplier = 100_000_000.0
    elif "万" in stripped:
        multiplier = 10_000.0
    elif "b" in stripped:
        multiplier = 1_000_000_000.0
    elif "m" in stripped:
        multiplier = 1_000_000.0
    elif "k" in stripped:
        multiplier = 1_000.0

    match = re.search(r"(\d+(?:\.\d+)?)", stripped)
    return int(float(match.group(1)) * multiplier) if match else None


def _views_bonus(views: Optional[int]) -> float:
    if not views or views <= 0:
        return 0.0
    return min(math.log10(views + 1) * 0.22, 1.5)


def _duration_bonus(duration_sec: Optional[int]) -> float:
    if duration_sec is None:
        return 0.0
    if duration_sec < 180:
        return -1.6
    if duration_sec < 600:
        return -0.7
    if duration_sec <= 7200:
        return min(duration_sec / 3600.0, 1.6)
    return 0.9


def _source_format_bonus(title: str, description: str, query_terms: set[str]) -> float:
    searchable = f"{title} {description}".lower()
    terms = _search_tokens(searchable)
    bonus = 0.8 if terms & PREFERRED_SOURCE_TERMS else 0.0
    noisy_terms = terms & NOISY_SOURCE_TERMS
    if noisy_terms and not noisy_terms & query_terms:
        bonus -= 1.2
    return bonus


def _direct_leader_source_bonus(title: str, channel: str, description: str, query_terms: set[str]) -> float:
    terms = _search_tokens(f"{title} {channel} {description}")
    direct_query_terms = LEADER_TOPIC_TERMS | DIRECT_SOURCE_TERMS | PREFERRED_SOURCE_TERMS
    if not query_terms & CHINA_TOPIC_TERMS or not query_terms & direct_query_terms:
        return 0.0

    bonus = 0.0
    leader_signal = terms & CHINESE_LEADER_SIGNAL_TERMS
    if terms & DIRECT_SOURCE_TERMS:
        bonus += 1.4
    if leader_signal:
        bonus += 1.2
    if terms & LEADER_TOPIC_TERMS:
        bonus += 0.8

    third_party_terms = terms & THIRD_PARTY_ANALYSIS_TERMS
    if not leader_signal and not terms & DIRECT_SOURCE_TERMS and not terms & LEADER_TOPIC_TERMS:
        bonus -= 1.0
    if third_party_terms and not terms & DIRECT_SOURCE_TERMS and not leader_signal:
        bonus -= 2.2
    if third_party_terms and not terms & LEADER_TOPIC_TERMS and not leader_signal:
        bonus -= 0.8
    return bonus


def _candidate_score(
    *,
    query_terms: set[str],
    title: str,
    channel: str,
    description: str,
    duration_sec: Optional[int],
    views: Optional[int],
    position: Any,
) -> float:
    position_bonus = max(0.0, (10.0 - float(position)) * 0.15) if isinstance(position, (int, float)) else 0.0
    score = position_bonus
    score += _term_overlap_score(query_terms, title, 2.0)
    score += _term_overlap_score(query_terms, channel, 2.3)
    score += _term_overlap_score(query_terms, description, 0.55)
    score += _duration_bonus(duration_sec)
    score += _views_bonus(views)
    score += _source_format_bonus(title, description, query_terms)
    score += _direct_leader_source_bonus(title, channel, description, query_terms)
    return round(score, 4)


def search_candidates(payload: dict[str, Any], query: str) -> list[VideoCandidate]:
    query_terms = _search_tokens(query)
    candidates: list[VideoCandidate] = []
    for fallback_position, item in enumerate(_flatten_search_items(payload), start=1):
        navigation = item.get("navigationEndpoint", {})
        watch = navigation.get("watchEndpoint", {}) if isinstance(navigation, dict) else {}
        video_id = item.get("videoId") or item.get("video_id") or item.get("id") or watch.get("videoId")
        if not video_id and isinstance(item.get("link"), str):
            video_id = extract_video_id(item["link"])
        if not video_id:
            continue
        title = _nested_text(item.get("title")) or _text(item.get("name")) or f"Video {video_id}"
        channel = (
            _channel_name(item.get("channel"))
            or _channel_name(item.get("uploader"))
            or _nested_text(item.get("ownerText"))
            or _nested_text(item.get("shortBylineText"))
            or _nested_text(item.get("longBylineText"))
            or "Unknown channel"
        )
        description = _text(item.get("description")) or _text(item.get("snippet"))
        duration_sec = parse_duration_seconds(
            item.get("durationSeconds") or item.get("duration") or item.get("length") or _nested_text(item.get("lengthText"))
        )
        position = item.get("position_on_page") or fallback_position
        views = _parse_views(item.get("views") or item.get("views_count") or item.get("view_count"))
        candidates.append(
            VideoCandidate(
                video_id=video_id,
                title=title,
                channel=channel,
                url=item.get("url") or item.get("link") or f"https://www.youtube.com/watch?v={video_id}",
                duration_sec=duration_sec,
                score=_candidate_score(
                    query_terms=query_terms,
                    title=title,
                    channel=channel,
                    description=description,
                    duration_sec=duration_sec,
                    views=views,
                    position=position,
                ),
                transcript_available=False,
                subtitles_available=False,
                views=views,
                published_date=_text(item.get("published_date") or item.get("published_time")),
                description=description,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def resolve_single_video(
    raw_input: str,
    client: Any,
    *,
    language_code: str,
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

    search_payload = client.search(raw_input)
    candidates = search_candidates(search_payload, raw_input)[:search_limit]
    if not candidates:
        raise SerpApiError(f"No YouTube candidates were found for query: {raw_input}")

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
    raise SerpApiError(f"Search results were found for '{raw_input}', but none produced usable transcript or subtitle content.") from last_error


def search_youtube_context(
    query: str,
    *,
    output_dir: Path,
    run_id: str | None = None,
    client: Optional[Any] = None,
) -> Path:
    runtime_client = build_runtime_client(client, Path.cwd())
    payload = runtime_client.search(query)
    candidates = search_candidates(payload, query)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_query = canonicalize_search_query(query)
    query_hash = hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()[:SEARCH_QUERY_HASH_LENGTH]
    destination = unique_search_output_path(canonical_query, output_dir=output_dir, query_hash=query_hash)
    raw_destination = raw_search_output_path(destination)
    generated_at = datetime.now(timezone.utc).isoformat()
    raw_destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": generated_at,
                "run_id": run_id,
                "query": query,
                "canonical_query": canonical_query,
                "query_hash": query_hash,
                "provider": "serpapi",
                "payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": generated_at,
                "run_id": run_id,
                "query": query,
                "canonical_query": canonical_query,
                "query_hash": query_hash,
                "provider": "serpapi",
                "raw_output_path": str(raw_destination),
                "candidates": [candidate.__dict__ for candidate in candidates],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    append_search_manifest(
        output_dir,
        {
            "created_at": generated_at,
            "run_id": run_id,
            "query": query,
            "canonical_query": canonical_query,
            "query_hash": query_hash,
            "provider": "serpapi",
            "output_path": str(destination),
            "raw_output_path": str(raw_destination),
            "candidate_count": len(candidates),
            "top_video_ids": [candidate.video_id for candidate in candidates[:5]],
        },
    )
    return destination


def canonicalize_search_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).casefold()


def unique_search_output_path(query: str, *, output_dir: Path, query_hash: str) -> Path:
    stem = f"{slugify(query, fallback='youtube-search')}-{query_hash}"
    candidate = output_dir / f"{stem}.search.json"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = output_dir / f"{stem}-{index}.search.json"
        if not candidate.exists():
            return candidate
        index += 1


def raw_search_output_path(search_output_path: Path) -> Path:
    name = search_output_path.name
    if name.endswith(".search.json"):
        return search_output_path.with_name(f"{name.removesuffix('.search.json')}.raw-search.json")
    return search_output_path.with_suffix(".raw-search.json")


def append_search_manifest(output_dir: Path, entry: dict[str, Any]) -> Path:
    manifest_path = output_dir / SEARCH_MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}

    searches = manifest.get("searches")
    if not isinstance(searches, list):
        searches = []

    now = datetime.now(timezone.utc).isoformat()
    searches.append({"round": len(searches) + 1, **entry})
    manifest.update(
        {
            "schema_version": 1,
            "generated_at": manifest.get("generated_at") or now,
            "updated_at": now,
            "search_dir": str(output_dir),
            "search_count": len(searches),
            "searches": searches,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


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


def _chapter_context(segments: list[Segment], chapters: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    normalized = sorted(chapters, key=lambda item: int(item["start_time"]))
    chapter_payloads: list[dict[str, Any]] = []
    for index, chapter in enumerate(normalized):
        start_sec = int(chapter["start_time"])
        end_sec = int(normalized[index + 1]["start_time"]) if index + 1 < len(normalized) else None
        matching = [segment for segment in segments if segment.start_sec >= start_sec and (end_sec is None or segment.start_sec < end_sec)]
        if not matching:
            continue
        chapter_payloads.append(
            {
                "index": index,
                "title": str(chapter["title"]),
                "start_sec": start_sec,
                "end_sec": _segment_end_sec(matching[-1]),
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
    search_limit: int = 5,
    run_id: str | None = None,
    client: Optional[Any] = None,
) -> Path:
    runtime_client = build_runtime_client(client, Path.cwd())
    resolved = resolve_single_video(
        raw_input,
        runtime_client,
        language_code=language_code,
        search_limit=search_limit,
    )
    metadata = parse_metadata(resolved.metadata_payload, resolved.candidate.video_id)
    segments = merge_timed_segments(
        normalize_timed_content(
            resolved.content_payload,
            video_id=resolved.candidate.video_id,
            source_kind=resolved.source_kind,
            language=metadata["language"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{slugify(metadata['title'], fallback=resolved.candidate.video_id)}.transcript.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "input": raw_input,
        "provider": "serpapi",
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
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
