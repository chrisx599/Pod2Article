from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

if __package__ in {None, ""}:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from aliyun_asr_client import AliyunAsrClient, aliyun_asr_is_enabled, aliyun_asr_is_preferred
    from normalize import Segment, merge_timed_segments, normalize_timed_content
    from serpapi_client import SerpApiClient, SerpApiError
    from utils import detect_input_type, extract_video_id, format_timestamp, load_local_env, parse_serpapi_key, slugify
else:
    from .aliyun_asr_client import AliyunAsrClient, aliyun_asr_is_enabled, aliyun_asr_is_preferred
    from .normalize import Segment, merge_timed_segments, normalize_timed_content
    from .serpapi_client import SerpApiClient, SerpApiError
    from .utils import detect_input_type, extract_video_id, format_timestamp, load_local_env, parse_serpapi_key, slugify


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
SEARCH_QUERY_HASH_LENGTH = 8
SEARCH_MANIFEST_FILENAME = "search-manifest.json"
WEB_SEARCH_MANIFEST_FILENAME = "web-search-manifest.json"
RESEARCH_PLAN_FILENAME = "research-plan.json"
VIDEO_ENRICHMENT_MANIFEST_FILENAME = "video-enrichment-manifest.json"
SELECTION_MANIFEST_FILENAME = "selection-manifest.json"
DEFAULT_DISCOVERY_QUERY_COUNT = 4
DEFAULT_DISCOVERY_ENRICHMENT_LIMIT = 14
DEFAULT_SELECTION_CANDIDATE_LIMIT = 18
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
    source_bucket: str = "video_results"
    discovery_source: Optional[str] = None
    score_breakdown: Optional[dict[str, float]] = None


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


def build_aliyun_asr_client(asr_client: Optional[Any], credential_root: Path) -> Optional[Any]:
    load_local_env(credential_root)
    if asr_client is not None:
        return asr_client
    if not aliyun_asr_is_enabled(credential_root):
        return None
    return AliyunAsrClient.from_environment(credential_root)


def _fetch_aliyun_asr_probe(video_id: str, metadata: dict[str, Any], asr_client: Any) -> Any:
    payload = asr_client.transcribe_youtube_video(video_id)
    return type(
        "Probe",
        (),
        {
            "metadata": metadata,
            "content_payload": payload,
            "source_kind": "asr",
            "origin": "aliyun_asr",
        },
    )()


def fetch_best_timed_content_with_fallback(
    video_id: str,
    client: Any,
    *,
    language_code: str,
    asr_client: Optional[Any],
    credential_root: Path,
) -> Any:
    fallback_client = build_aliyun_asr_client(asr_client, credential_root)
    if fallback_client is not None and aliyun_asr_is_preferred(credential_root):
        metadata = client.metadata(video_id)
        return _fetch_aliyun_asr_probe(video_id, metadata, fallback_client)

    try:
        return client.fetch_best_timed_content(video_id, language_code=language_code)
    except Exception as primary_error:
        if fallback_client is None:
            raise
        try:
            metadata = client.metadata(video_id)
            return _fetch_aliyun_asr_probe(video_id, metadata, fallback_client)
        except Exception as fallback_error:
            raise SerpApiError(
                f"Unable to retrieve transcript for video {video_id}; "
                f"Aliyun ASR fallback also failed: {fallback_error}"
            ) from primary_error


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


def _with_source_bucket(item: dict[str, Any], bucket: str) -> dict[str, Any]:
    copied = dict(item)
    copied["_source_bucket"] = bucket
    return copied


def _flatten_search_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    serpapi_results: list[dict[str, Any]] = []
    for key in ("video_results", "shorts_results", "short_videos", "ads_results"):
        value = payload.get(key)
        if isinstance(value, list):
            serpapi_results.extend(_with_source_bucket(item, key) for item in value if isinstance(item, dict))
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
        return [_with_source_bucket(item, "legacy_results") for item in decoded if isinstance(item, dict)]
    if isinstance(content, dict) and isinstance(content.get("results"), list):
        return [_with_source_bucket(item, "legacy_results") for item in content["results"] if isinstance(item, dict)]
    if isinstance(content, list):
        return [_with_source_bucket(item, "legacy_results") for item in content if isinstance(item, dict)]
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
            if len(text) > 1:
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


def _candidate_score_breakdown(
    *,
    query_terms: set[str],
    title: str,
    channel: str,
    description: str,
    duration_sec: Optional[int],
    views: Optional[int],
    position: Any,
) -> dict[str, float]:
    position_bonus = max(0.0, (10.0 - float(position)) * 0.15) if isinstance(position, (int, float)) else 0.0
    return {
        "position_bonus": round(position_bonus, 4),
        "title_match": round(_term_overlap_score(query_terms, title, 2.0), 4),
        "channel_match": round(_term_overlap_score(query_terms, channel, 2.3), 4),
        "description_match": round(_term_overlap_score(query_terms, description, 0.55), 4),
        "duration_bonus": round(_duration_bonus(duration_sec), 4),
        "views_bonus": round(_views_bonus(views), 4),
    }


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
    return round(
        sum(
            _candidate_score_breakdown(
                query_terms=query_terms,
                title=title,
                channel=channel,
                description=description,
                duration_sec=duration_sec,
                views=views,
                position=position,
            ).values()
        ),
        4,
    )


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
        score_breakdown = _candidate_score_breakdown(
            query_terms=query_terms,
            title=title,
            channel=channel,
            description=description,
            duration_sec=duration_sec,
            views=views,
            position=position,
        )
        candidates.append(
            VideoCandidate(
                video_id=video_id,
                title=title,
                channel=channel,
                url=item.get("url") or item.get("link") or f"https://www.youtube.com/watch?v={video_id}",
                duration_sec=duration_sec,
                score=round(sum(score_breakdown.values()), 4),
                transcript_available=False,
                subtitles_available=False,
                views=views,
                published_date=_text(item.get("published_date") or item.get("published_time")),
                description=description,
                source_bucket=str(item.get("_source_bucket") or "video_results"),
                score_breakdown=score_breakdown,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _candidate_payload(candidate: VideoCandidate) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.__dict__.items()
        if value is not None
    }


def extract_related_search_queries(payload: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for key in ("related_searches", "search_refinements", "refine_this_search"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                query = item
            elif isinstance(item, dict):
                query = _text(item.get("query") or item.get("title") or item.get("name"))
            else:
                query = ""
            query = canonicalize_search_query(query)
            if query and query not in queries:
                queries.append(query)
    return queries


def _transcript_link(payload: dict[str, Any]) -> Optional[str]:
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str) and "youtube_video_transcript" in value:
                    return value
                if "transcript" in key.lower() and isinstance(value, str) and value.startswith(("http://", "https://")):
                    return value
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(value for value in item if isinstance(value, (dict, list)))
    return None


def _extract_video_candidates_from_items(
    items: Any,
    *,
    source_bucket: str,
    discovery_source: str,
    query_terms: set[str],
) -> list[VideoCandidate]:
    if not isinstance(items, list):
        return []
    candidates: list[VideoCandidate] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        video_id = (
            item.get("video_id")
            or item.get("videoId")
            or item.get("id")
            or extract_video_id(str(item.get("link") or item.get("url") or ""))
        )
        if not isinstance(video_id, str) or not video_id:
            continue
        title = _nested_text(item.get("title")) or _text(item.get("name")) or f"Video {video_id}"
        channel = _channel_name(item.get("channel")) or _channel_name(item.get("uploader")) or _text(item.get("author")) or "Unknown channel"
        description = _text(item.get("description") or item.get("snippet"))
        duration_sec = parse_duration_seconds(item.get("duration") or item.get("length") or item.get("durationSeconds"))
        views = _parse_views(item.get("views") or item.get("views_count") or item.get("view_count"))
        breakdown = _candidate_score_breakdown(
            query_terms=query_terms,
            title=title,
            channel=channel,
            description=description,
            duration_sec=duration_sec,
            views=views,
            position=index,
        )
        candidates.append(
            VideoCandidate(
                video_id=video_id,
                title=title,
                channel=channel,
                url=str(item.get("url") or item.get("link") or f"https://www.youtube.com/watch?v={video_id}"),
                duration_sec=duration_sec,
                score=round(sum(breakdown.values()) + 0.7, 4),
                transcript_available=False,
                subtitles_available=False,
                views=views,
                published_date=_text(item.get("published_date") or item.get("published_time")),
                description=description,
                source_bucket=source_bucket,
                discovery_source=discovery_source,
                score_breakdown={**breakdown, "related_video_bonus": 0.7},
            )
        )
    return candidates


def normalize_video_enrichment(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = parse_metadata(payload, video_id)
    query_terms = _search_tokens(f"{metadata.get('title', '')} {metadata.get('channel', '')}")
    related_candidates: list[VideoCandidate] = []
    for key in ("related_videos", "related_video_results", "end_screen_video_results"):
        related_candidates.extend(
            _extract_video_candidates_from_items(
                payload.get(key),
                source_bucket=key,
                discovery_source=video_id,
                query_terms=query_terms,
            )
        )
    return {
        "video_id": video_id,
        "title": metadata.get("title"),
        "channel": metadata.get("channel"),
        "duration_sec": metadata.get("duration_sec"),
        "language": metadata.get("language"),
        "url": metadata.get("url"),
        "views": _parse_views(payload.get("views") or payload.get("view_count")),
        "description": _text(payload.get("description")),
        "chapters_count": len(metadata.get("chapters") or []),
        "has_transcript_link": _transcript_link(payload) is not None,
        "transcript_link": _transcript_link(payload),
        "related_video_ids": [candidate.video_id for candidate in related_candidates],
        "related_candidates": [_candidate_payload(candidate) for candidate in related_candidates],
    }


def _merge_candidates(candidates: Iterable[VideoCandidate]) -> list[VideoCandidate]:
    merged: dict[str, VideoCandidate] = {}
    for candidate in candidates:
        existing = merged.get(candidate.video_id)
        if existing is None or candidate.score > existing.score:
            merged[candidate.video_id] = candidate
            continue
        if not existing.description and candidate.description:
            existing.description = candidate.description
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def _candidate_from_payload(payload: dict[str, Any]) -> Optional[VideoCandidate]:
    video_id = payload.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return None
    score = payload.get("score")
    score_value = float(score) if isinstance(score, (int, float)) else 0.0
    score_breakdown = payload.get("score_breakdown")
    return VideoCandidate(
        video_id=video_id,
        title=str(payload.get("title") or f"Video {video_id}"),
        channel=str(payload.get("channel") or "Unknown channel"),
        url=str(payload.get("url") or f"https://www.youtube.com/watch?v={video_id}"),
        duration_sec=parse_duration_seconds(payload.get("duration_sec")),
        score=score_value,
        transcript_available=bool(payload.get("transcript_available")),
        subtitles_available=bool(payload.get("subtitles_available")),
        views=_parse_views(payload.get("views")),
        published_date=str(payload["published_date"]) if payload.get("published_date") else None,
        description=str(payload["description"]) if payload.get("description") else None,
        source_bucket=str(payload.get("source_bucket") or "video_results"),
        discovery_source=str(payload["discovery_source"]) if payload.get("discovery_source") else None,
        score_breakdown=score_breakdown if isinstance(score_breakdown, dict) else None,
    )


def build_research_queries(input_value: str, question: str, *, research_mode: str = "wide") -> list[dict[str, Any]]:
    base_terms = []
    if input_value.strip() and input_value.strip() != question.strip() and detect_input_type(input_value) == "search_query":
        base_terms.append(input_value.strip())
    topic = re.sub(r"\s+", " ", (question or input_value).strip())
    if topic:
        base_terms.append(topic)

    queries: list[dict[str, Any]] = []
    for base in base_terms:
        query = canonicalize_search_query(base)
        if not query or any(item["query"] == query for item in queries):
            continue
        queries.append(
            {
                "round": 1,
                "query": query,
                "intent": "fallback discovery query",
                "expected_source_type": "model-planned or user-supplied",
                "language": "mixed",
            }
        )
        if len(queries) >= DEFAULT_DISCOVERY_QUERY_COUNT:
            return queries
    return queries


def _run_parallel_searches(client: Any, queries: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any] | None, str | None]]:
    results: list[tuple[dict[str, Any], dict[str, Any] | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=min(4, max(len(queries), 1))) as executor:
        future_map = {executor.submit(client.search, str(item["query"])): item for item in queries}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                results.append((item, future.result(), None))
            except Exception as exc:
                results.append((item, None, str(exc)))
    return results


def _run_parallel_enrichment(client: Any, candidates: list[VideoCandidate]) -> list[dict[str, Any]]:
    enrichments: list[dict[str, Any]] = []
    if not candidates or not hasattr(client, "metadata"):
        return enrichments
    with ThreadPoolExecutor(max_workers=min(6, len(candidates))) as executor:
        future_map = {executor.submit(client.metadata, candidate.video_id): candidate for candidate in candidates}
        for future in as_completed(future_map):
            candidate = future_map[future]
            try:
                payload = future.result()
                enrichment = normalize_video_enrichment(candidate.video_id, payload)
                enrichment["source_candidate"] = _candidate_payload(candidate)
                enrichments.append(enrichment)
            except Exception as exc:
                enrichments.append(
                    {
                        "video_id": candidate.video_id,
                        "title": candidate.title,
                        "channel": candidate.channel,
                        "error": str(exc),
                        "source_candidate": _candidate_payload(candidate),
                    }
                )
    return sorted(enrichments, key=lambda item: str(item.get("video_id")))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _domain_from_url(value: str) -> str:
    try:
        host = urlparse(value).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _compact_string(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return re.sub(r"\s+", " ", " ".join(str(item) for item in value)).strip()
    return ""


def normalize_web_search_results(payload: dict[str, Any], query: str, *, limit: int = 12) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    rank = 0
    for bucket in ("organic_results", "news_results", "top_stories"):
        raw_items = payload.get(bucket)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            title = _compact_string(item.get("title"))
            link = _compact_string(item.get("link") or item.get("url") or item.get("source_link"))
            if not title or not link:
                continue
            rank += 1
            source = _compact_string(item.get("source") or item.get("displayed_link")) or _domain_from_url(link)
            snippet = _compact_string(
                item.get("snippet")
                or item.get("description")
                or item.get("summary")
                or item.get("snippet_highlighted_words")
            )
            date = _compact_string(item.get("date") or item.get("published") or item.get("publication_date"))
            position = item.get("position") or item.get("position_on_page") or rank
            results.append(
                {
                    "rank": rank,
                    "position": position,
                    "result_type": bucket,
                    "query": query,
                    "title": title,
                    "url": link,
                    "source": source,
                    "date": date,
                    "snippet": snippet,
                }
            )
            if len(results) >= limit:
                return results
    return results


def unique_web_search_output_path(query: str, *, output_dir: Path, query_hash: str) -> Path:
    stem = f"{slugify(query, fallback='web-search')}-{query_hash}"
    candidate = output_dir / f"{stem}.web-search.json"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = output_dir / f"{stem}-{index}.web-search.json"
        if not candidate.exists():
            return candidate
        index += 1


def raw_web_search_output_path(web_search_output_path: Path) -> Path:
    name = web_search_output_path.name
    if name.endswith(".web-search.json"):
        return web_search_output_path.with_name(f"{name.removesuffix('.web-search.json')}.raw-web-search.json")
    return web_search_output_path.with_suffix(".raw-web-search.json")


def append_web_search_manifest(output_dir: Path, entry: dict[str, Any]) -> Path:
    manifest_path = output_dir / WEB_SEARCH_MANIFEST_FILENAME
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
    return _write_json(manifest_path, manifest)


def write_web_search_artifacts(
    *,
    query: str,
    payload: dict[str, Any],
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_query = canonicalize_search_query(query)
    query_hash = hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()[:SEARCH_QUERY_HASH_LENGTH]
    destination = unique_web_search_output_path(canonical_query, output_dir=output_dir, query_hash=query_hash)
    raw_destination = raw_web_search_output_path(destination)
    generated_at = datetime.now(timezone.utc).isoformat()
    results = normalize_web_search_results(payload, query)
    _write_json(
        raw_destination,
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "run_id": run_id,
            "query": query,
            "canonical_query": canonical_query,
            "query_hash": query_hash,
            "provider": "serpapi",
            "engine": "google",
            "payload": payload,
        },
    )
    _write_json(
        destination,
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "run_id": run_id,
            "query": query,
            "canonical_query": canonical_query,
            "query_hash": query_hash,
            "provider": "serpapi",
            "engine": "google",
            "raw_output_path": str(raw_destination),
            "result_count": len(results),
            "results": results,
        },
    )
    append_web_search_manifest(
        output_dir,
        {
            "created_at": generated_at,
            "run_id": run_id,
            "query": query,
            "canonical_query": canonical_query,
            "query_hash": query_hash,
            "provider": "serpapi",
            "engine": "google",
            "output_path": str(destination),
            "raw_output_path": str(raw_destination),
            "result_count": len(results),
            "top_urls": [str(item.get("url")) for item in results[:5] if item.get("url")],
        },
    )
    return destination


def write_search_artifacts(
    *,
    query: str,
    payload: dict[str, Any],
    output_dir: Path,
    run_id: str | None = None,
    round_number: int | None = None,
) -> Path:
    candidates = search_candidates(payload, query)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_query = canonicalize_search_query(query)
    query_hash = hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()[:SEARCH_QUERY_HASH_LENGTH]
    destination = unique_search_output_path(canonical_query, output_dir=output_dir, query_hash=query_hash)
    raw_destination = raw_search_output_path(destination)
    generated_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        raw_destination,
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
    )
    _write_json(
        destination,
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "run_id": run_id,
            "query": query,
            "canonical_query": canonical_query,
            "query_hash": query_hash,
            "provider": "serpapi",
            "raw_output_path": str(raw_destination),
            "candidate_buckets": {
                key: len(value) for key, value in payload.items() if isinstance(value, list) and key.endswith("_results")
            },
            "related_search_queries": extract_related_search_queries(payload),
            "candidates": [_candidate_payload(candidate) for candidate in candidates],
        },
    )
    entry = {
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
    }
    if round_number is not None:
        entry["search_round"] = round_number
    append_search_manifest(output_dir, entry)
    return destination


def prepare_research_discovery(
    *,
    input_value: str,
    question: str,
    research_mode: str,
    workspace_dir: Path,
    search_dir: Path,
    run_id: str,
    max_search_rounds: int = 2,
    enrichment_limit: int = DEFAULT_DISCOVERY_ENRICHMENT_LIMIT,
    selection_candidate_limit: int = DEFAULT_SELECTION_CANDIDATE_LIMIT,
    planned_queries: Optional[list[dict[str, Any]]] = None,
    client: Optional[Any] = None,
) -> dict[str, Path]:
    runtime_client = build_runtime_client(client, Path.cwd())
    generated_at = datetime.now(timezone.utc).isoformat()
    search_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    planned_queries = (
        [dict(item) for item in planned_queries]
        if planned_queries is not None
        else build_research_queries(input_value, question, research_mode=research_mode)
    )
    plan_path = workspace_dir / RESEARCH_PLAN_FILENAME
    _write_json(
        plan_path,
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "run_id": run_id,
            "input": input_value,
            "question": question,
            "research_mode": research_mode,
            "strategy": "adaptive_transcript_acquisition",
            "transcript_policy": {
                "mode": "model_decides",
                "quality_rule": "No fixed transcript count; read every transcript needed to answer the question comprehensively.",
            },
            "queries": planned_queries,
        },
    )

    search_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    search_paths: list[str] = []
    errors: list[dict[str, Any]] = []
    for query_info, payload, error in _run_parallel_searches(runtime_client, planned_queries):
        if error or payload is None:
            errors.append({"query": query_info.get("query"), "round": query_info.get("round"), "error": error})
            continue
        search_payloads.append((query_info, payload))
        search_paths.append(
            str(
                write_search_artifacts(
                    query=str(query_info["query"]),
                    payload=payload,
                    output_dir=search_dir,
                    run_id=run_id,
                    round_number=int(query_info.get("round") or 1),
                )
            )
        )

    if max_search_rounds > 1:
        existing_queries = {str(item["query"]) for item in planned_queries}
        related_queries: list[dict[str, Any]] = []
        for _, payload in search_payloads:
            for related_query in extract_related_search_queries(payload):
                if related_query in existing_queries:
                    continue
                related_queries.append(
                    {
                        "round": 2,
                        "query": related_query,
                        "intent": "follow SerpApi related search expansion",
                        "expected_source_type": "related YouTube results",
                        "language": "mixed",
                    }
                )
                existing_queries.add(related_query)
                if len(related_queries) >= 2:
                    break
            if len(related_queries) >= 2:
                break
        if related_queries:
            planned_queries.extend(related_queries)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_payload["queries"] = planned_queries
            plan_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(plan_path, plan_payload)
            for query_info, payload, error in _run_parallel_searches(runtime_client, related_queries):
                if error or payload is None:
                    errors.append({"query": query_info.get("query"), "round": query_info.get("round"), "error": error})
                    continue
                search_payloads.append((query_info, payload))
                search_paths.append(
                    str(
                        write_search_artifacts(
                            query=str(query_info["query"]),
                            payload=payload,
                            output_dir=search_dir,
                            run_id=run_id,
                            round_number=2,
                        )
                    )
                )

    all_candidates = _merge_candidates(
        candidate
        for query_info, payload in search_payloads
        for candidate in search_candidates(payload, str(query_info["query"]))
    )
    enrichments = _run_parallel_enrichment(runtime_client, all_candidates[:enrichment_limit])
    related_candidates = _merge_candidates(
        hydrated
        for enrichment in enrichments
        for candidate in enrichment.get("related_candidates", [])
        if isinstance(candidate, dict)
        for hydrated in [_candidate_from_payload(candidate)]
        if hydrated is not None
    )
    candidate_pool = _merge_candidates([*all_candidates, *related_candidates])
    selection_candidates = candidate_pool[:selection_candidate_limit]
    enrichment_path = workspace_dir / VIDEO_ENRICHMENT_MANIFEST_FILENAME
    _write_json(
        enrichment_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "provider": "serpapi",
            "search_paths": search_paths,
            "enrichment_count": len(enrichments),
            "enrichments": enrichments,
        },
    )
    selection_path = workspace_dir / SELECTION_MANIFEST_FILENAME
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "strategy": "model_selects_transcripts_from_ranked_candidates",
            "transcript_policy": {
                "mode": "adaptive",
                "instruction": "The writing agent should fetch and read as many transcript files as needed; there is no fixed target count.",
            },
            "search_round_count": max((int(item.get("round") or 1) for item in planned_queries), default=0),
            "candidate_count": len(candidate_pool),
            "selected_candidates": [_candidate_payload(candidate) for candidate in selection_candidates],
            "skipped_sources": [],
            "errors": errors,
        },
    )
    return {
        "research_plan": plan_path,
        "video_enrichment_manifest": enrichment_path,
        "selection_manifest": selection_path,
    }


def resolve_single_video(
    raw_input: str,
    client: Any,
    *,
    language_code: str,
    search_limit: int,
    asr_client: Optional[Any] = None,
    credential_root: Path = REPO_ROOT,
) -> ResolvedVideo:
    input_type = detect_input_type(raw_input)
    if input_type in {"youtube_url", "video_id"}:
        video_id = extract_video_id(raw_input) or raw_input.strip()
        probe = fetch_best_timed_content_with_fallback(
            video_id,
            client,
            language_code=language_code,
            asr_client=asr_client,
            credential_root=credential_root,
        )
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
            probe = fetch_best_timed_content_with_fallback(
                candidate.video_id,
                client,
                language_code=language_code,
                asr_client=asr_client,
                credential_root=credential_root,
            )
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
    return write_search_artifacts(query=query, payload=payload, output_dir=output_dir, run_id=run_id)


def search_web_context(
    query: str,
    *,
    output_dir: Path,
    run_id: str | None = None,
    client: Optional[Any] = None,
) -> Path:
    runtime_client = build_runtime_client(client, Path.cwd())
    if not hasattr(runtime_client, "web_search"):
        raise SerpApiError("Runtime client does not support SerpApi web_search.")
    payload = runtime_client.web_search(query)
    return write_web_search_artifacts(query=query, payload=payload, output_dir=output_dir, run_id=run_id)


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
    asr_client: Optional[Any] = None,
) -> Path:
    credential_root = Path.cwd()
    runtime_client = build_runtime_client(client, Path.cwd())
    resolved = resolve_single_video(
        raw_input,
        runtime_client,
        language_code=language_code,
        search_limit=search_limit,
        asr_client=asr_client,
        credential_root=credential_root,
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
        "provider": "aliyun_asr" if resolved.origin == "aliyun_asr" else "serpapi",
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
