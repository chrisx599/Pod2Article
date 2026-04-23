from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass
class Segment:
    start_sec: int
    end_sec: Optional[int]
    text: str
    source_kind: str
    language: Optional[str]
    video_id: str
    label: Optional[str] = None


TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?:\d+\s+(?:hours?|minutes?|seconds?)\s*,?\s*)+",
    re.IGNORECASE,
)


def _coerce_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _segment_text(payload: dict[str, Any]) -> str:
    accessibility_text = (
        payload.get("accessibility", {})
        .get("accessibilityData", {})
        .get("label")
    )
    if isinstance(accessibility_text, str) and accessibility_text.strip():
        cleaned = TIMESTAMP_PREFIX_RE.sub("", accessibility_text.strip()).strip()
        if cleaned:
            return cleaned
    snippet = payload.get("snippet", {})
    if isinstance(snippet, dict):
        if "runs" in snippet:
            return " ".join(run.get("text", "").strip() for run in snippet.get("runs", [])).strip()
        if "simpleText" in snippet:
            return str(snippet.get("simpleText", "")).strip()
    if "text" in payload:
        return str(payload.get("text", "")).strip()
    return ""


def _extract_items(result_content: Any) -> Iterable[Any]:
    if isinstance(result_content, list):
        return result_content
    if isinstance(result_content, dict):
        if "content" in result_content:
            return _extract_items(result_content["content"])
        if "results" in result_content:
            return _extract_items(result_content["results"])
    return []


def _normalize_raw_subtitle_events(
    content: dict[str, Any],
    *,
    video_id: str,
    source_kind: str,
    language: Optional[str],
) -> list[Segment]:
    segments: list[Segment] = []
    subtitle_roots = [
        content.get("auto_generated"),
        content.get("uploader_provided"),
        content.get("user_generated"),
        content.get("translated"),
    ]
    for root in subtitle_roots:
        if not isinstance(root, dict):
            continue
        for lang_code, lang_payload in root.items():
            if not isinstance(lang_payload, dict):
                continue
            events = lang_payload.get("events", [])
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                raw_segs = event.get("segs", [])
                if not isinstance(raw_segs, list):
                    continue
                text = "".join(str(seg.get("utf8", "")) for seg in raw_segs).replace("\n", " ").strip()
                if not text:
                    continue
                start_ms = _coerce_ms(event.get("tStartMs"))
                duration_ms = _coerce_ms(event.get("dDurationMs"))
                if start_ms is None:
                    continue
                end_sec = None
                if duration_ms is not None:
                    end_sec = (start_ms + duration_ms) // 1000
                segments.append(
                    Segment(
                        start_sec=start_ms // 1000,
                        end_sec=end_sec,
                        text=text,
                        source_kind=source_kind,
                        language=language or lang_code,
                        video_id=video_id,
                        label=None,
                    )
                )
            if segments:
                return segments
    return segments


def _chapter_title(chapter: dict[str, Any], index: int) -> str:
    title = chapter.get("chapter") or chapter.get("title") or f"Chapter {index + 1}"
    cleaned = re.sub(r"^Chapter\s+\d+\s*:\s*", "", str(title).strip(), flags=re.IGNORECASE)
    return cleaned or f"Chapter {index + 1}"


def _normalize_serpapi_transcript(
    payload: dict[str, Any],
    *,
    video_id: str,
    source_kind: str,
    language: Optional[str],
) -> list[Segment]:
    transcript = payload.get("transcript")
    if not isinstance(transcript, list):
        return []

    raw_chapters = payload.get("chapters", [])
    chapters: list[tuple[int, str]] = []
    if isinstance(raw_chapters, list):
        for index, chapter in enumerate(raw_chapters):
            if not isinstance(chapter, dict):
                continue
            start_ms = _coerce_ms(chapter.get("start_ms"))
            if start_ms is None:
                start_sec = _coerce_ms(chapter.get("time_start"))
                start_ms = start_sec * 1000 if start_sec is not None else None
            if start_ms is None:
                continue
            chapters.append((start_ms // 1000, _chapter_title(chapter, index)))
    chapters.sort(key=lambda item: item[0])

    def label_for(start_sec: int) -> Optional[str]:
        active = None
        for chapter_start, title in chapters:
            if chapter_start > start_sec:
                break
            active = title
        return active

    segments: list[Segment] = []
    for item in transcript:
        if not isinstance(item, dict):
            continue
        text = str(item.get("snippet", "")).replace("\n", " ").strip()
        if not text:
            continue
        start_ms = _coerce_ms(item.get("start_ms"))
        if start_ms is None:
            continue
        end_ms = _coerce_ms(item.get("end_ms"))
        start_sec = start_ms // 1000
        segments.append(
            Segment(
                start_sec=start_sec,
                end_sec=(end_ms // 1000) if end_ms is not None else None,
                text=text,
                source_kind=source_kind,
                language=language,
                video_id=video_id,
                label=label_for(start_sec),
            )
        )
    return segments


def normalize_timed_content(
    payload: dict[str, Any],
    *,
    video_id: str,
    source_kind: str,
    language: Optional[str] = None,
) -> list[Segment]:
    serpapi_segments = _normalize_serpapi_transcript(
        payload,
        video_id=video_id,
        source_kind=source_kind,
        language=language,
    )
    if serpapi_segments:
        return serpapi_segments

    results = payload.get("results", [])
    if not results:
        raise ValueError("No Oxylabs or SerpApi results were returned.")

    first = results[0]
    content = first.get("content")
    if content is None:
        raise ValueError("Oxylabs result is missing content.")

    if isinstance(content, dict):
        raw_subtitle_segments = _normalize_raw_subtitle_events(
            content,
            video_id=video_id,
            source_kind=source_kind,
            language=language,
        )
        if raw_subtitle_segments:
            return raw_subtitle_segments

    segments: list[Segment] = []
    current_label: Optional[str] = None
    for item in _extract_items(content):
        if not isinstance(item, dict):
            continue
        if "transcriptSectionHeaderRenderer" in item:
            header = item["transcriptSectionHeaderRenderer"]
            current_label = (
                header.get("sectionHeader", {})
                .get("sectionHeaderViewModel", {})
                .get("headline", {})
                .get("content")
            )
            continue

        renderer = item.get("transcriptSegmentRenderer") or item.get("cueRenderer") or item.get("subtitleSegmentRenderer")
        if not isinstance(renderer, dict):
            continue
        text = _segment_text(renderer)
        if not text:
            continue
        start_ms = _coerce_ms(renderer.get("startMs"))
        end_ms = _coerce_ms(renderer.get("endMs"))
        if start_ms is None:
            continue
        segments.append(
            Segment(
                start_sec=start_ms // 1000,
                end_sec=(end_ms // 1000) if end_ms is not None else None,
                text=text,
                source_kind=source_kind,
                language=language,
                video_id=video_id,
                label=current_label,
            )
        )

    if not segments:
        raise ValueError("No usable timestamped segments were found in the timed-content payload.")
    return segments


def merge_timed_segments(segments: list[Segment], max_gap_sec: int = 1, target_chars: int = 220) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = []
    current = Segment(
        start_sec=segments[0].start_sec,
        end_sec=segments[0].end_sec,
        text=segments[0].text.strip(),
        source_kind=segments[0].source_kind,
        language=segments[0].language,
        video_id=segments[0].video_id,
        label=segments[0].label,
    )

    def should_merge(left: Segment, right: Segment) -> bool:
        if left.video_id != right.video_id:
            return False
        if left.label != right.label:
            return False
        left_end = left.end_sec if left.end_sec is not None else left.start_sec
        if right.start_sec - left_end > max_gap_sec:
            return False
        if len(left.text) + len(right.text) > 500:
            return False
        left_open = not re.search(r'[.!?]["\']?$', left.text.strip()) or len(left.text) < target_chars
        right_continues = bool(right.text[:1].islower())
        return left_open or right_continues

    for segment in segments[1:]:
        candidate_text = segment.text.strip()
        if not candidate_text:
            continue
        if should_merge(current, segment):
            joiner = "" if current.text.endswith("-") else " "
            current.text = f"{current.text.rstrip()}{joiner}{candidate_text.lstrip()}".strip()
            current.end_sec = segment.end_sec if segment.end_sec is not None else current.end_sec
            continue
        merged.append(current)
        current = Segment(
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            text=candidate_text,
            source_kind=segment.source_kind,
            language=segment.language,
            video_id=segment.video_id,
            label=segment.label,
        )

    merged.append(current)
    return merged
