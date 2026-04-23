from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

if __package__ in {None, ""}:
    from normalize import Segment
    from utils import (
        build_youtube_timestamp_url,
        format_timestamp,
        split_sentences,
        tokenize,
    )
else:
    from .normalize import Segment
    from .utils import (
        build_youtube_timestamp_url,
        format_timestamp,
        split_sentences,
        tokenize,
    )


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


@dataclass
class ArticleSection:
    heading: str
    summary: str
    supporting_segments: list[Segment]
    excerpt: str
    start_sec: int
    end_sec: Optional[int]


def _clean_text(text: str) -> str:
    cleaned = text.replace("\u00a0", " ").replace("\n", " ")
    cleaned = cleaned.replace(">>", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _is_reasonable_sentence(sentence: str) -> bool:
    cleaned = _clean_text(sentence)
    words = tokenize(cleaned)
    if len(words) < 6:
        return False
    if len(cleaned) < 40:
        return False
    if cleaned.count("...") > 0:
        return False
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.45:
        return False
    return True


def _candidate_sentences(texts: Iterable[str]) -> list[str]:
    sentences: list[str] = []
    for text in texts:
        for sentence in split_sentences(_clean_text(text)):
            cleaned = _sentence_case(_clean_text(sentence))
            if cleaned:
                sentences.append(cleaned)
    return sentences


def _sentence_case(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def _summarize_texts(texts: Iterable[str], max_sentences: int = 3) -> str:
    ordered = _candidate_sentences(texts)
    strong = [sentence for sentence in ordered if _is_reasonable_sentence(sentence)]
    chosen = strong[:max_sentences]
    if not chosen:
        chosen = ordered[:max_sentences]
    return " ".join(chosen).strip()


def _best_excerpt(segments: list[Segment]) -> str:
    candidate = _summarize_texts([segment.text for segment in segments[:3]], max_sentences=1)
    if candidate:
        return candidate
    for segment in segments:
        cleaned = _clean_text(segment.text)
        if cleaned:
            return cleaned
    return ""


def _section_heading(segments: list[Segment], index: int) -> str:
    label = next((segment.label for segment in segments if segment.label), None)
    if label:
        return label
    summary = _summarize_texts([segment.text for segment in segments], max_sentences=1)
    summary = summary.strip().rstrip(".!?")
    if summary:
        words = summary.split()
        return " ".join(words[:8])
    return f"Section {index}"


def _group_by_labels(segments: list[Segment]) -> list[list[Segment]]:
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    current_label: Optional[str] = None
    for segment in segments:
        label = segment.label
        if current and label != current_label and label is not None:
            groups.append(current)
            current = []
        current.append(segment)
        current_label = label
    if current:
        groups.append(current)
    if len(groups) <= 1:
        return []
    return groups


def _group_by_chapters(segments: list[Segment], chapters: list[dict[str, object]]) -> list[tuple[str, list[Segment]]]:
    if not chapters:
        return []
    normalized = [
        {"title": str(chapter.get("title", "")).strip(), "start_time": int(chapter.get("start_time", 0))}
        for chapter in chapters
        if isinstance(chapter, dict) and chapter.get("title") is not None and chapter.get("start_time") is not None
    ]
    if not normalized:
        return []
    normalized = sorted(normalized, key=lambda item: item["start_time"])
    groups: list[tuple[str, list[Segment]]] = []
    for index, chapter in enumerate(normalized):
        start = int(chapter["start_time"])
        end = int(normalized[index + 1]["start_time"]) if index + 1 < len(normalized) else 10**9
        matching = [segment for segment in segments if start <= segment.start_sec < end]
        if matching:
            groups.append((str(chapter["title"]).strip() or f"Section {index + 1}", matching))
    return groups


def build_outline_sections(
    segments: list[Segment],
    target_sections: int = 4,
    chapters: Optional[list[dict[str, object]]] = None,
) -> list[ArticleSection]:
    if not segments:
        raise ValueError("Cannot build an article without segments.")

    sections: list[ArticleSection] = []
    chapter_groups = _group_by_chapters(segments, chapters or [])
    if chapter_groups:
        grouped_chunks = chapter_groups[:target_sections]
    else:
        label_groups = _group_by_labels(segments)
        if label_groups:
            grouped_chunks = [(_section_heading(chunk, idx), chunk) for idx, chunk in enumerate(label_groups[:target_sections], start=1)]
        else:
            target_sections = max(1, min(target_sections, len(segments)))
            chunk_size = max(1, len(segments) // target_sections)
            raw_chunks: list[list[Segment]] = []
            for index in range(0, len(segments), chunk_size):
                raw_chunks.append(segments[index : index + chunk_size])
            if len(raw_chunks) > target_sections:
                tail = [segment for chunk in raw_chunks[target_sections - 1 :] for segment in chunk]
                raw_chunks = raw_chunks[: target_sections - 1] + [tail]
            grouped_chunks = [(_section_heading(chunk, idx), chunk) for idx, chunk in enumerate(raw_chunks, start=1)]

    for idx, payload in enumerate(grouped_chunks, start=1):
        heading, chunk = payload
        if not chunk:
            continue
        summary = _summarize_texts([segment.text for segment in chunk], max_sentences=3)
        excerpt = _best_excerpt(chunk)
        sections.append(
            ArticleSection(
                heading=heading or _section_heading(chunk, idx),
                summary=summary,
                supporting_segments=chunk,
                excerpt=excerpt,
                start_sec=chunk[0].start_sec,
                end_sec=chunk[-1].end_sec,
            )
        )
    return sections


def build_tldr(sections: list[ArticleSection]) -> str:
    sentences: list[str] = []
    for section in sections[:3]:
        candidate = _summarize_texts([section.summary], max_sentences=1)
        if candidate:
            sentences.append(candidate)
    return " ".join(sentences).strip()


def build_takeaways(sections: list[ArticleSection], limit: int = 5) -> list[str]:
    takeaways: list[str] = []
    for section in sections:
        sentence = _summarize_texts([section.summary], max_sentences=1)
        if sentence:
            takeaways.append(sentence)
    return takeaways[:limit]


def build_timeline(segments: list[Segment], limit: int = 8) -> list[str]:
    if not segments:
        return []
    stride = max(1, len(segments) // limit)
    selected = [segments[index] for index in range(0, len(segments), stride)][:limit]
    lines: list[str] = []
    for segment in selected:
        timestamp = format_timestamp(segment.start_sec)
        url = build_youtube_timestamp_url(segment.video_id, segment.start_sec)
        excerpt = _summarize_texts([segment.text], max_sentences=1) or _clean_text(segment.text)
        lines.append(f"- [{timestamp}]({url}) {excerpt}")
    return lines


def render_article_markdown(
    *,
    title: str,
    source_title: str,
    channel: str,
    video_url: str,
    language: str,
    sections: list[ArticleSection],
    template_text: str,
) -> str:
    outline = "\n".join(f"- {section.heading}" for section in sections)
    tldr = build_tldr(sections)
    takeaways = "\n".join(f"- {item}" for item in build_takeaways(sections))
    all_segments = [segment for section in sections for segment in section.supporting_segments]
    timeline = "\n".join(build_timeline(all_segments))

    rendered_sections: list[str] = []
    for section in sections:
        first_segment = section.supporting_segments[0]
        timestamp = format_timestamp(section.start_sec)
        url = build_youtube_timestamp_url(first_segment.video_id, section.start_sec)
        excerpt = section.excerpt or _summarize_texts([segment.text for segment in section.supporting_segments[:2]], max_sentences=1)
        rendered_sections.append(
            "\n".join(
                [
                    f"### {section.heading}",
                    "",
                    section.summary,
                    "",
                    f"Source moment: [{timestamp}]({url})  ",
                    f'Excerpt: "{excerpt}"',
                ]
            )
        )

    mapping = {
        "{{ title }}": title,
        "{{ source_title }}": source_title,
        "{{ channel }}": channel or "Unknown channel",
        "{{ video_url }}": video_url,
        "{{ language }}": language or "unknown",
        "{{ tldr }}": tldr,
        "{{ outline }}": outline,
        "{{ sections }}": "\n\n".join(rendered_sections),
        "{{ takeaways }}": takeaways,
        "{{ timeline }}": timeline,
    }

    content = template_text
    for key, value in mapping.items():
        content = content.replace(key, value or "")
    return content.strip() + "\n"
