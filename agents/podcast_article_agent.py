from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse
import uuid

from agents.log_format import (
    configure_logging,
    format_log_event,
    format_log_text_block,
    sanitize_for_log,
)


try:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query
except ImportError:  # pragma: no cover - exercised only when SDK is not installed.
    AssistantMessage = None
    ResultMessage = None
    query = None

    @dataclass
    class ClaudeAgentOptions:  # type: ignore[no-redef]
        cwd: str
        setting_sources: list[str]
        allowed_tools: list[str]
        permission_mode: str
        model: str | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(Path("output") / "agent")
DEFAULT_LOG_DIR = Path(DEFAULT_OUTPUT_DIR) / "logs"
AGENT_ENV_PATH = Path(__file__).resolve().parent / ".env"
SKILL_NAME = "podcast-to-article"
MODEL_ENV_KEYS = ("CLAUDE_AGENT_MODEL", "DEFAULT_MODEL")
LOG_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_AGENT_MODEL", "DEFAULT_MODEL", "SERPAPI_API_KEY")
CLAUDE_CODE_AUTH_ENV_KEYS = ("ANTHROPIC_AUTH_TOKEN",)
ProgressSink = Callable[[dict[str, object]], None]
ARTIFACT_PROGRESS_POLL_SECONDS = 0.5
RESEARCH_MODES = {"auto", "deep", "wide"}
RUN_MANIFEST_FILENAME = "run-manifest.json"
SOURCES_MANIFEST_FILENAME = "sources-manifest.json"
ARTICLE_MANIFEST_FILENAME = "article-manifest.json"
QUALITY_REPORT_FILENAME = "quality-report.json"


def default_log_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_LOG_DIR / f"podcast-article-agent-{timestamp}.log"


def _slugify(value: str, fallback: str = "video-deep-research") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or fallback


def extract_youtube_video_id(value: str) -> str | None:
    candidate = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate
    if not candidate.startswith(("http://", "https://")):
        return None
    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
    if "youtube.com" in host:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            return video_id if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return parts[1] if re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[1]) else None
    return None


def build_source_id(input_value: str) -> str:
    return extract_youtube_video_id(input_value) or _slugify(input_value)


def build_workspace_paths(output_root: Path, source_id: str) -> dict[str, Path]:
    workspace_dir = output_root / source_id
    return {
        "workspace_dir": workspace_dir,
        "search_dir": workspace_dir / "search-results",
        "transcript_dir": workspace_dir / "transcripts",
        "articles_root": workspace_dir / "articles",
    }


def build_article_dir(articles_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return articles_root / f"article-{timestamp}-{uuid.uuid4().hex[:8]}"


def resolve_research_mode(input_value: str, question: str, mode: str = "auto") -> str:
    if mode not in RESEARCH_MODES:
        raise ValueError(f"Unsupported research mode: {mode}")
    if mode in {"deep", "wide"}:
        return mode
    return "wide" if input_value.strip() == question.strip() else "deep"


def build_prompt(
    *,
    input_value: str,
    question: str,
    workspace_dir: str,
    search_dir: str,
    transcript_dir: str,
    article_path: str,
    run_id: str | None = None,
    research_mode: str = "deep",
) -> str:
    run_id_arg = f" --run-id {shlex.quote(run_id)}" if run_id else ""
    fetch_command = (
        "python3 podcast-to-article/scripts/fetch_transcript.py "
        f"{shlex.quote(input_value)} --output-dir {shlex.quote(transcript_dir)}{run_id_arg}"
    )
    run_id_block = f"\nUse this exact run id:\n{run_id}\n" if run_id else ""
    if research_mode == "wide":
        return f"""Use the {SKILL_NAME} skill to produce a grounded wide video deep research article.

Research topic:
{question}

YouTube search query:
{input_value}

Use this exact workspace directory:
{workspace_dir}

Use this exact search output directory:
{search_dir}

Use this exact transcript output directory:
{transcript_dir}
{run_id_block}

Write the final Markdown article only to this exact path:
{article_path}

Required wide-search workflow:
1. Derive 2-3 concise, complementary YouTube search queries from the research topic before running any search tool. Do not use the full user request verbatim unless it is already a compact search phrase.
   - Prefer concrete entities, domain terms, and source formats such as interview, podcast, talk, panel, or keynote.
   - Remove task wording such as "please research", "write an article", "summarize", or "help me".
   - Add a year, region, person, company, or product name only when the research topic supports it.
   - For Chinese topics, use the most likely YouTube-discoverable query. Include English terms such as AI, AGI, LLM, agent, interview, or podcast when they materially improve recall.
   - Use one primary precise query and one broader/source-format query. Add a bilingual or English query only when it is likely to improve YouTube recall.
   - If the topic asks about a broad group such as industry leaders, founders, investors, researchers, or companies, do not anchor all queries on one prominent person. Spread queries across roles, organizations, and viewpoints unless the user named a specific person.
   - If the user asks for Chinese industry leaders or Chinese companies, the main queries must target direct Chinese sources: Chinese-language interviews, talks, panels, keynotes, and founder/executive names or company names. Avoid broad English queries such as "China AI race", "China AI founder podcast", or "China AI analysis" unless they include a specific Chinese speaker or company.
2. Run the bundled YouTube search tool from the repository root for each derived query:
   python3 podcast-to-article/scripts/search_youtube.py "<derived-search-query>" --output-dir {shlex.quote(search_dir)}{run_id_arg}
3. Open `search-manifest.json` in the search output directory, then open every generated `.search.json` entry for this run id. Merge candidates by video_id, inspect the ranked candidates, and choose 3-5 relevant videos when available. Prefer substantive interviews, talks, panels, keynotes, or podcast episodes over short clips, reactions, trailers, and news snippets.
   - Enforce source diversity when the topic is broad: prefer different speakers, channels, organizations, roles, and perspectives over multiple videos centered on the same person or event.
   - If the best usable candidates are skewed toward one person, run one additional broader query before drafting to fill the missing coverage.
   - For a question about what a group of people thinks, selected main sources must be first-person or event sources from that group: the leader speaking, being interviewed, joining a panel, or giving a talk. Do not count third-party media analysis, news explainers, or foreign podcasts discussing China as "industry leader" coverage.
   - Use third-party analysis only as background context and at most one supporting source. If fewer than two direct in-scope transcripts are available, continue searching with named Chinese speakers/companies before drafting.
4. For each selected video, run the bundled transcript fetcher with the video URL. If a selected candidate has no transcript/subtitles, skip it and continue down the ranked list until you have up to 3-5 usable transcript files or all relevant candidates are exhausted:
   python3 podcast-to-article/scripts/fetch_transcript.py "<selected-video-url>" --output-dir {shlex.quote(transcript_dir)}{run_id_arg}
5. Open and read every generated `.transcript.json` file before drafting.
6. Create source coverage notes before drafting: for each transcript, identify the speaker/channel, role, whether it is direct first-person evidence or third-party analysis, main claims, and where it adds a distinct perspective. Use every usable in-scope direct transcript in the synthesis unless it is clearly off-topic; if a transcript is excluded, state the exclusion reason in the article.
7. Synthesize across the gathered transcripts. Compare recurring claims, changes over time, disagreements, and caveats when the source material supports them. Do not let one long transcript dominate a broad-topic article when other usable transcripts are available.
8. Write a coherent Markdown article that answers the research topic and includes clickable YouTube timestamp links. For broad-topic articles, the title, introduction, and conclusion must reflect the actual source breadth; do not frame the article as one person's view unless only one usable transcript was acquired. If the gathered evidence is mostly third-party analysis rather than the requested group's direct statements, narrow the title and introduction to that limitation instead of presenting it as the group's collective view.
9. Do not create article drafts in any other directory. Do not expose hidden reasoning.

Required outputs:
- one or more `.search.json` files under {search_dir}
- `search-manifest.json` under {search_dir}
- one or more `.transcript.json` files under {transcript_dir}
- {article_path}

If only one relevant transcript can be acquired, write the article from that transcript and state the coverage limitation in the article.

At the end, print:
search_queries: <derived search queries>
search: <paths to generated search json files>
transcripts: <paths to generated transcript json files>
article: {article_path}
"""
    return f"""Use the {SKILL_NAME} skill to produce a grounded video deep research article.

Input video, video ID, or search query:
{input_value}

Research request:
{question}

Use this exact workspace directory:
{workspace_dir}

Use this exact transcript output directory:
{transcript_dir}
{run_id_block}

Write the final Markdown article only to this exact path:
{article_path}

Required workflow:
1. Run the bundled transcript fetcher from the repository root:
   {fetch_command}
2. Open and read the generated `.transcript.json` file before drafting.
3. Use the complete transcript context as source evidence, including metadata, chapters, coverage, segments, and timestamp URLs.
4. Write a coherent Markdown article that answers the research request and includes clickable YouTube timestamp links.
5. Do not create article drafts in any other directory. Do not expose hidden reasoning.

Required outputs:
- one `.transcript.json` file under {transcript_dir}
- {article_path}

At the end, print:
transcript: <path to generated transcript json>
article: {article_path}
"""


def build_article_retry_prompt(*, transcript_path: Path, article_path: Path, question: str) -> str:
    return f"""The previous run fetched the transcript but stopped before writing the article.

Use this existing transcript context:
{transcript_path}

Research request:
{question}

Write only this Markdown article file:
{article_path}

Do not fetch the transcript again. Do not create other article files.
After writing the article, print the article path.
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_manifest(
    path: Path,
    *,
    status: str,
    input_value: str,
    question: str,
    source_id: str,
    research_mode: str,
    model: str | None,
    paths: dict[str, Path],
    article_path: Path,
    run_id: str,
    error_message: str | None = None,
    sources_manifest_path: Path | None = None,
    article_manifest_path: Path | None = None,
    quality_report_path: Path | None = None,
) -> None:
    existing = _read_json_object(path) or {}
    created_at = (
        existing.get("created_at")
        if existing.get("run_id") == run_id and isinstance(existing.get("created_at"), str)
        else _utc_now()
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "created_at": created_at,
        "updated_at": _utc_now(),
        "input": input_value,
        "question": question,
        "question_length": len(question),
        "source_id": source_id,
        "research_mode": research_mode,
        "model": model,
        "workspace_dir": str(paths["workspace_dir"]),
        "search_dir": str(paths["search_dir"]),
        "transcript_dir": str(paths["transcript_dir"]),
        "articles_root": str(paths["articles_root"]),
        "article_path": str(article_path),
        "artifacts": {
            "run_manifest": str(path),
            "search_manifest": str(paths["search_dir"] / "search-manifest.json"),
            "sources_manifest": str(sources_manifest_path or paths["workspace_dir"] / SOURCES_MANIFEST_FILENAME),
            "article": str(article_path),
        },
    }
    if error_message:
        payload["error_message"] = error_message
    resolved_sources_manifest_path = sources_manifest_path or paths["workspace_dir"] / SOURCES_MANIFEST_FILENAME
    resolved_article_manifest_path = article_manifest_path or article_path.parent / ARTICLE_MANIFEST_FILENAME
    resolved_quality_report_path = quality_report_path or paths["workspace_dir"] / QUALITY_REPORT_FILENAME
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict):
        artifacts = {}
    payload["artifacts"] = {
        **artifacts,
        "article_manifest": str(resolved_article_manifest_path),
        "quality_report": str(resolved_quality_report_path),
    }
    sources_manifest = _read_json_object(resolved_sources_manifest_path)
    if sources_manifest is not None:
        payload["artifact_summary"] = {
            "search_count": sources_manifest.get("search_count"),
            "transcript_count": sources_manifest.get("transcript_count"),
            "article_referenced_video_count": sources_manifest.get("article_referenced_video_count"),
        }
    quality_report = _read_json_object(resolved_quality_report_path)
    if quality_report is not None:
        summary = payload.setdefault("artifact_summary", {})
        if isinstance(summary, dict):
            summary["quality_status"] = quality_report.get("status")
            summary["quality_issue_count"] = quality_report.get("issue_count")
    _write_json(path, payload)


def _video_ids_from_text(text: str) -> set[str]:
    ids = set(re.findall(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})", text))
    ids.update(re.findall(r"[?&]v=([A-Za-z0-9_-]{11})", text))
    return ids


def write_sources_manifest(
    path: Path,
    *,
    search_dir: Path,
    transcript_dir: Path,
    article_path: Path,
    run_id: str | None = None,
) -> dict[str, object]:
    search_files = sorted(search_dir.glob("*.search.json")) if search_dir.exists() else []
    transcript_files = sorted(transcript_dir.glob("*.transcript.json")) if transcript_dir.exists() else []

    sources: dict[str, dict[str, object]] = {}
    search_summaries: list[dict[str, object]] = []
    for search_path in search_files:
        payload = _read_json_object(search_path)
        if payload is None:
            continue
        if run_id is not None and payload.get("run_id") != run_id:
            continue
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        query = payload.get("query")
        search_summaries.append(
            {
                "path": str(search_path),
                "raw_output_path": payload.get("raw_output_path"),
                "query": query,
                "canonical_query": payload.get("canonical_query"),
                "query_hash": payload.get("query_hash"),
                "candidate_count": len(candidates),
            }
        )
        for rank, raw_candidate in enumerate(candidates, start=1):
            if not isinstance(raw_candidate, dict):
                continue
            video_id = raw_candidate.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                continue
            source = sources.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "title": raw_candidate.get("title"),
                    "channel": raw_candidate.get("channel"),
                    "url": raw_candidate.get("url"),
                    "best_score": raw_candidate.get("score"),
                    "search_hits": [],
                    "transcript_path": None,
                    "has_transcript": False,
                    "referenced_in_article": False,
                },
            )
            if isinstance(raw_candidate.get("score"), (int, float)):
                best_score = source.get("best_score")
                if not isinstance(best_score, (int, float)) or raw_candidate["score"] > best_score:
                    source["best_score"] = raw_candidate["score"]
            search_hits = source.get("search_hits")
            if isinstance(search_hits, list):
                search_hits.append(
                    {
                        "search_path": str(search_path),
                        "query": query,
                        "rank": rank,
                        "score": raw_candidate.get("score"),
                    }
                )

    transcript_summaries: list[dict[str, object]] = []
    for transcript_path in transcript_files:
        payload = _read_json_object(transcript_path)
        if payload is None:
            continue
        if run_id is not None and payload.get("run_id") != run_id:
            continue
        video = payload.get("video")
        if not isinstance(video, dict):
            continue
        video_id = video.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            continue
        source = sources.setdefault(
            video_id,
            {
                "video_id": video_id,
                "title": video.get("title"),
                "channel": video.get("channel"),
                "url": video.get("url"),
                "best_score": None,
                "search_hits": [],
                "transcript_path": None,
                "has_transcript": False,
                "referenced_in_article": False,
            },
        )
        source.update(
            {
                "title": video.get("title") or source.get("title"),
                "channel": video.get("channel") or source.get("channel"),
                "url": video.get("url") or source.get("url"),
                "transcript_path": str(transcript_path),
                "has_transcript": True,
                "source_kind": payload.get("source_kind"),
                "origin": payload.get("origin"),
                "coverage": payload.get("coverage"),
            }
        )
        transcript_summaries.append(
            {
                "path": str(transcript_path),
                "video_id": video_id,
                "title": video.get("title"),
                "channel": video.get("channel"),
                "source_kind": payload.get("source_kind"),
            }
        )

    article_video_ids: set[str] = set()
    article_exists = article_path.exists() and article_path.stat().st_size > 0
    if article_exists:
        article_video_ids = _video_ids_from_text(article_path.read_text(encoding="utf-8", errors="replace"))
    for video_id in article_video_ids:
        source = sources.setdefault(
            video_id,
            {
                "video_id": video_id,
                "title": None,
                "channel": None,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "best_score": None,
                "search_hits": [],
                "transcript_path": None,
                "has_transcript": False,
                "referenced_in_article": True,
            },
        )
        source["referenced_in_article"] = True

    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "search_dir": str(search_dir),
        "transcript_dir": str(transcript_dir),
        "article_path": str(article_path),
        "search_count": len(search_summaries),
        "transcript_count": len(transcript_summaries),
        "article_referenced_video_count": len(article_video_ids),
        "searches": search_summaries,
        "transcripts": transcript_summaries,
        "sources": sorted(sources.values(), key=lambda item: str(item.get("video_id"))),
    }
    _write_json(path, manifest)
    return manifest


def _timestamp_link_count(text: str) -> int:
    return len(re.findall(r"https://(?:www\.)?(?:youtube\.com/watch\?[^)\s]+|youtu\.be/[^)\s]+)[^)\s]*(?:[?&]t=|&amp;t=)\d+s?", text))


def _article_word_count(text: str) -> int:
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(ascii_words) + len(cjk_chars)


def write_article_manifest(
    path: Path,
    *,
    run_id: str,
    article_path: Path,
    sources_manifest: dict[str, object],
) -> dict[str, object]:
    text = article_path.read_text(encoding="utf-8", errors="replace") if article_path.exists() else ""
    article_video_ids = sorted(_video_ids_from_text(text))
    sources = sources_manifest.get("sources")
    transcript_video_ids: list[str] = []
    if isinstance(sources, list):
        transcript_video_ids = sorted(
            str(source.get("video_id"))
            for source in sources
            if isinstance(source, dict) and source.get("has_transcript") and source.get("video_id")
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "article_path": str(article_path),
        "exists": article_path.exists(),
        "bytes": article_path.stat().st_size if article_path.exists() else 0,
        "chars": len(text),
        "word_count": _article_word_count(text),
        "timestamp_link_count": _timestamp_link_count(text),
        "referenced_video_ids": article_video_ids,
        "referenced_video_count": len(article_video_ids),
        "transcript_video_ids": transcript_video_ids,
        "transcript_video_count": len(transcript_video_ids),
    }
    _write_json(path, manifest)
    return manifest


def write_quality_report(
    path: Path,
    *,
    run_id: str,
    research_mode: str,
    article_path: Path,
    sources_manifest: dict[str, object],
    article_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    issues: list[dict[str, object]] = []

    def add_issue(severity: str, code: str, message: str) -> None:
        issues.append({"severity": severity, "code": code, "message": message})

    search_count = int(sources_manifest.get("search_count") or 0)
    transcript_count = int(sources_manifest.get("transcript_count") or 0)
    timestamp_link_count = int((article_manifest or {}).get("timestamp_link_count") or 0)
    referenced_ids = set((article_manifest or {}).get("referenced_video_ids") or [])
    transcript_ids = set((article_manifest or {}).get("transcript_video_ids") or [])

    if not article_path.exists() or article_path.stat().st_size == 0:
        add_issue("error", "article_missing", "article.md was not generated or is empty.")
    if transcript_count < 1:
        add_issue("error", "no_transcripts", "No transcript artifacts were generated for this run.")
    if article_path.exists() and article_path.stat().st_size > 0 and timestamp_link_count < 1:
        add_issue("warning", "no_timestamp_links", "Article does not contain clickable YouTube timestamp links.")
    if research_mode == "wide":
        if search_count < 2:
            add_issue("warning", "wide_search_count_low", "Wide mode produced fewer than two search artifacts.")
        if transcript_count < 2:
            add_issue("warning", "wide_transcript_count_low", "Wide mode produced fewer than two transcript artifacts.")

    missing_transcripts = sorted(referenced_ids - transcript_ids)
    if missing_transcripts:
        add_issue(
            "warning",
            "article_references_without_transcript",
            "Article references video ids that are not backed by current-run transcript artifacts.",
        )

    status = "passed"
    if any(issue["severity"] == "error" for issue in issues):
        status = "failed"
    elif issues:
        status = "warning"

    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "status": status,
        "issue_count": len(issues),
        "research_mode": research_mode,
        "article_path": str(article_path),
        "search_count": search_count,
        "transcript_count": transcript_count,
        "timestamp_link_count": timestamp_link_count,
        "referenced_video_count": len(referenced_ids),
        "referenced_without_transcript": missing_transcripts,
        "issues": issues,
    }
    _write_json(path, report)
    return report


def write_artifact_reports(
    *,
    sources_manifest_path: Path,
    article_manifest_path: Path,
    quality_report_path: Path,
    search_dir: Path,
    transcript_dir: Path,
    article_path: Path,
    run_id: str,
    research_mode: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sources_manifest = write_sources_manifest(
        sources_manifest_path,
        search_dir=search_dir,
        transcript_dir=transcript_dir,
        article_path=article_path,
        run_id=run_id,
    )
    article_manifest = write_article_manifest(
        article_manifest_path,
        run_id=run_id,
        article_path=article_path,
        sources_manifest=sources_manifest,
    )
    quality_report = write_quality_report(
        quality_report_path,
        run_id=run_id,
        research_mode=research_mode,
        article_path=article_path,
        sources_manifest=sources_manifest,
        article_manifest=article_manifest,
    )
    return sources_manifest, article_manifest, quality_report


def load_env_file(path: Path = AGENT_ENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
            os.environ[key] = value

    if "ANTHROPIC_API_KEY" in values:
        for key in CLAUDE_CODE_AUTH_ENV_KEYS:
            if key not in values:
                os.environ.pop(key, None)
    return values


def resolve_model(env_values: dict[str, str]) -> str | None:
    for key in MODEL_ENV_KEYS:
        value = os.environ.get(key) or env_values.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _masked_env_value(key: str, value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if "key" in key.lower() or "token" in key.lower():
        return "<set>"
    return value


def build_run_metadata(
    *,
    input_value: str,
    question: str,
    output_dir: str,
    model: str | None,
    env_values: dict[str, str],
) -> dict[str, object]:
    env_snapshot = {
        key: _masked_env_value(key, os.environ.get(key) or env_values.get(key))
        for key in LOG_ENV_KEYS
    }
    return {
        "input": input_value,
        "question_length": len(question),
        "output_dir": output_dir,
        "model": model,
        "skill": SKILL_NAME,
        "env": env_snapshot,
    }


def build_agent_options(project_root: Path = PROJECT_ROOT, model: str | None = None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=str(project_root),
        setting_sources=["project"],
        allowed_tools=["Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        model=model,
    )


def serialize_message(message: object) -> dict[str, object]:
    if is_dataclass(message):
        payload = asdict(message)
    elif hasattr(message, "__dict__"):
        payload = dict(vars(message))
    else:
        payload = {"repr": repr(message)}
    sanitized = sanitize_for_log(payload)
    if not isinstance(sanitized, dict):
        sanitized = {"value": sanitized}
    return {"type": type(message).__name__, **sanitized}


def _iter_text_blocks(message: object) -> Iterable[str]:
    if AssistantMessage is not None and isinstance(message, AssistantMessage):
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                yield text
            name = getattr(block, "name", None)
            if name:
                yield f"Tool: {name}"
    elif ResultMessage is not None and isinstance(message, ResultMessage):
        result = getattr(message, "result", None)
        if result:
            yield str(result)


def _emit_progress(
    progress_sink: ProgressSink | None,
    event_type: str,
    phase: str,
    message: str,
    *,
    data: dict[str, object] | None = None,
) -> None:
    if progress_sink is None:
        return
    progress_sink(
        {
            "type": event_type,
            "phase": phase,
            "message": message,
            "data": data or {},
        }
    )


def find_transcript_context(transcript_dir: Path) -> Path | None:
    if not transcript_dir.exists():
        return None
    transcripts = sorted(
        (path for path in transcript_dir.glob("*.transcript.json") if path.stat().st_size > 0),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return transcripts[0] if transcripts else None


async def _watch_artifact_progress(
    *,
    transcript_dir: Path,
    article_path: Path,
    progress_sink: ProgressSink | None,
    stop_event: asyncio.Event,
    poll_seconds: float = ARTIFACT_PROGRESS_POLL_SECONDS,
) -> None:
    emitted_transcript = False
    emitted_article = False

    def emit_current_progress() -> None:
        nonlocal emitted_transcript, emitted_article

        transcript_path = find_transcript_context(transcript_dir)
        if transcript_path is not None and not emitted_transcript:
            _emit_progress(
                progress_sink,
                "phase_progress",
                "source_fetch",
                "已获取转录上下文",
                data={"transcript_path": str(transcript_path)},
            )
            _emit_progress(
                progress_sink,
                "phase_started",
                "article_write",
                "正在撰写深度文章",
                data={"article_path": str(article_path)},
            )
            emitted_transcript = True

        if article_path.exists() and article_path.stat().st_size > 0 and not emitted_article:
            _emit_progress(
                progress_sink,
                "phase_progress",
                "article_write",
                "已写入深度文章",
                data={"article_path": str(article_path)},
            )
            emitted_article = True

    while not stop_event.is_set():
        emit_current_progress()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
        except asyncio.TimeoutError:
            continue

    emit_current_progress()


async def run_agent(
    *,
    input_value: str,
    question: str,
    output_dir: str,
    research_mode: str = "auto",
    project_root: Path = PROJECT_ROOT,
    log_path: Path | None = None,
    progress_sink: ProgressSink | None = None,
) -> Path:
    if query is None:
        raise RuntimeError(
            "claude-agent-sdk is not installed. Install it with: pip install claude-agent-sdk"
        )

    active_log_path = log_path or default_log_path()
    logger = configure_logging(active_log_path)
    print(f"Log file: {active_log_path}")

    env_values = load_env_file()
    model = resolve_model(env_values)
    resolved_mode = resolve_research_mode(input_value, question, research_mode)
    output_root = Path(output_dir)
    source_id = build_source_id(input_value)
    paths = build_workspace_paths(output_root, source_id)
    paths["search_dir"].mkdir(parents=True, exist_ok=True)
    paths["transcript_dir"].mkdir(parents=True, exist_ok=True)
    paths["articles_root"].mkdir(parents=True, exist_ok=True)
    article_dir = build_article_dir(paths["articles_root"])
    article_dir.mkdir(parents=True, exist_ok=False)
    article_path = article_dir / "article.md"
    run_id = article_dir.name
    run_manifest_path = paths["workspace_dir"] / RUN_MANIFEST_FILENAME
    sources_manifest_path = paths["workspace_dir"] / SOURCES_MANIFEST_FILENAME
    article_manifest_path = article_dir / ARTICLE_MANIFEST_FILENAME
    quality_report_path = paths["workspace_dir"] / QUALITY_REPORT_FILENAME
    write_run_manifest(
        run_manifest_path,
        status="running",
        input_value=input_value,
        question=question,
        source_id=source_id,
        research_mode=resolved_mode,
        model=model,
        paths=paths,
        article_path=article_path,
        run_id=run_id,
        sources_manifest_path=sources_manifest_path,
        article_manifest_path=article_manifest_path,
        quality_report_path=quality_report_path,
    )

    _emit_progress(
        progress_sink,
        "phase_started",
        "source_fetch",
        "正在获取视频转录上下文",
        data={"source_id": source_id, "research_mode": resolved_mode},
    )

    prompt = build_prompt(
        input_value=input_value,
        question=question,
        workspace_dir=str(paths["workspace_dir"]),
        search_dir=str(paths["search_dir"]),
        transcript_dir=str(paths["transcript_dir"]),
        article_path=str(article_path),
        run_id=run_id,
        research_mode=resolved_mode,
    )
    options = build_agent_options(project_root, model=model)

    metadata = build_run_metadata(
        input_value=input_value,
        question=question,
        output_dir=str(paths["workspace_dir"]),
        model=model,
        env_values=env_values,
    )
    logger.info(
        format_log_event(
            "agent_start",
            {
                **metadata,
                "source_id": source_id,
                "research_mode": resolved_mode,
                "search_dir": str(paths["search_dir"]),
                "transcript_dir": str(paths["transcript_dir"]),
                "article_path": str(article_path),
            },
        )
    )
    logger.info(format_log_event("prompt_ready", {"prompt_chars": len(prompt), "project_root": str(project_root)}))

    stop_event = asyncio.Event()
    artifact_watcher = asyncio.create_task(
        _watch_artifact_progress(
            transcript_dir=paths["transcript_dir"],
            article_path=article_path,
            progress_sink=progress_sink,
            stop_event=stop_event,
        )
    )
    try:
        async for message in query(prompt=prompt, options=options):
            logger.info(format_log_event("sdk_message", serialize_message(message)))
            for text in _iter_text_blocks(message):
                logger.info(format_log_text_block("message_text", text))
                print(text)
    except Exception as exc:
        logger.exception(format_log_event("agent_failed"))
        write_artifact_reports(
            sources_manifest_path=sources_manifest_path,
            article_manifest_path=article_manifest_path,
            quality_report_path=quality_report_path,
            search_dir=paths["search_dir"],
            transcript_dir=paths["transcript_dir"],
            article_path=article_path,
            run_id=run_id,
            research_mode=resolved_mode,
        )
        write_run_manifest(
            run_manifest_path,
            status="failed",
            input_value=input_value,
            question=question,
            source_id=source_id,
            research_mode=resolved_mode,
            model=model,
            paths=paths,
            article_path=article_path,
            run_id=run_id,
            error_message=str(exc),
            sources_manifest_path=sources_manifest_path,
            article_manifest_path=article_manifest_path,
            quality_report_path=quality_report_path,
        )
        _emit_progress(progress_sink, "task_failed", "failed", "任务失败")
        raise
    finally:
        stop_event.set()
        await artifact_watcher

    transcript_path = find_transcript_context(paths["transcript_dir"])
    if transcript_path is not None and (not article_path.exists() or article_path.stat().st_size == 0):
        retry_prompt = build_article_retry_prompt(
            transcript_path=transcript_path,
            article_path=article_path,
            question=question,
        )
        logger.info(
            format_log_event(
                "article_retry_started",
                {"transcript_path": str(transcript_path), "article_path": str(article_path)},
            )
        )
        try:
            async for message in query(prompt=retry_prompt, options=options):
                logger.info(format_log_event("sdk_message", serialize_message(message)))
                for text in _iter_text_blocks(message):
                    logger.info(format_log_text_block("message_text", text))
                    print(text)
        except Exception as exc:
            logger.exception(format_log_event("article_retry_failed"))
            write_artifact_reports(
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                search_dir=paths["search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
                research_mode=resolved_mode,
            )
            write_run_manifest(
                run_manifest_path,
                status="failed",
                input_value=input_value,
                question=question,
                source_id=source_id,
                research_mode=resolved_mode,
                model=model,
                paths=paths,
                article_path=article_path,
                run_id=run_id,
                error_message=str(exc),
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
            )
            raise

    if not article_path.exists() or article_path.stat().st_size == 0:
        write_artifact_reports(
            sources_manifest_path=sources_manifest_path,
            article_manifest_path=article_manifest_path,
            quality_report_path=quality_report_path,
            search_dir=paths["search_dir"],
            transcript_dir=paths["transcript_dir"],
            article_path=article_path,
            run_id=run_id,
            research_mode=resolved_mode,
        )
        write_run_manifest(
            run_manifest_path,
            status="failed",
            input_value=input_value,
            question=question,
            source_id=source_id,
            research_mode=resolved_mode,
            model=model,
            paths=paths,
            article_path=article_path,
            run_id=run_id,
            error_message="agent stopped before writing article.md",
            sources_manifest_path=sources_manifest_path,
            article_manifest_path=article_manifest_path,
            quality_report_path=quality_report_path,
        )
        raise RuntimeError("agent stopped before writing article.md")

    sources_manifest, article_manifest, quality_report = write_artifact_reports(
        sources_manifest_path=sources_manifest_path,
        article_manifest_path=article_manifest_path,
        quality_report_path=quality_report_path,
        search_dir=paths["search_dir"],
        transcript_dir=paths["transcript_dir"],
        article_path=article_path,
        run_id=run_id,
        research_mode=resolved_mode,
    )
    write_run_manifest(
        run_manifest_path,
        status="completed",
        input_value=input_value,
        question=question,
        source_id=source_id,
        research_mode=resolved_mode,
        model=model,
        paths=paths,
        article_path=article_path,
        run_id=run_id,
        sources_manifest_path=sources_manifest_path,
        article_manifest_path=article_manifest_path,
        quality_report_path=quality_report_path,
    )

    _emit_progress(
        progress_sink,
        "phase_progress",
        "article_write",
        "已写入深度文章",
        data={
            "article_path": str(article_path),
            "sources_manifest_path": str(sources_manifest_path),
            "article_manifest_path": str(article_manifest_path),
            "quality_report_path": str(quality_report_path),
            "transcript_count": sources_manifest.get("transcript_count"),
            "quality_status": quality_report.get("status"),
        },
    )
    logger.info(
        format_log_event(
            "agent_completed",
            {
                "article_path": str(article_path),
                "run_manifest_path": str(run_manifest_path),
                "sources_manifest_path": str(sources_manifest_path),
                "article_manifest_path": str(article_manifest_path),
                "quality_report_path": str(quality_report_path),
                "quality_status": quality_report.get("status"),
            },
        )
    )
    return article_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the podcast-to-article Claude Agent SDK runner.")
    parser.add_argument("--input", default="", help="YouTube URL, video ID, or search query. Optional in --mode wide.")
    parser.add_argument(
        "--question",
        default="请基于这个视频生成一篇结构化深度研究文章。",
        help="Research request to answer in the article.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(RESEARCH_MODES),
        default="auto",
        help="deep uses the supplied input as one source; wide searches multiple videos. Defaults to auto.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Workspace output root. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to output/agent/logs/podcast-article-agent-<timestamp>.log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_value = args.input.strip() or args.question.strip()
    asyncio.run(
        run_agent(
            input_value=input_value,
            question=args.question,
            output_dir=args.output_dir,
            research_mode=args.mode,
            log_path=Path(args.log_file) if args.log_file else None,
        )
    )


if __name__ == "__main__":
    main()
