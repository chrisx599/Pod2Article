from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Callable, Iterable
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
EVIDENCE_MODEL_ENV_KEYS = ("EVIDENCE_AGENT_MODEL",)
LOG_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_AGENT_MODEL",
    "EVIDENCE_AGENT_MODEL",
    "DEFAULT_MODEL",
    "SERPAPI_API_KEY",
)
CLAUDE_CODE_AUTH_ENV_KEYS = ("ANTHROPIC_AUTH_TOKEN",)
ProgressSink = Callable[[dict[str, object]], None]
ARTIFACT_PROGRESS_POLL_SECONDS = 0.5
RESEARCH_MODES = {"auto", "deep", "wide"}
RUN_MANIFEST_FILENAME = "run-manifest.json"
SOURCES_MANIFEST_FILENAME = "sources-manifest.json"
ARTICLE_MANIFEST_FILENAME = "article-manifest.json"
QUALITY_REPORT_FILENAME = "quality-report.json"
RESEARCH_PLAN_FILENAME = "research-plan.json"
VIDEO_ENRICHMENT_MANIFEST_FILENAME = "video-enrichment-manifest.json"
SELECTION_MANIFEST_FILENAME = "selection-manifest.json"
EVIDENCE_DIRNAME = "evidence"
EVIDENCE_MANIFEST_FILENAME = "evidence-manifest.json"
EVIDENCE_MAX_CONCURRENCY = 3
WEB_SEARCH_DIRNAME = "web-search"
WEB_EVIDENCE_FILENAME = "web-evidence.json"
WEB_EVIDENCE_MAX_CARDS = 12
TRANSCRIPT_FETCH_PLAN_FILENAME = "transcript-fetch-plan.json"
TRANSCRIPT_FETCH_MANIFEST_FILENAME = "transcript-fetch-manifest.json"
QUERY_PLAN_FILENAME = "query-plan.json"
INITIAL_SEARCH_QUERY_PLAN_FILENAME = "initial-search-query-plan.json"
INITIAL_DISCOVERY_QUERY_COUNT = 4
WIDE_TRANSCRIPT_TARGET_COUNT = 10
WIDE_TRANSCRIPT_PROBE_LIMIT = 18
WIDE_TRANSCRIPT_MAX_CONCURRENCY = 4
WIDE_SUPPLEMENTAL_QUERY_COUNT = 4
WIDE_SUPPLEMENTAL_WEB_QUERY_COUNT = 4
WIDE_SUPPLEMENTAL_SEARCH_MAX_CONCURRENCY = 4
PODCAST_SCRIPTS_DIR = PROJECT_ROOT / "podcast-to-article" / "scripts"
if str(PODCAST_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PODCAST_SCRIPTS_DIR))
from youtube_sources import fetch_transcript_context, prepare_research_discovery, search_web_context, search_youtube_context  # noqa: E402


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
        "web_search_dir": workspace_dir / WEB_SEARCH_DIRNAME,
        "transcript_dir": workspace_dir / "transcripts",
        "evidence_dir": workspace_dir / EVIDENCE_DIRNAME,
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
    research_plan_path: str | None = None,
    video_enrichment_manifest_path: str | None = None,
    selection_manifest_path: str | None = None,
    transcript_fetch_plan_path: str | None = None,
) -> str:
    run_id_arg = f" --run-id {shlex.quote(run_id)}" if run_id else ""
    fetch_command = (
        "python3 podcast-to-article/scripts/fetch_transcript.py "
        f"{shlex.quote(input_value)} --output-dir {shlex.quote(transcript_dir)}{run_id_arg}"
    )
    run_id_block = f"\nUse this exact run id:\n{run_id}\n" if run_id else ""
    if research_mode == "wide" or selection_manifest_path:
        fetch_plan_path = transcript_fetch_plan_path or str(Path(workspace_dir) / TRANSCRIPT_FETCH_PLAN_FILENAME)
        return f"""Use the {SKILL_NAME} skill. Plan only supplemental YouTube and Web search queries for a grounded wide video deep research article.

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

Reserve this final Markdown article path for the later writing step:
{article_path}

The Python runner, not you, will write the transcript fetch plan here:
{fetch_plan_path}

Required workflow:
1. Do not run Bash, Read, Write, search tools, or transcript fetchers.
2. Return only a compact JSON object in your final message.
3. Propose up to {WIDE_SUPPLEMENTAL_QUERY_COUNT} targeted supplemental YouTube search queries and up to {WIDE_SUPPLEMENTAL_WEB_QUERY_COUNT} targeted Web search queries that would improve source breadth for this research topic.
4. Focus on semantic gaps that generic search may miss: specific speakers, organizations, roles, English/Chinese balance, panels, podcasts, keynotes, or recency.
5. Every query must be a search-engine query, not the user's task sentence. Do not copy task wording such as write, report, summarize, research, 搜集, 写, 报告, 研判, 总结, 调研.
6. Do not select videos and do not write transcript-fetch-plan.json; the Python runner will execute your queries, merge candidates, fetch transcripts in parallel, and build compact web evidence.

Return JSON exactly in this shape:
{{
  "schema_version": 1,
  "supplemental_youtube_queries": [
    {{"query": "<youtube search query>", "reason": "<brief reason>"}}
  ],
  "supplemental_web_queries": [
    {{"query": "<google web search query>", "reason": "<brief reason>"}}
  ]
}}
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
4. Write a coherent Markdown article that answers the research request and includes clickable YouTube timestamp links. The visible text of each timestamp link must be a concise but informative, consistent video-title cue plus the timestamp, exactly like `[马斯克 Lex Fridman 访谈 00:55:33](https://www.youtube.com/watch?v=...&t=3333s)`. Use the same cue for every link from the same video. Do not use bare timestamp links such as `[00:55:33](...)`, parenthetical cue wrappers such as `(马斯克访谈 [00:55:33](...))`, or labels such as `▶ 12:34`.
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


def build_evidence_prompt(*, question: str, transcript_path: Path, evidence_path: Path) -> str:
    return f"""Extract compact, question-focused evidence cards from one YouTube transcript.

Research request:
{question}

Read this transcript JSON:
{transcript_path}

Write one JSON object only to this exact path:
{evidence_path}

Required JSON schema:
{{
  "schema_version": 1,
  "video_id": "<video id>",
  "title": "<video title>",
  "channel": "<channel>",
  "source_kind": "<transcript or subtitles>",
  "transcript_path": "{transcript_path}",
  "relevance": "high|medium|low",
  "coverage_note": "<brief note on transcript coverage and fit>",
  "excluded": false,
  "exclusion_reason": "",
  "cards": [
    {{
      "claim": "<one source-grounded claim relevant to the request>",
      "why_it_matters": "<why this evidence matters for the final article>",
      "timestamp": "HH:MM:SS",
      "start_sec": 0,
      "url": "https://www.youtube.com/watch?v=<id>&t=<seconds>s",
      "quote_or_paraphrase": "<short source-faithful quote or paraphrase>",
      "source_cue": "<concise but informative video-title cue suitable inside a timestamp link before the time>"
    }}
  ]
}}

Rules:
- Read the transcript before writing the JSON.
- Prefer 4-8 high-signal cards when the transcript is relevant; use fewer for low-relevance sources.
- Every card must include a valid timestamp, start_sec, and YouTube URL from the transcript.
- Keep quote_or_paraphrase concise; do not copy long transcript passages.
- Set excluded=true only when the transcript is clearly off-topic, low-quality, or duplicate evidence.
- Do not write Markdown, an article, or any file except {evidence_path}.
"""


def build_wide_article_prompt(
    *,
    question: str,
    evidence_manifest_path: Path,
    article_path: Path,
    sources_manifest_path: Path,
    web_evidence_path: Path | None = None,
) -> str:
    web_evidence_block = (
        f"\nAlso read this web evidence file for background and corroboration:\n{web_evidence_path}\n"
        if web_evidence_path is not None
        else ""
    )
    return f"""Write the final grounded wide video deep research article from compact evidence cards.

Research request:
{question}

Read this evidence manifest first:
{evidence_manifest_path}

Also read this source manifest for video titles, channels, and source cues:
{sources_manifest_path}
{web_evidence_block}

Write the final Markdown article only to this exact path:
{article_path}

Required workflow:
1. Read `evidence-manifest.json`, then read every successful `.evidence.json` file listed there.
2. Use the evidence cards as the primary context. Do not read full `.transcript.json` files unless the evidence is clearly insufficient for a specific claim; if you do, keep it narrowly targeted.
3. Synthesize across sources. Compare recurring claims, changes over time, disagreements, and caveats when the evidence supports them.
4. Do not let one long transcript dominate a broad-topic article when other usable evidence cards are available.
5. Write a coherent Markdown article that answers the research request and includes clickable YouTube timestamp links.
6. The visible text of each timestamp citation must be a concise but informative, consistent video-title cue plus the timestamp, exactly like `[马斯克 Lex Fridman 访谈 00:55:33](https://www.youtube.com/watch?v=...&t=3333s)`. Use the same cue for every link from the same video. Do not use bare timestamp links such as `[00:55:33](...)`, parenthetical cue wrappers such as `(马斯克访谈 [00:55:33](...))`, or labels such as `▶ 12:34`.
7. Use web evidence only for background, fact-checking, timelines, official/company/person context, and corroboration. Cite web sources with normal Markdown links when used, and do not turn snippet-level evidence into unsupported strong claims.
8. If evidence generation failed for some transcripts or useful sources were excluded, state the coverage limitation briefly in the article.
9. Do not create article drafts in any other directory. Do not expose hidden reasoning.

At the end, print:
evidence: {evidence_manifest_path}
article: {article_path}
"""


def _candidate_summary_for_query_planner(selection_manifest_path: Path, *, limit: int = 18) -> str:
    selection_manifest = _read_json_object(selection_manifest_path) or {}
    candidates = selection_manifest.get("selected_candidates")
    if not isinstance(candidates, list):
        return "No prebuilt candidates were available."
    lines: list[str] = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        if not isinstance(candidate, dict):
            continue
        title = str(candidate.get("title") or "").strip()
        channel = str(candidate.get("channel") or "").strip()
        published = str(candidate.get("published_date") or "").strip()
        score = candidate.get("score")
        video_id = str(candidate.get("video_id") or "").strip()
        lines.append(f"{index}. {title} | {channel} | {published or 'date unknown'} | score={score} | {video_id}")
    return "\n".join(lines) or "No prebuilt candidates were available."


def build_query_planner_prompt(*, question: str, selection_manifest_path: Path) -> str:
    return f"""Plan only supplemental YouTube and Web search queries for a video deep research task.

Research request:
{question}

Current candidate snapshot:
{_candidate_summary_for_query_planner(selection_manifest_path)}

Return only compact JSON:
{{
  "schema_version": 1,
  "supplemental_youtube_queries": [
    {{"query": "<youtube search query>", "reason": "<brief reason>"}}
  ],
  "supplemental_web_queries": [
    {{"query": "<google web search query>", "reason": "<brief reason>"}}
  ]
}}

Rules:
- Do not use tools.
- Do not select videos.
- Do not write files.
- Return at most {WIDE_SUPPLEMENTAL_QUERY_COUNT} YouTube queries and at most {WIDE_SUPPLEMENTAL_WEB_QUERY_COUNT} Web queries.
- Each query must be a search-engine query, not a user task or sentence.
- Do not include words whose only purpose is instructing the agent, such as write, report, summarize, research, 搜集, 写, 报告, 研判, 总结, 调研.
- YouTube queries should look like terms a person would type into YouTube: topic + speaker/entity if useful + interview/podcast/talk/keynote + year.
- Web queries should look like Google queries: topic + entity + interview/podcast/keynote + date or recency terms when useful.
- YouTube queries should improve transcript source diversity, recency, speaker diversity, and English/Chinese balance.
- Web queries should find background facts, timelines, official announcements, company/person context, and corroborating sources.
"""


def build_initial_search_query_prompt(*, input_value: str, question: str, max_queries: int = INITIAL_DISCOVERY_QUERY_COUNT) -> str:
    return f"""Generate initial YouTube search queries for a wide video deep research task.

Research request:
{question}

Raw user input:
{input_value}

Return only compact JSON:
{{
  "schema_version": 1,
  "youtube_search_queries": [
    {{"query": "<youtube search query>", "reason": "<brief reason>", "language": "<zh|en|mixed>"}}
  ]
}}

Rules:
- Generate up to {max_queries} queries.
- Each query must be a search-engine query, not a user task or sentence.
- Do not copy the raw user request if it contains instructions like write a report, summarize, research, 搜集, 写一份, 报告, 研判, 总结, 调研.
- Prefer concise YouTube-style queries: topic + source type + speaker/entity if useful + year.
- Include source-type words users actually search for, such as interview, podcast, talk, keynote, panel, 访谈, 播客, 对谈, 演讲.
- Mix Chinese and English queries when the topic spans both markets.
- If the request has a recent time window, express it as search-friendly year/month or recency terms, not as a full natural-language instruction.

Bad query:
搜集近三个月以来 ai 行业的重要访谈，播客，写一份该行业的研判报告 interview talk

Good queries:
AI industry interview podcast 2026
AI leaders interview podcast 2026
人工智能 行业 访谈 播客 2026
AI keynote panel 2026
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
    research_plan_path: Path | None = None,
    video_enrichment_manifest_path: Path | None = None,
    selection_manifest_path: Path | None = None,
    evidence_manifest_path: Path | None = None,
) -> None:
    existing = _read_json_object(path) or {}
    created_at = (
        existing.get("created_at")
        if existing.get("run_id") == run_id and isinstance(existing.get("created_at"), str)
        else _utc_now()
    )
    resolved_sources_manifest_path = sources_manifest_path or paths["workspace_dir"] / SOURCES_MANIFEST_FILENAME
    resolved_article_manifest_path = article_manifest_path or article_path.parent / ARTICLE_MANIFEST_FILENAME
    resolved_quality_report_path = quality_report_path or paths["workspace_dir"] / QUALITY_REPORT_FILENAME
    resolved_evidence_manifest_path = evidence_manifest_path or paths["evidence_dir"] / EVIDENCE_MANIFEST_FILENAME
    resolved_research_plan_path = research_plan_path or paths["workspace_dir"] / RESEARCH_PLAN_FILENAME
    resolved_video_enrichment_manifest_path = (
        video_enrichment_manifest_path or paths["workspace_dir"] / VIDEO_ENRICHMENT_MANIFEST_FILENAME
    )
    resolved_selection_manifest_path = selection_manifest_path or paths["workspace_dir"] / SELECTION_MANIFEST_FILENAME
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
        "web_search_dir": str(paths["web_search_dir"]),
        "transcript_dir": str(paths["transcript_dir"]),
        "evidence_dir": str(paths["evidence_dir"]),
        "articles_root": str(paths["articles_root"]),
        "article_path": str(article_path),
        "artifacts": {
            "run_manifest": str(path),
            "search_manifest": str(paths["search_dir"] / "search-manifest.json"),
            "web_search_manifest": str(paths["web_search_dir"] / "web-search-manifest.json"),
            "web_evidence": str(paths["web_search_dir"] / WEB_EVIDENCE_FILENAME),
            "sources_manifest": str(resolved_sources_manifest_path),
            "initial_query_plan": str(paths["workspace_dir"] / INITIAL_SEARCH_QUERY_PLAN_FILENAME),
            "query_plan": str(paths["workspace_dir"] / QUERY_PLAN_FILENAME),
            "transcript_fetch_plan": str(paths["workspace_dir"] / TRANSCRIPT_FETCH_PLAN_FILENAME),
            "transcript_fetch_manifest": str(paths["workspace_dir"] / TRANSCRIPT_FETCH_MANIFEST_FILENAME),
            "evidence_manifest": str(resolved_evidence_manifest_path),
            "article": str(article_path),
        },
    }
    if error_message:
        payload["error_message"] = error_message
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict):
        artifacts = {}
    payload["artifacts"] = {
        **artifacts,
        "article_manifest": str(resolved_article_manifest_path),
        "quality_report": str(resolved_quality_report_path),
        "research_plan": str(resolved_research_plan_path),
        "video_enrichment_manifest": str(resolved_video_enrichment_manifest_path),
        "selection_manifest": str(resolved_selection_manifest_path),
    }
    sources_manifest = _read_json_object(resolved_sources_manifest_path)
    if sources_manifest is not None:
        payload["artifact_summary"] = {
            "search_count": sources_manifest.get("search_count"),
            "web_search_count": sources_manifest.get("web_search_count"),
            "transcript_count": sources_manifest.get("transcript_count"),
            "article_referenced_video_count": sources_manifest.get("article_referenced_video_count"),
        }
    quality_report = _read_json_object(resolved_quality_report_path)
    if quality_report is not None:
        summary = payload.setdefault("artifact_summary", {})
        if isinstance(summary, dict):
            summary["quality_status"] = quality_report.get("status")
            summary["quality_issue_count"] = quality_report.get("issue_count")
    evidence_manifest = _read_json_object(resolved_evidence_manifest_path)
    if evidence_manifest is not None:
        summary = payload.setdefault("artifact_summary", {})
        if isinstance(summary, dict):
            summary["evidence_success_count"] = evidence_manifest.get("success_count")
            summary["evidence_failed_count"] = evidence_manifest.get("failed_count")
    _write_json(path, payload)


def _video_ids_from_text(text: str) -> set[str]:
    ids = set(re.findall(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})", text))
    ids.update(re.findall(r"[?&]v=([A-Za-z0-9_-]{11})", text))
    return ids


def write_sources_manifest(
    path: Path,
    *,
    search_dir: Path,
    web_search_dir: Path | None = None,
    transcript_dir: Path,
    article_path: Path,
    run_id: str | None = None,
) -> dict[str, object]:
    search_files = sorted(search_dir.glob("*.search.json")) if search_dir.exists() else []
    web_search_files = sorted(web_search_dir.glob("*.web-search.json")) if web_search_dir is not None and web_search_dir.exists() else []
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

    web_search_summaries: list[dict[str, object]] = []
    for web_search_path in web_search_files:
        payload = _read_json_object(web_search_path)
        if payload is None:
            continue
        if run_id is not None and payload.get("run_id") != run_id:
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        web_search_summaries.append(
            {
                "path": str(web_search_path),
                "raw_output_path": payload.get("raw_output_path"),
                "query": payload.get("query"),
                "canonical_query": payload.get("canonical_query"),
                "query_hash": payload.get("query_hash"),
                "result_count": len(results),
                "top_urls": [str(item.get("url")) for item in results[:5] if isinstance(item, dict) and item.get("url")],
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
        "web_search_dir": str(web_search_dir) if web_search_dir is not None else None,
        "transcript_dir": str(transcript_dir),
        "article_path": str(article_path),
        "search_count": len(search_summaries),
        "web_search_count": len(web_search_summaries),
        "transcript_count": len(transcript_summaries),
        "article_referenced_video_count": len(article_video_ids),
        "searches": search_summaries,
        "web_searches": web_search_summaries,
        "transcripts": transcript_summaries,
        "sources": sorted(sources.values(), key=lambda item: str(item.get("video_id"))),
    }
    _write_json(path, manifest)
    return manifest


def _timestamp_link_count(text: str) -> int:
    return len(re.findall(r"https://(?:www\.)?(?:youtube\.com/watch\?[^)\s]+|youtu\.be/[^)\s]+)[^)\s]*(?:[?&]t=|&amp;t=)\d+s?", text))


TIMESTAMP_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[([^\]\n]+)\]\((https://(?:www\.)?(?:youtube\.com/watch\?[^)\s]+|youtu\.be/[^)\s]+)[^)\s]*)\)"
)
OLD_TIMESTAMP_CITATION_RE = re.compile(
    r"[\(（][A-Za-z0-9\u4e00-\u9fff][^()（）\[\]\n]{0,40}\s+"
    r"(?P<link>\[[^\]\n]*\d{2}:\d{2}:\d{2}\]"
    r"\(https://(?:www\.)?(?:youtube\.com/watch\?[^)\s]+|youtu\.be/[^)\s]+)[^)\s]*\))[\)）]"
)


def _format_timestamp_text(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_youtube_timestamp_value(value: str) -> int | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned.endswith("s") and cleaned[:-1].isdigit():
        return int(cleaned[:-1])
    if cleaned.isdigit():
        return int(cleaned)
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?", cleaned)
    if not match or not any(match.groups()):
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _timestamp_seconds_from_youtube_url(url: str) -> int | None:
    parsed = urlparse(url.replace("&amp;", "&"))
    query = parse_qs(parsed.query)
    values = query.get("t") or query.get("start")
    if not values:
        return None
    return _parse_youtube_timestamp_value(values[0])


def _video_id_from_youtube_url(url: str) -> str | None:
    parsed = urlparse(url.replace("&amp;", "&"))
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
    if "youtube.com" in host:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        return video_id if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
    return None


GENERIC_TIMESTAMP_LINK_LABELS = {
    "at",
    "here",
    "link",
    "source",
    "start",
    "time",
    "timestamp",
    "video",
    "watch",
    "youtube",
    "来源",
    "链接",
    "视频",
    "时间",
    "时间戳",
}


def _timestamp_link_intro(label: str, expected_label: str) -> str:
    cleaned = label.strip().strip("`").strip()
    cleaned = cleaned.replace(expected_label, " ")
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", cleaned)
    cleaned = re.sub(r"\b\d+\s*s\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s:：,，.。;；|｜\-–—▶▷►]+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.lower() in GENERIC_TIMESTAMP_LINK_LABELS:
        return ""
    return cleaned


def _timestamp_link_cue(label: str, expected_label: str) -> str:
    cleaned = label.strip().strip("`").strip()
    if cleaned.endswith(expected_label):
        cleaned = cleaned[: -len(expected_label)].strip()
    else:
        cleaned = _timestamp_link_intro(cleaned, expected_label)
    cleaned = re.sub(r"^[\[(（]+|[\])）]+$", "", cleaned).strip()
    cleaned = re.sub(r"[\s:：,，.。;；|｜\-–—▶▷►]+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.lower() in GENERIC_TIMESTAMP_LINK_LABELS:
        return ""
    return cleaned


def _source_cue_from_title(title: str | None, channel: str | None = None) -> str:
    value = re.sub(r"\s+", " ", (title or "").strip())
    match = re.search(r"对([^的：:，,|丨\-\s]{1,8})的.{0,4}访谈", value)
    if match:
        return f"{match.group(1)}访谈"
    value = re.sub(r"^\s*\d+\s*[.、:：-]\s*", "", value)
    value = re.split(r"[|丨:：\-—]", value, maxsplit=1)[0].strip()
    if re.search(r"[\u4e00-\u9fff]", value):
        cue = value[:32].strip()
        cue = re.sub(r"[，,。.;；:：|｜\-–—▶▷►\s]+$", "", cue).strip()
        return cue if cue else "视频片段"
    words = re.findall(r"[A-Za-z0-9]+", value)
    if words:
        return " ".join(words[:6])
    channel_value = re.sub(r"\s+", " ", (channel or "").strip())
    return channel_value[:32].strip() if channel_value else "视频片段"


def _source_cues_from_manifest(sources_manifest: dict[str, object] | None) -> dict[str, str]:
    sources = (sources_manifest or {}).get("sources")
    if not isinstance(sources, list):
        return {}
    cues: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        video_id = source.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            continue
        title = source.get("title") if isinstance(source.get("title"), str) else None
        channel = source.get("channel") if isinstance(source.get("channel"), str) else None
        cues[video_id] = _source_cue_from_title(title, channel)
    return cues


def _line_prefix_timestamp_intro(text: str, link_start: int) -> str:
    line_start = text.rfind("\n", 0, link_start) + 1
    prefix = text[line_start:link_start].strip()
    prefix = re.sub(r"^[>\-\*\d.)\s]+", "", prefix).strip()
    prefix = re.sub(r"[，,。.;；:：|｜\-–—▶▷►\s]+$", "", prefix).strip()
    prefix = re.split(r"[。！？!?；;]|\)\s*", prefix)[-1].strip()
    prefix = re.split(r"[\(（]", prefix)[-1].strip()
    if len(prefix) > 18:
        prefix = re.split(r"\s+", prefix)[-1].strip()
    prefix = prefix.strip("*_` ()（）[]【】")
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", prefix):
        return ""
    if len(prefix) > 18:
        return ""
    if re.search(r"[，,。！？!?；;:：]", prefix):
        return ""
    return prefix


def _timestamp_link_label_is_valid(label: str, expected_label: str) -> bool:
    return bool(_timestamp_link_cue(label, expected_label)) and label.strip().strip("`").strip().endswith(expected_label)


def _collapse_old_timestamp_citations(text: str) -> tuple[str, int]:
    normalized = text
    replacement_count = 0
    for _ in range(3):
        normalized, count = OLD_TIMESTAMP_CITATION_RE.subn(lambda match: match.group("link"), normalized)
        replacement_count += count
        if count == 0:
            break
    return normalized, replacement_count


def normalize_youtube_timestamp_link_text(
    article_path: Path,
    *,
    sources_manifest: dict[str, object] | None = None,
) -> int:
    if not article_path.exists():
        return 0
    text = article_path.read_text(encoding="utf-8", errors="replace")
    replacement_count = 0
    source_cues = _source_cues_from_manifest(sources_manifest)

    def replace_link(match: re.Match[str]) -> str:
        nonlocal replacement_count
        label = match.group(1)
        url = match.group(2)
        seconds = _timestamp_seconds_from_youtube_url(url)
        if seconds is None:
            return match.group(0)
        expected_label = _format_timestamp_text(seconds)
        video_id = _video_id_from_youtube_url(url)
        cue = source_cues.get(video_id or "", "") if video_id else ""
        if not cue:
            cue = _timestamp_link_cue(label, expected_label)
        if not cue:
            cue = _line_prefix_timestamp_intro(text, match.start())
        if not cue:
            return match.group(0)
        new_label = f"{cue} {expected_label}"
        if label == new_label:
            return match.group(0)
        replacement_count += 1
        return f"[{new_label}]({url})"

    normalized = TIMESTAMP_MARKDOWN_LINK_RE.sub(replace_link, text)
    normalized, collapsed_count = _collapse_old_timestamp_citations(normalized)
    replacement_count += collapsed_count
    if normalized != text:
        article_path.write_text(normalized, encoding="utf-8")
    return replacement_count


def _timestamp_link_text_issue_count(text: str) -> int:
    issue_count = 0
    for match in TIMESTAMP_MARKDOWN_LINK_RE.finditer(text):
        seconds = _timestamp_seconds_from_youtube_url(match.group(2))
        if seconds is None:
            continue
        if not _timestamp_link_label_is_valid(match.group(1), _format_timestamp_text(seconds)):
            issue_count += 1
    return issue_count


def _timestamp_link_intro_issue_count(text: str) -> int:
    issue_count = 0
    for match in TIMESTAMP_MARKDOWN_LINK_RE.finditer(text):
        seconds = _timestamp_seconds_from_youtube_url(match.group(2))
        if seconds is None:
            continue
        if not _timestamp_link_cue(match.group(1), _format_timestamp_text(seconds)):
            issue_count += 1
    return issue_count


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
        "timestamp_link_text_issue_count": _timestamp_link_text_issue_count(text),
        "timestamp_link_intro_issue_count": _timestamp_link_intro_issue_count(text),
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
    research_plan_path: Path | None = None,
    video_enrichment_manifest_path: Path | None = None,
    selection_manifest_path: Path | None = None,
    evidence_manifest_path: Path | None = None,
) -> dict[str, object]:
    issues: list[dict[str, object]] = []

    def add_issue(severity: str, code: str, message: str) -> None:
        issues.append({"severity": severity, "code": code, "message": message})

    search_count = int(sources_manifest.get("search_count") or 0)
    web_search_count = int(sources_manifest.get("web_search_count") or 0)
    transcript_count = int(sources_manifest.get("transcript_count") or 0)
    timestamp_link_count = int((article_manifest or {}).get("timestamp_link_count") or 0)
    timestamp_link_text_issue_count = int((article_manifest or {}).get("timestamp_link_text_issue_count") or 0)
    timestamp_link_intro_issue_count = int((article_manifest or {}).get("timestamp_link_intro_issue_count") or 0)
    referenced_ids = set((article_manifest or {}).get("referenced_video_ids") or [])
    transcript_ids = set((article_manifest or {}).get("transcript_video_ids") or [])
    evidence_manifest = _read_json_object(evidence_manifest_path) if evidence_manifest_path is not None else None
    evidence_transcript_count = int((evidence_manifest or {}).get("transcript_count") or 0)
    evidence_success_count = int((evidence_manifest or {}).get("success_count") or 0)
    evidence_failed_count = int((evidence_manifest or {}).get("failed_count") or 0)

    if not article_path.exists() or article_path.stat().st_size == 0:
        add_issue("error", "article_missing", "article.md was not generated or is empty.")
    if transcript_count < 1:
        add_issue("error", "no_transcripts", "No transcript artifacts were generated for this run.")
    if article_path.exists() and article_path.stat().st_size > 0 and timestamp_link_count < 1:
        add_issue("warning", "no_timestamp_links", "Article does not contain clickable YouTube timestamp links.")
    if timestamp_link_text_issue_count:
        add_issue(
            "warning",
            "timestamp_link_text_inconsistent",
            "Some YouTube timestamp links do not use '<short video title> HH:MM:SS' as their visible text.",
        )
    if timestamp_link_intro_issue_count:
        add_issue(
            "warning",
            "timestamp_link_intro_missing",
            "Some YouTube timestamp links do not include a short video-title cue before the timestamp inside the link text.",
        )
    if research_mode == "wide":
        if search_count < 2:
            add_issue("warning", "wide_search_count_low", "Wide mode produced fewer than two search artifacts.")
        if research_plan_path is not None and not research_plan_path.exists():
            add_issue("warning", "research_plan_missing", "Research discovery plan artifact is missing.")
        if video_enrichment_manifest_path is not None and not video_enrichment_manifest_path.exists():
            add_issue("warning", "video_enrichment_missing", "SerpApi video enrichment artifact is missing.")
        if selection_manifest_path is not None and not selection_manifest_path.exists():
            add_issue("warning", "selection_manifest_missing", "Selection candidate artifact is missing.")
        if evidence_manifest_path is not None:
            if not evidence_manifest_path.exists():
                add_issue("warning", "evidence_manifest_missing", "Evidence extraction manifest is missing.")
            elif evidence_success_count < 1:
                add_issue("error", "evidence_generation_empty", "Evidence extraction did not produce usable evidence.")
            elif evidence_failed_count:
                add_issue(
                    "warning",
                    "evidence_generation_partial",
                    "Some transcript evidence extraction jobs failed; article coverage may be partial.",
                )

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
        "web_search_count": web_search_count,
        "transcript_count": transcript_count,
        "evidence_transcript_count": evidence_transcript_count,
        "evidence_success_count": evidence_success_count,
        "evidence_failed_count": evidence_failed_count,
        "timestamp_link_count": timestamp_link_count,
        "timestamp_link_text_issue_count": timestamp_link_text_issue_count,
        "timestamp_link_intro_issue_count": timestamp_link_intro_issue_count,
        "referenced_video_count": len(referenced_ids),
        "referenced_without_transcript": missing_transcripts,
        "issues": issues,
    }
    if selection_manifest_path is not None and selection_manifest_path.exists():
        selection_manifest = _read_json_object(selection_manifest_path)
        if selection_manifest is not None:
            report["selection_candidate_count"] = selection_manifest.get("candidate_count")
            report["search_round_count"] = selection_manifest.get("search_round_count")
    if evidence_manifest is not None:
        report["evidence_manifest_path"] = str(evidence_manifest_path)
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
    web_search_dir: Path | None = None,
    research_plan_path: Path | None = None,
    video_enrichment_manifest_path: Path | None = None,
    selection_manifest_path: Path | None = None,
    evidence_manifest_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sources_manifest = write_sources_manifest(
        sources_manifest_path,
        search_dir=search_dir,
        web_search_dir=web_search_dir,
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
        research_plan_path=research_plan_path,
        video_enrichment_manifest_path=video_enrichment_manifest_path,
        selection_manifest_path=selection_manifest_path,
        evidence_manifest_path=evidence_manifest_path,
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


def resolve_evidence_model(env_values: dict[str, str], fallback_model: str | None) -> str | None:
    for key in EVIDENCE_MODEL_ENV_KEYS:
        value = os.environ.get(key) or env_values.get(key)
        if value and value.strip():
            return value.strip()
    return fallback_model


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


async def _consume_sdk_query_text(prompt: str, options: ClaudeAgentOptions, logger: object, event_name: str) -> tuple[str, str | None]:
    texts: list[str] = []
    sdk_error_message: str | None = None
    try:
        async for message in query(prompt=prompt, options=options):
            sdk_error_message = _sdk_error_message(message) or sdk_error_message
            logger.info(format_log_event(event_name, serialize_message(message)))
            for text in _iter_text_blocks(message):
                texts.append(text)
                logger.info(format_log_text_block("message_text", text))
                print(text)
    except Exception as exc:
        raise AgentRunError(sdk_error_message or str(exc)) from exc
    return "\n".join(texts), sdk_error_message


def _sdk_error_message(message: object) -> str | None:
    is_error = bool(getattr(message, "is_error", False))
    status = getattr(message, "api_error_status", None)
    result = getattr(message, "result", None)
    if not is_error and not status:
        return None
    result_text = str(result).strip() if result else ""
    if status:
        prefix = f"API Error {status}"
        if result_text.lower().startswith("api error:"):
            detail = result_text.split(":", 1)[1].strip()
            detail = re.sub(rf"^{re.escape(str(status))}\s*", "", detail).strip()
            return f"{prefix}: {detail}" if detail else prefix
        if result_text:
            return f"{prefix}: {result_text}"
        return prefix
    return result_text or None


class AgentRunError(RuntimeError):
    pass


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


def list_transcript_contexts(transcript_dir: Path, *, run_id: str | None = None) -> list[Path]:
    if not transcript_dir.exists():
        return []
    paths = sorted(path for path in transcript_dir.glob("*.transcript.json") if path.stat().st_size > 0)
    if run_id is None:
        return paths
    current_run_paths: list[Path] = []
    for path in paths:
        payload = _read_json_object(path)
        if payload is not None and payload.get("run_id") == run_id:
            current_run_paths.append(path)
    return current_run_paths


def _transcript_video_id(transcript_path: Path) -> str:
    payload = _read_json_object(transcript_path) or {}
    video = payload.get("video")
    if isinstance(video, dict) and video.get("video_id"):
        return str(video["video_id"])
    return transcript_path.stem.removesuffix(".transcript")


def _evidence_path_for_transcript(evidence_dir: Path, transcript_path: Path) -> Path:
    return evidence_dir / f"{_slugify(_transcript_video_id(transcript_path), fallback=transcript_path.stem)}.evidence.json"


def _video_url_from_id(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def write_web_evidence_cards(
    *,
    web_search_dir: Path,
    web_evidence_path: Path,
    run_id: str,
    limit: int = WEB_EVIDENCE_MAX_CARDS,
) -> dict[str, object]:
    cards: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    web_search_files = sorted(web_search_dir.glob("*.web-search.json")) if web_search_dir.exists() else []
    for search_path in web_search_files:
        payload = _read_json_object(search_path)
        if payload is None or payload.get("run_id") != run_id:
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            title = str(result.get("title") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)
            cards.append(
                {
                    "source_kind": "web",
                    "query": result.get("query") or payload.get("query"),
                    "title": title,
                    "url": url,
                    "source": result.get("source"),
                    "date": result.get("date"),
                    "rank": result.get("rank"),
                    "result_type": result.get("result_type"),
                    "claim_summary": snippet or title,
                    "snippet": snippet,
                    "search_path": str(search_path),
                    "usage_note": "Snippet-level web evidence; use for background or corroboration, not unsupported strong claims.",
                }
            )
            if len(cards) >= limit:
                break
        if len(cards) >= limit:
            break
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "source_kind": "web",
        "web_search_dir": str(web_search_dir),
        "card_count": len(cards),
        "cards": cards,
    }
    _write_json(web_evidence_path, manifest)
    return manifest


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _transcript_fetch_candidates_from_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_items: object = None
    for key in ("selected_videos", "videos", "candidates", "selected_candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            raw_items = value
            break
    if not isinstance(raw_items, list):
        return []

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url") or item.get("link") or "")
        video_id = str(item.get("video_id") or item.get("videoId") or extract_youtube_video_id(raw_url) or "").strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        candidates.append(
            {
                "video_id": video_id,
                "url": raw_url or _video_url_from_id(video_id),
                "title": str(item.get("title") or f"Video {video_id}"),
                "channel": str(item.get("channel") or "Unknown channel"),
                "priority": int(item.get("priority") or index) if str(item.get("priority") or index).isdigit() else index,
                "reason": str(item.get("reason") or ""),
                "score": item.get("score"),
            }
        )
    return sorted(candidates, key=lambda item: (int(item.get("priority") or 999), -_safe_float(item.get("score"))))


def transcript_fetch_candidates(
    *,
    plan_path: Path,
    selection_manifest_path: Path,
    limit: int = WIDE_TRANSCRIPT_PROBE_LIMIT,
) -> list[dict[str, object]]:
    plan = _read_json_object(plan_path)
    candidates = _transcript_fetch_candidates_from_payload(plan or {})
    if not candidates:
        selection_manifest = _read_json_object(selection_manifest_path)
        candidates = _transcript_fetch_candidates_from_payload(selection_manifest or {})
    return candidates[:limit]


def write_transcript_fetch_manifest(
    path: Path,
    *,
    run_id: str,
    candidates: list[dict[str, object]],
    successes: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "candidate_count": len(candidates),
        "success_count": len(successes),
        "failed_count": len(failures),
        "successes": successes,
        "failures": failures,
    }
    _write_json(path, manifest)
    return manifest


def _extract_json_object_from_text(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _supplemental_queries_from_payload(
    payload: dict[str, object] | None,
    *,
    key: str = "supplemental_queries",
    limit: int = WIDE_SUPPLEMENTAL_QUERY_COUNT,
) -> list[dict[str, str]]:
    raw_queries = (payload or {}).get(key)
    if not isinstance(raw_queries, list):
        return []
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_queries:
        if isinstance(item, str):
            query_text = item
            reason = ""
        elif isinstance(item, dict):
            query_text = str(item.get("query") or "").strip()
            reason = str(item.get("reason") or "").strip()
        else:
            continue
        normalized = re.sub(r"\s+", " ", query_text).strip()
        seen_key = normalized.casefold()
        if not normalized or seen_key in seen:
            continue
        seen.add(seen_key)
        queries.append({"query": normalized, "reason": reason})
        if len(queries) >= limit:
            break
    return queries


def _youtube_queries_from_payload(payload: dict[str, object] | None) -> list[dict[str, str]]:
    queries = _supplemental_queries_from_payload(
        payload,
        key="supplemental_youtube_queries",
        limit=WIDE_SUPPLEMENTAL_QUERY_COUNT,
    )
    if queries:
        return queries
    return _supplemental_queries_from_payload(
        payload,
        key="supplemental_queries",
        limit=WIDE_SUPPLEMENTAL_QUERY_COUNT,
    )


def _web_queries_from_payload(payload: dict[str, object] | None) -> list[dict[str, str]]:
    return _supplemental_queries_from_payload(
        payload,
        key="supplemental_web_queries",
        limit=WIDE_SUPPLEMENTAL_WEB_QUERY_COUNT,
    )


def _initial_youtube_queries_from_payload(payload: dict[str, object] | None) -> list[dict[str, object]]:
    raw_queries = (payload or {}).get("youtube_search_queries")
    if not isinstance(raw_queries, list):
        raw_queries = (payload or {}).get("queries")
    if not isinstance(raw_queries, list):
        return []
    queries: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw_queries:
        if isinstance(item, str):
            query_text = item
            reason = "LLM-generated initial YouTube discovery query"
            language = "mixed"
        elif isinstance(item, dict):
            query_text = str(item.get("query") or "").strip()
            reason = str(item.get("reason") or "").strip() or "LLM-generated initial YouTube discovery query"
            language = str(item.get("language") or "mixed").strip() or "mixed"
        else:
            continue
        normalized = re.sub(r"\s+", " ", query_text).strip()
        seen_key = normalized.casefold()
        if not normalized or seen_key in seen:
            continue
        seen.add(seen_key)
        queries.append(
            {
                "round": 1,
                "query": normalized,
                "intent": reason,
                "expected_source_type": "interview/podcast/talk/panel/keynote",
                "language": language,
            }
        )
        if len(queries) >= INITIAL_DISCOVERY_QUERY_COUNT:
            break
    return queries


def write_initial_search_query_plan(
    path: Path,
    *,
    run_id: str,
    queries: list[dict[str, object]],
    raw_text: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "queries": queries,
        "raw_text": raw_text,
    }
    _write_json(path, payload)
    return payload


async def plan_initial_search_queries(
    *,
    input_value: str,
    question: str,
    initial_query_plan_path: Path,
    run_id: str,
    options: ClaudeAgentOptions,
    logger: object,
) -> list[dict[str, object]]:
    prompt = build_initial_search_query_prompt(input_value=input_value, question=question)
    logger.info(format_log_event("initial_query_planner_prompt_ready", {"prompt_chars": len(prompt)}))
    text, _ = await _consume_sdk_query_text(prompt, options, logger, "initial_query_planner_sdk_message")
    payload = _extract_json_object_from_text(text)
    queries = _initial_youtube_queries_from_payload(payload)
    write_initial_search_query_plan(
        initial_query_plan_path,
        run_id=run_id,
        queries=queries,
        raw_text=text,
    )
    logger.info(
        format_log_event(
            "initial_query_plan_ready",
            {
                "query_count": len(queries),
                "initial_query_plan_path": str(initial_query_plan_path),
            },
        )
    )
    if not queries:
        raise AgentRunError("initial search query planner returned no YouTube queries")
    return queries


def write_query_plan(
    path: Path,
    *,
    run_id: str,
    youtube_queries: list[dict[str, str]],
    web_queries: list[dict[str, str]],
    raw_text: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "supplemental_queries": youtube_queries,
        "supplemental_youtube_queries": youtube_queries,
        "supplemental_web_queries": web_queries,
        "raw_text": raw_text,
    }
    _write_json(path, payload)
    return payload


async def plan_supplemental_queries(
    *,
    question: str,
    selection_manifest_path: Path,
    query_plan_path: Path,
    run_id: str,
    options: ClaudeAgentOptions,
    logger: object,
) -> dict[str, object]:
    prompt = build_query_planner_prompt(question=question, selection_manifest_path=selection_manifest_path)
    logger.info(format_log_event("query_planner_prompt_ready", {"prompt_chars": len(prompt)}))
    text, _ = await _consume_sdk_query_text(prompt, options, logger, "query_planner_sdk_message")
    payload = _extract_json_object_from_text(text)
    youtube_queries = _youtube_queries_from_payload(payload)
    web_queries = _web_queries_from_payload(payload)
    plan = write_query_plan(
        query_plan_path,
        run_id=run_id,
        youtube_queries=youtube_queries,
        web_queries=web_queries,
        raw_text=text,
    )
    logger.info(
        format_log_event(
            "query_plan_ready",
            {
                "youtube_query_count": len(youtube_queries),
                "web_query_count": len(web_queries),
                "query_plan_path": str(query_plan_path),
            },
        )
    )
    return plan


async def run_supplemental_searches(
    *,
    query_plan: dict[str, object],
    search_dir: Path,
    run_id: str,
    progress_sink: ProgressSink | None,
    logger: object,
    max_concurrency: int = WIDE_SUPPLEMENTAL_SEARCH_MAX_CONCURRENCY,
) -> list[str]:
    queries = _youtube_queries_from_payload(query_plan)
    if not queries:
        return []
    _emit_progress(
        progress_sink,
        "phase_progress",
        "source_fetch",
        "正在并发执行补充 YouTube 搜索",
        data={"query_count": len(queries), "max_concurrency": max_concurrency},
    )
    paths: list[str] = []
    loop = asyncio.get_running_loop()

    def run_pool() -> list[str]:
        results: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
            future_map = {
                executor.submit(search_youtube_context, item["query"], output_dir=search_dir, run_id=run_id): item
                for item in queries
            }
            for future in wait(future_map.keys()).done:
                item = future_map[future]
                try:
                    results.append(str(future.result()))
                except Exception as exc:
                    logger.warning(format_log_event("supplemental_search_failed", {"query": item["query"], "error_message": str(exc)}))
        return sorted(results)

    paths = await loop.run_in_executor(None, run_pool)
    logger.info(format_log_event("supplemental_searches_completed", {"search_count": len(paths), "paths": paths}))
    return paths


async def run_supplemental_web_searches(
    *,
    query_plan: dict[str, object],
    web_search_dir: Path,
    run_id: str,
    progress_sink: ProgressSink | None,
    logger: object,
    max_concurrency: int = WIDE_SUPPLEMENTAL_SEARCH_MAX_CONCURRENCY,
) -> list[str]:
    queries = _web_queries_from_payload(query_plan)
    if not queries:
        return []
    _emit_progress(
        progress_sink,
        "phase_progress",
        "source_fetch",
        "正在并发执行补充 Web 搜索",
        data={"query_count": len(queries), "max_concurrency": max_concurrency},
    )
    loop = asyncio.get_running_loop()

    def run_pool() -> list[str]:
        results: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
            future_map = {
                executor.submit(search_web_context, item["query"], output_dir=web_search_dir, run_id=run_id): item
                for item in queries
            }
            for future in wait(future_map.keys()).done:
                item = future_map[future]
                try:
                    results.append(str(future.result()))
                except Exception as exc:
                    logger.warning(format_log_event("supplemental_web_search_failed", {"query": item["query"], "error_message": str(exc)}))
        return sorted(results)

    paths = await loop.run_in_executor(None, run_pool)
    logger.info(format_log_event("supplemental_web_searches_completed", {"search_count": len(paths), "paths": paths}))
    return paths


def _score_transcript_candidate(candidate: dict[str, object], *, rank: int) -> float:
    score = _safe_float(candidate.get("score"))
    title = str(candidate.get("title") or "").casefold()
    channel = str(candidate.get("channel") or "").casefold()
    published = str(candidate.get("published_date") or "").casefold()
    bucket = str(candidate.get("source_bucket") or "")
    if any(term in title for term in ("interview", "podcast", "访谈", "对话", "discussion", "debate", "keynote", "panel")):
        score += 8.0
    if any(term in title for term in ("ai", "agi", "agent", "llm", "人工智能", "大模型", "智能体")):
        score += 6.0
    if any(name in title or name in channel for name in ("sam altman", "dario", "demis", "jensen", "huang", "reid hoffman", "musk", "罗福莉", "田渊栋", "周鸿祎")):
        score += 5.0
    if any(term in published for term in ("day", "week", "month", "天", "周", "个月")):
        score += 3.0
    if bucket == "related_videos":
        score -= 4.0
    score -= rank * 0.05
    return score


def _search_candidates_from_files(search_dir: Path, *, run_id: str) -> list[dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}
    rank = 0
    for path in sorted(search_dir.glob("*.search.json")):
        payload = _read_json_object(path)
        if payload is None or payload.get("run_id") != run_id:
            continue
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            continue
        for item in raw_candidates:
            if not isinstance(item, dict) or not item.get("video_id"):
                continue
            rank += 1
            video_id = str(item["video_id"])
            candidate = dict(item)
            candidate["priority_score"] = _score_transcript_candidate(candidate, rank=rank)
            existing = candidates.get(video_id)
            if existing is None or _safe_float(candidate["priority_score"]) > _safe_float(existing.get("priority_score")):
                candidates[video_id] = candidate
    return sorted(candidates.values(), key=lambda item: _safe_float(item.get("priority_score")), reverse=True)


def generate_transcript_fetch_plan_from_searches(
    *,
    plan_path: Path,
    search_dir: Path,
    run_id: str,
    limit: int = WIDE_TRANSCRIPT_PROBE_LIMIT,
) -> dict[str, object]:
    selected: list[dict[str, object]] = []
    channel_counts: dict[str, int] = {}
    for candidate in _search_candidates_from_files(search_dir, run_id=run_id):
        channel = str(candidate.get("channel") or "Unknown channel")
        if channel_counts.get(channel, 0) >= 2:
            continue
        video_id = str(candidate.get("video_id") or "")
        if not video_id:
            continue
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        selected.append(
            {
                "video_id": video_id,
                "url": str(candidate.get("url") or _video_url_from_id(video_id)),
                "title": str(candidate.get("title") or f"Video {video_id}"),
                "channel": channel,
                "priority": len(selected) + 1,
                "reason": "programmatic selection from merged search candidates",
                "score": candidate.get("priority_score"),
            }
        )
        if len(selected) >= limit:
            break
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "selected_videos": selected,
    }
    _write_json(plan_path, payload)
    return payload


def _fetch_one_planned_transcript(
    candidate: dict[str, object],
    *,
    transcript_dir: Path,
    run_id: str,
) -> dict[str, object]:
    raw_input = str(candidate.get("url") or _video_url_from_id(str(candidate["video_id"])))
    output_path = fetch_transcript_context(raw_input, output_dir=transcript_dir, run_id=run_id)
    payload = _read_json_object(output_path) or {}
    video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
    return {
        "video_id": str(video.get("video_id") or candidate.get("video_id")),
        "title": str(video.get("title") or candidate.get("title") or ""),
        "channel": str(video.get("channel") or candidate.get("channel") or ""),
        "path": str(output_path),
        "source_kind": str(payload.get("source_kind") or ""),
    }


async def fetch_wide_transcripts_from_plan(
    *,
    plan_path: Path,
    selection_manifest_path: Path,
    transcript_dir: Path,
    manifest_path: Path,
    run_id: str,
    progress_sink: ProgressSink | None,
    logger: object,
    target_count: int = WIDE_TRANSCRIPT_TARGET_COUNT,
    probe_limit: int = WIDE_TRANSCRIPT_PROBE_LIMIT,
    max_concurrency: int = WIDE_TRANSCRIPT_MAX_CONCURRENCY,
) -> dict[str, object]:
    candidates = transcript_fetch_candidates(
        plan_path=plan_path,
        selection_manifest_path=selection_manifest_path,
        limit=probe_limit,
    )
    existing_paths = list_transcript_contexts(transcript_dir, run_id=run_id)
    existing_ids = {_transcript_video_id(path) for path in existing_paths}
    candidates = [candidate for candidate in candidates if str(candidate.get("video_id")) not in existing_ids]
    _emit_progress(
        progress_sink,
        "phase_started",
        "source_fetch",
        "正在并发获取计划中的转录上下文",
        data={"candidate_count": len(candidates), "target_count": target_count, "max_concurrency": max_concurrency},
    )

    successes: list[dict[str, object]] = [
        {"video_id": _transcript_video_id(path), "path": str(path), "reused": True}
        for path in existing_paths
    ]
    failures: list[dict[str, object]] = []
    if not candidates or len(successes) >= target_count:
        return write_transcript_fetch_manifest(
            manifest_path,
            run_id=run_id,
            candidates=candidates,
            successes=successes,
            failures=failures,
        )

    loop = asyncio.get_running_loop()

    def run_pool() -> dict[str, object]:
        pending: set[Future[dict[str, object]]] = set()
        future_candidates: dict[Future[dict[str, object]], dict[str, object]] = {}
        next_index = 0

        def submit_next(executor: ThreadPoolExecutor) -> None:
            nonlocal next_index
            while len(pending) < max(1, max_concurrency) and next_index < len(candidates):
                candidate = candidates[next_index]
                next_index += 1
                future = executor.submit(
                    _fetch_one_planned_transcript,
                    candidate,
                    transcript_dir=transcript_dir,
                    run_id=run_id,
                )
                pending.add(future)
                future_candidates[future] = candidate

        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
            submit_next(executor)
            while pending and len(successes) < target_count:
                done, pending_remainder = wait(pending, return_when=FIRST_COMPLETED)
                pending = set(pending_remainder)
                for future in done:
                    candidate = future_candidates.pop(future, {})
                    try:
                        successes.append(future.result())
                    except Exception as exc:
                        failures.append(
                            {
                                "video_id": str(candidate.get("video_id") or ""),
                                "url": str(candidate.get("url") or ""),
                                "title": str(candidate.get("title") or ""),
                                "error_message": str(exc),
                            }
                        )
                    _emit_progress(
                        progress_sink,
                        "phase_progress",
                        "source_fetch",
                        "并发转录获取进度",
                        data={
                            "completed": len(successes) + len(failures),
                            "success_count": len(successes),
                            "failed_count": len(failures),
                            "target_count": target_count,
                        },
                    )
                submit_next(executor)
            for future in pending:
                future.cancel()

        return write_transcript_fetch_manifest(
            manifest_path,
            run_id=run_id,
            candidates=candidates,
            successes=successes,
            failures=failures,
        )

    manifest = await loop.run_in_executor(None, run_pool)
    logger.info(format_log_event("parallel_transcript_fetch_completed", manifest))
    return manifest


def write_evidence_manifest(
    path: Path,
    *,
    run_id: str,
    transcript_paths: list[Path],
    successes: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "transcript_count": len(transcript_paths),
        "success_count": len(successes),
        "failed_count": len(failures),
        "evidence_files": successes,
        "failures": failures,
    }
    _write_json(path, manifest)
    return manifest


async def _consume_sdk_query(prompt: str, options: ClaudeAgentOptions, logger: object, event_name: str) -> str | None:
    _, sdk_error_message = await _consume_sdk_query_text(prompt, options, logger, event_name)
    return sdk_error_message


async def _extract_one_evidence(
    *,
    question: str,
    transcript_path: Path,
    evidence_dir: Path,
    options: ClaudeAgentOptions,
    logger: object,
) -> dict[str, object]:
    evidence_path = _evidence_path_for_transcript(evidence_dir, transcript_path)
    prompt = build_evidence_prompt(
        question=question,
        transcript_path=transcript_path,
        evidence_path=evidence_path,
    )
    logger.info(
        format_log_event(
            "evidence_extract_started",
            {"transcript_path": str(transcript_path), "evidence_path": str(evidence_path), "prompt_chars": len(prompt)},
        )
    )
    sdk_error_message: str | None = None
    try:
        sdk_error_message = await _consume_sdk_query(prompt, options, logger, "evidence_sdk_message")
    except Exception as exc:
        error_message = sdk_error_message or str(exc)
        logger.exception(format_log_event("evidence_extract_failed", {"transcript_path": str(transcript_path), "error_message": error_message}))
        raise AgentRunError(error_message) from exc

    payload = _read_json_object(evidence_path)
    if payload is None:
        raise AgentRunError(f"evidence file was not generated or is invalid JSON: {evidence_path}")
    video_id = str(payload.get("video_id") or _transcript_video_id(transcript_path))
    cards = payload.get("cards")
    card_count = len(cards) if isinstance(cards, list) else 0
    return {
        "video_id": video_id,
        "path": str(evidence_path),
        "transcript_path": str(transcript_path),
        "relevance": payload.get("relevance"),
        "excluded": bool(payload.get("excluded", False)),
        "card_count": card_count,
    }


async def extract_evidence_cards(
    *,
    question: str,
    transcript_dir: Path,
    evidence_dir: Path,
    evidence_manifest_path: Path,
    run_id: str,
    options: ClaudeAgentOptions,
    logger: object,
    progress_sink: ProgressSink | None,
    max_concurrency: int = EVIDENCE_MAX_CONCURRENCY,
) -> dict[str, object]:
    transcript_paths = list_transcript_contexts(transcript_dir, run_id=run_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress_sink,
        "phase_started",
        "evidence_extract",
        "正在提取核心证据卡片",
        data={"transcript_count": len(transcript_paths), "evidence_dir": str(evidence_dir)},
    )
    if not transcript_paths:
        write_evidence_manifest(
            evidence_manifest_path,
            run_id=run_id,
            transcript_paths=[],
            successes=[],
            failures=[],
        )
        raise AgentRunError("no transcript artifacts available for evidence extraction")

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    completed = 0

    async def run_one(transcript_path: Path) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        async with semaphore:
            try:
                return (
                    await _extract_one_evidence(
                        question=question,
                        transcript_path=transcript_path,
                        evidence_dir=evidence_dir,
                        options=options,
                        logger=logger,
                    ),
                    None,
                )
            except Exception as exc:
                return None, {
                    "transcript_path": str(transcript_path),
                    "video_id": _transcript_video_id(transcript_path),
                    "error_message": str(exc),
                }

    tasks = [asyncio.create_task(run_one(path)) for path in transcript_paths]
    for task in asyncio.as_completed(tasks):
        success, failure = await task
        completed += 1
        if success is not None:
            successes.append(success)
        if failure is not None:
            failures.append(failure)
        _emit_progress(
            progress_sink,
            "phase_progress",
            "evidence_extract",
            "核心证据卡片提取进度",
            data={"completed": completed, "total": len(transcript_paths), "success_count": len(successes), "failed_count": len(failures)},
        )

    successes.sort(key=lambda item: str(item.get("video_id")))
    failures.sort(key=lambda item: str(item.get("video_id")))
    manifest = write_evidence_manifest(
        evidence_manifest_path,
        run_id=run_id,
        transcript_paths=transcript_paths,
        successes=successes,
        failures=failures,
    )
    logger.info(format_log_event("evidence_extract_completed", manifest))
    if not successes:
        raise AgentRunError("evidence extraction failed for every transcript")
    return manifest


def should_prepare_discovery(input_value: str, research_mode: str) -> bool:
    if research_mode == "wide":
        return True
    return extract_youtube_video_id(input_value) is None and not input_value.strip().startswith(("http://", "https://"))


async def _watch_artifact_progress(
    *,
    transcript_dir: Path,
    article_path: Path,
    progress_sink: ProgressSink | None,
    stop_event: asyncio.Event,
    start_article_on_transcript: bool = True,
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
            if start_article_on_transcript:
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
    paths["web_search_dir"].mkdir(parents=True, exist_ok=True)
    paths["transcript_dir"].mkdir(parents=True, exist_ok=True)
    paths["evidence_dir"].mkdir(parents=True, exist_ok=True)
    paths["articles_root"].mkdir(parents=True, exist_ok=True)
    article_dir = build_article_dir(paths["articles_root"])
    article_dir.mkdir(parents=True, exist_ok=False)
    article_path = article_dir / "article.md"
    run_id = article_dir.name
    run_manifest_path = paths["workspace_dir"] / RUN_MANIFEST_FILENAME
    sources_manifest_path = paths["workspace_dir"] / SOURCES_MANIFEST_FILENAME
    article_manifest_path = article_dir / ARTICLE_MANIFEST_FILENAME
    quality_report_path = paths["workspace_dir"] / QUALITY_REPORT_FILENAME
    evidence_manifest_path = paths["evidence_dir"] / EVIDENCE_MANIFEST_FILENAME
    web_evidence_path = paths["web_search_dir"] / WEB_EVIDENCE_FILENAME
    initial_query_plan_path = paths["workspace_dir"] / INITIAL_SEARCH_QUERY_PLAN_FILENAME
    query_plan_path = paths["workspace_dir"] / QUERY_PLAN_FILENAME
    transcript_fetch_plan_path = paths["workspace_dir"] / TRANSCRIPT_FETCH_PLAN_FILENAME
    transcript_fetch_manifest_path = paths["workspace_dir"] / TRANSCRIPT_FETCH_MANIFEST_FILENAME
    research_plan_path = paths["workspace_dir"] / RESEARCH_PLAN_FILENAME
    video_enrichment_manifest_path = paths["workspace_dir"] / VIDEO_ENRICHMENT_MANIFEST_FILENAME
    selection_manifest_path = paths["workspace_dir"] / SELECTION_MANIFEST_FILENAME
    options = build_agent_options(project_root, model=model)
    evidence_model = resolve_evidence_model(env_values, model)
    evidence_options = build_agent_options(project_root, model=evidence_model)
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
        evidence_manifest_path=evidence_manifest_path,
        research_plan_path=research_plan_path,
        video_enrichment_manifest_path=video_enrichment_manifest_path,
        selection_manifest_path=selection_manifest_path,
    )

    _emit_progress(
        progress_sink,
        "phase_started",
        "source_fetch",
        "正在准备视频研究上下文",
        data={"source_id": source_id, "research_mode": resolved_mode},
    )

    discovery_artifacts: dict[str, Path] = {}
    if should_prepare_discovery(input_value, resolved_mode):
        try:
            planned_discovery_queries: list[dict[str, object]] | None = None
            if resolved_mode == "wide":
                _emit_progress(
                    progress_sink,
                    "phase_progress",
                    "source_fetch",
                    "正在用 LLM 生成搜索引擎友好的初始 YouTube query",
                    data={"initial_query_plan_path": str(initial_query_plan_path)},
                )
                planned_discovery_queries = await plan_initial_search_queries(
                    input_value=input_value,
                    question=question,
                    initial_query_plan_path=initial_query_plan_path,
                    run_id=run_id,
                    options=options,
                    logger=logger,
                )
            _emit_progress(
                progress_sink,
                "phase_progress",
                "source_fetch",
                "正在并行搜索并富化 YouTube 候选源",
                data={
                    "search_dir": str(paths["search_dir"]),
                    "query_count": len(planned_discovery_queries) if planned_discovery_queries is not None else None,
                },
            )
            discovery_artifacts = prepare_research_discovery(
                input_value=input_value,
                question=question,
                research_mode=resolved_mode,
                workspace_dir=paths["workspace_dir"],
                search_dir=paths["search_dir"],
                run_id=run_id,
                planned_queries=planned_discovery_queries,
            )
        except Exception as exc:
            write_artifact_reports(
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                search_dir=paths["search_dir"],
                web_search_dir=paths["web_search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
                research_mode=resolved_mode,
                research_plan_path=research_plan_path,
                video_enrichment_manifest_path=video_enrichment_manifest_path,
                selection_manifest_path=selection_manifest_path,
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
                research_plan_path=research_plan_path,
                video_enrichment_manifest_path=video_enrichment_manifest_path,
                selection_manifest_path=selection_manifest_path,
            )
            raise

    prompt = (
        build_query_planner_prompt(question=question, selection_manifest_path=selection_manifest_path)
        if resolved_mode == "wide"
        else build_prompt(
            input_value=input_value,
            question=question,
            workspace_dir=str(paths["workspace_dir"]),
            search_dir=str(paths["search_dir"]),
            transcript_dir=str(paths["transcript_dir"]),
            article_path=str(article_path),
            run_id=run_id,
            research_mode=resolved_mode,
        )
    )

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
                "evidence_model": evidence_model,
                "search_dir": str(paths["search_dir"]),
                "transcript_dir": str(paths["transcript_dir"]),
                "article_path": str(article_path),
            },
        )
    )
    logger.info(format_log_event("prompt_ready", {"prompt_chars": len(prompt), "project_root": str(project_root)}))
    discovery_report_paths = {
        "research_plan_path": research_plan_path,
        "video_enrichment_manifest_path": video_enrichment_manifest_path,
        "selection_manifest_path": selection_manifest_path,
        "evidence_manifest_path": evidence_manifest_path if resolved_mode == "wide" else None,
    }

    if resolved_mode == "wide":
        try:
            query_plan = await plan_supplemental_queries(
                question=question,
                selection_manifest_path=selection_manifest_path,
                query_plan_path=query_plan_path,
                run_id=run_id,
                options=options,
                logger=logger,
            )
            await asyncio.gather(
                run_supplemental_searches(
                    query_plan=query_plan,
                    search_dir=paths["search_dir"],
                    run_id=run_id,
                    progress_sink=progress_sink,
                    logger=logger,
                ),
                run_supplemental_web_searches(
                    query_plan=query_plan,
                    web_search_dir=paths["web_search_dir"],
                    run_id=run_id,
                    progress_sink=progress_sink,
                    logger=logger,
                ),
            )
            write_web_evidence_cards(
                web_search_dir=paths["web_search_dir"],
                web_evidence_path=web_evidence_path,
                run_id=run_id,
            )
            generate_transcript_fetch_plan_from_searches(
                plan_path=transcript_fetch_plan_path,
                search_dir=paths["search_dir"],
                run_id=run_id,
            )
        except Exception as exc:
            error_message = str(exc)
            logger.exception(format_log_event("query_planning_failed", {"error_message": error_message}))
            write_artifact_reports(
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                search_dir=paths["search_dir"],
                web_search_dir=paths["web_search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
                research_mode=resolved_mode,
                **discovery_report_paths,
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
                error_message=error_message,
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                **discovery_report_paths,
            )
            _emit_progress(progress_sink, "task_failed", "failed", "任务失败", data={"error_message": error_message})
            raise AgentRunError(error_message) from exc
    else:
        stop_event = asyncio.Event()
        artifact_watcher = asyncio.create_task(
            _watch_artifact_progress(
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                progress_sink=progress_sink,
                stop_event=stop_event,
                start_article_on_transcript=True,
            )
        )
        sdk_error_message: str | None = None
        try:
            sdk_error_message = await _consume_sdk_query(prompt, options, logger, "sdk_message")
        except Exception as exc:
            error_message = sdk_error_message or str(exc)
            logger.exception(format_log_event("agent_failed", {"error_message": error_message}))
            write_artifact_reports(
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                search_dir=paths["search_dir"],
                web_search_dir=paths["web_search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
                research_mode=resolved_mode,
                **discovery_report_paths,
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
                error_message=error_message,
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                **discovery_report_paths,
            )
            _emit_progress(progress_sink, "task_failed", "failed", "任务失败", data={"error_message": error_message})
            raise AgentRunError(error_message) from exc
        finally:
            stop_event.set()
            await artifact_watcher

    if resolved_mode == "wide":
        try:
            await fetch_wide_transcripts_from_plan(
                plan_path=transcript_fetch_plan_path,
                selection_manifest_path=selection_manifest_path,
                transcript_dir=paths["transcript_dir"],
                manifest_path=transcript_fetch_manifest_path,
                run_id=run_id,
                progress_sink=progress_sink,
                logger=logger,
            )
            write_sources_manifest(
                sources_manifest_path,
                search_dir=paths["search_dir"],
                web_search_dir=paths["web_search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
            )
            extract_evidence_manifest = await extract_evidence_cards(
                question=question,
                transcript_dir=paths["transcript_dir"],
                evidence_dir=paths["evidence_dir"],
                evidence_manifest_path=evidence_manifest_path,
                run_id=run_id,
                options=evidence_options,
                logger=logger,
                progress_sink=progress_sink,
            )
            _emit_progress(
                progress_sink,
                "phase_started",
                "article_write",
                "正在撰写深度文章",
                data={
                    "article_path": str(article_path),
                    "evidence_manifest_path": str(evidence_manifest_path),
                    "evidence_success_count": extract_evidence_manifest.get("success_count"),
                },
            )
            final_prompt = build_wide_article_prompt(
                question=question,
                evidence_manifest_path=evidence_manifest_path,
                web_evidence_path=web_evidence_path if web_evidence_path.exists() else None,
                article_path=article_path,
                sources_manifest_path=sources_manifest_path,
            )
            logger.info(format_log_event("wide_article_prompt_ready", {"prompt_chars": len(final_prompt)}))
            await _consume_sdk_query(final_prompt, options, logger, "sdk_message")
        except Exception as exc:
            error_message = str(exc)
            logger.exception(format_log_event("evidence_or_wide_article_failed", {"error_message": error_message}))
            write_artifact_reports(
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                search_dir=paths["search_dir"],
                web_search_dir=paths["web_search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
                research_mode=resolved_mode,
                **discovery_report_paths,
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
                error_message=error_message,
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                **discovery_report_paths,
            )
            _emit_progress(progress_sink, "task_failed", "failed", "任务失败", data={"error_message": error_message})
            raise AgentRunError(error_message) from exc

    transcript_path = find_transcript_context(paths["transcript_dir"])
    if transcript_path is not None and (not article_path.exists() or article_path.stat().st_size == 0):
        if resolved_mode == "wide" and evidence_manifest_path.exists():
            retry_prompt = build_wide_article_prompt(
                question=question,
                evidence_manifest_path=evidence_manifest_path,
                web_evidence_path=web_evidence_path if web_evidence_path.exists() else None,
                article_path=article_path,
                sources_manifest_path=sources_manifest_path,
            )
        else:
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
        retry_sdk_error_message: str | None = None
        try:
            retry_sdk_error_message = await _consume_sdk_query(retry_prompt, options, logger, "sdk_message")
        except Exception as exc:
            error_message = retry_sdk_error_message or str(exc)
            logger.exception(format_log_event("article_retry_failed", {"error_message": error_message}))
            write_artifact_reports(
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                search_dir=paths["search_dir"],
                web_search_dir=paths["web_search_dir"],
                transcript_dir=paths["transcript_dir"],
                article_path=article_path,
                run_id=run_id,
                research_mode=resolved_mode,
                **discovery_report_paths,
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
                error_message=error_message,
                sources_manifest_path=sources_manifest_path,
                article_manifest_path=article_manifest_path,
                quality_report_path=quality_report_path,
                **discovery_report_paths,
            )
            raise AgentRunError(error_message) from exc

    if not article_path.exists() or article_path.stat().st_size == 0:
        write_artifact_reports(
            sources_manifest_path=sources_manifest_path,
            article_manifest_path=article_manifest_path,
            quality_report_path=quality_report_path,
            search_dir=paths["search_dir"],
            web_search_dir=paths["web_search_dir"],
            transcript_dir=paths["transcript_dir"],
            article_path=article_path,
            run_id=run_id,
            research_mode=resolved_mode,
            **discovery_report_paths,
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
            **discovery_report_paths,
        )
        raise RuntimeError("agent stopped before writing article.md")

    pre_normalize_sources_manifest = write_sources_manifest(
        sources_manifest_path,
        search_dir=paths["search_dir"],
        web_search_dir=paths["web_search_dir"],
        transcript_dir=paths["transcript_dir"],
        article_path=article_path,
        run_id=run_id,
    )
    normalized_timestamp_link_count = normalize_youtube_timestamp_link_text(
        article_path,
        sources_manifest=pre_normalize_sources_manifest,
    )
    if normalized_timestamp_link_count:
        logger.info(
            format_log_event(
                "timestamp_links_normalized",
                {
                    "article_path": str(article_path),
                    "normalized_count": normalized_timestamp_link_count,
                },
            )
        )

    sources_manifest, article_manifest, quality_report = write_artifact_reports(
        sources_manifest_path=sources_manifest_path,
        article_manifest_path=article_manifest_path,
        quality_report_path=quality_report_path,
        search_dir=paths["search_dir"],
        web_search_dir=paths["web_search_dir"],
        transcript_dir=paths["transcript_dir"],
        article_path=article_path,
        run_id=run_id,
        research_mode=resolved_mode,
        **discovery_report_paths,
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
        **discovery_report_paths,
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
            "normalized_timestamp_link_count": normalized_timestamp_link_count,
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
