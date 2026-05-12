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
    research_mode: str = "deep",
) -> str:
    fetch_command = (
        "python3 podcast-to-article/scripts/fetch_transcript.py "
        f"{shlex.quote(input_value)} --output-dir {shlex.quote(transcript_dir)}"
    )
    search_command = (
        "python3 podcast-to-article/scripts/search_youtube.py "
        f"{shlex.quote(input_value)} --output-dir {shlex.quote(search_dir)}"
    )
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

Write the final Markdown article only to this exact path:
{article_path}

Required wide-search workflow:
1. Run the bundled YouTube search tool from the repository root:
   {search_command}
2. Open the generated `.search.json`, inspect the ranked candidates, and choose 3-5 relevant videos when available. Prefer substantive interviews, talks, or podcast episodes over short clips.
3. For each selected video, run the bundled transcript fetcher with the video URL:
   python3 podcast-to-article/scripts/fetch_transcript.py "<selected-video-url>" --output-dir {shlex.quote(transcript_dir)}
4. Open and read every generated `.transcript.json` file before drafting.
5. Synthesize across the gathered transcripts. Compare recurring claims, changes over time, disagreements, and caveats when the source material supports them.
6. Write a coherent Markdown article that answers the research topic and includes clickable YouTube timestamp links.
7. Do not create article drafts in any other directory. Do not expose hidden reasoning.

Required outputs:
- one `.search.json` file under {search_dir}
- one or more `.transcript.json` files under {transcript_dir}
- {article_path}

If only one relevant transcript can be acquired, write the article from that transcript and state the coverage limitation in the article.

At the end, print:
search: <path to generated search json>
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
    except Exception:
        logger.exception(format_log_event("agent_failed"))
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
        async for message in query(prompt=retry_prompt, options=options):
            logger.info(format_log_event("sdk_message", serialize_message(message)))
            for text in _iter_text_blocks(message):
                logger.info(format_log_text_block("message_text", text))
                print(text)

    if not article_path.exists() or article_path.stat().st_size == 0:
        raise RuntimeError("agent stopped before writing article.md")

    _emit_progress(
        progress_sink,
        "phase_progress",
        "article_write",
        "已写入深度文章",
        data={"article_path": str(article_path)},
    )
    logger.info(format_log_event("agent_completed", {"article_path": str(article_path)}))
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
