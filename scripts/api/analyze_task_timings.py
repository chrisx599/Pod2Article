#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z \|")


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    value = (end - start).total_seconds()
    return value if value >= 0 else None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = round(seconds, 1)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if minutes:
        return f"{minutes}m{rest:04.1f}s"
    return f"{rest:.1f}s"


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_progress(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def read_agent_log_times(path: Path) -> list[datetime]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    times: list[datetime] = []
    for line in lines:
        match = LOG_TS_RE.match(line)
        if not match:
            continue
        parsed = parse_time(match.group(1) + "Z")
        if parsed is not None:
            times.append(parsed)
    return times


def first_event_time(
    events: list[dict[str, Any]],
    *,
    phase: str | None = None,
    event_type: str | None = None,
    message_contains: str | None = None,
    message_equals: str | None = None,
) -> datetime | None:
    for event in events:
        if phase is not None and event.get("phase") != phase:
            continue
        if event_type is not None and event.get("type") != event_type:
            continue
        message = str(event.get("message") or "")
        if message_contains is not None and message_contains not in message:
            continue
        if message_equals is not None and message != message_equals:
            continue
        parsed = parse_time(event.get("ts"))
        if parsed is not None:
            return parsed
    return None


def find_workspace_dir(task_dir: Path) -> Path | None:
    manifests = sorted(task_dir.glob("*/run-manifest.json"))
    return manifests[0].parent if manifests else None


@dataclass
class TimingSummary:
    task_id: str
    status: str
    research_mode: str
    input_text: str
    started_at: datetime | None
    total_seconds: float | None
    observed_seconds: float | None
    api_to_agent_seconds: float | None
    pre_agent_discovery_seconds: float | None
    source_to_first_transcript_seconds: float | None
    agent_to_first_transcript_seconds: float | None
    article_write_seconds: float | None
    finalize_seconds: float | None
    agent_window_seconds: float | None
    first_sdk_delay_seconds: float | None
    search_count: int | None
    search_candidate_count: int | None
    selection_candidate_count: int | None
    transcript_count: int | None
    quality_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "research_mode": self.research_mode,
            "input": self.input_text,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "total_seconds": self.total_seconds,
            "observed_seconds": self.observed_seconds,
            "api_to_agent_seconds": self.api_to_agent_seconds,
            "pre_agent_discovery_seconds": self.pre_agent_discovery_seconds,
            "source_to_first_transcript_seconds": self.source_to_first_transcript_seconds,
            "agent_to_first_transcript_seconds": self.agent_to_first_transcript_seconds,
            "article_write_seconds": self.article_write_seconds,
            "finalize_seconds": self.finalize_seconds,
            "agent_window_seconds": self.agent_window_seconds,
            "first_sdk_delay_seconds": self.first_sdk_delay_seconds,
            "search_count": self.search_count,
            "search_candidate_count": self.search_candidate_count,
            "selection_candidate_count": self.selection_candidate_count,
            "transcript_count": self.transcript_count,
            "quality_status": self.quality_status,
        }


def summarize_task(task_dir: Path) -> TimingSummary | None:
    progress = read_progress(task_dir / "progress.jsonl")
    log_times = read_agent_log_times(task_dir / "agent.log")
    status_payload = read_json(task_dir / "status.json")
    workspace_dir = find_workspace_dir(task_dir)

    run_manifest = read_json(workspace_dir / "run-manifest.json") if workspace_dir else {}
    search_manifest = read_json(workspace_dir / "search-results" / "search-manifest.json") if workspace_dir else {}
    selection_manifest = read_json(workspace_dir / "selection-manifest.json") if workspace_dir else {}
    quality_report = read_json(workspace_dir / "quality-report.json") if workspace_dir else {}
    sources_manifest = read_json(workspace_dir / "sources-manifest.json") if workspace_dir else {}

    if not progress and not log_times and not run_manifest:
        return None

    start = parse_time(progress[0].get("ts")) if progress else parse_time(run_manifest.get("created_at"))
    completed = first_event_time(progress, event_type="task_completed")
    last_progress = parse_time(progress[-1].get("ts")) if progress else None

    source_start = first_event_time(progress, phase="source_fetch", event_type="phase_started")
    discovery_start = first_event_time(progress, phase="source_fetch", message_contains="search")
    if discovery_start is None:
        discovery_start = first_event_time(progress, phase="source_fetch", message_contains="搜索")
    first_transcript = first_event_time(progress, message_equals="已获取转录上下文")
    article_start = first_event_time(progress, phase="article_write", event_type="phase_started")
    article_written = first_event_time(
        progress,
        phase="article_write",
        event_type="phase_progress",
        message_equals="已写入深度文章",
    )

    agent_start = log_times[0] if log_times else None
    agent_last = log_times[-1] if log_times else None
    # First two log records are normally AGENT START and PROMPT READY.
    first_sdk = log_times[2] if len(log_times) > 2 else None

    searches = search_manifest.get("searches")
    search_candidate_count = None
    if isinstance(searches, list):
        search_candidate_count = sum(
            int(item.get("candidate_count") or 0)
            for item in searches
            if isinstance(item, dict)
        )

    return TimingSummary(
        task_id=task_dir.name,
        status=str(status_payload.get("status") or run_manifest.get("status") or "unknown"),
        research_mode=str(run_manifest.get("research_mode") or status_payload.get("research_mode") or "unknown"),
        input_text=str(run_manifest.get("input") or status_payload.get("input") or ""),
        started_at=start,
        total_seconds=seconds_between(start, completed),
        observed_seconds=seconds_between(start, last_progress),
        api_to_agent_seconds=seconds_between(start, agent_start),
        pre_agent_discovery_seconds=seconds_between(discovery_start, agent_start),
        source_to_first_transcript_seconds=seconds_between(source_start, first_transcript),
        agent_to_first_transcript_seconds=seconds_between(agent_start, first_transcript),
        article_write_seconds=seconds_between(article_start, article_written),
        finalize_seconds=seconds_between(article_written, completed),
        agent_window_seconds=seconds_between(agent_start, agent_last),
        first_sdk_delay_seconds=seconds_between(agent_start, first_sdk),
        search_count=as_int(search_manifest.get("search_count") or sources_manifest.get("search_count")),
        search_candidate_count=search_candidate_count,
        selection_candidate_count=as_int(selection_manifest.get("candidate_count")),
        transcript_count=as_int(quality_report.get("transcript_count") or sources_manifest.get("transcript_count")),
        quality_status=str(quality_report.get("status") or "-"),
    )


def as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def iter_summaries(output_dir: Path) -> list[TimingSummary]:
    if not output_dir.exists():
        return []
    summaries: list[TimingSummary] = []
    for task_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        summary = summarize_task(task_dir)
        if summary is not None:
            summaries.append(summary)
    return summaries


def render_markdown(summaries: list[TimingSummary]) -> str:
    headers = [
        "task",
        "status",
        "total",
        "pre-agent discovery",
        "agent window",
        "first transcript",
        "article write",
        "finalize",
        "searches",
        "candidates",
        "transcripts",
        "input",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in summaries:
        first_transcript = item.agent_to_first_transcript_seconds
        rows.append(
            "| "
            + " | ".join(
                [
                    item.task_id,
                    item.status,
                    format_duration(item.total_seconds),
                    format_duration(item.pre_agent_discovery_seconds),
                    format_duration(item.agent_window_seconds),
                    format_duration(first_transcript),
                    format_duration(item.article_write_seconds),
                    format_duration(item.finalize_seconds),
                    str(item.search_count if item.search_count is not None else "-"),
                    str(item.search_candidate_count if item.search_candidate_count is not None else "-"),
                    str(item.transcript_count if item.transcript_count is not None else "-"),
                    item.input_text.replace("|", "\\|")[:48],
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Video Deep Research task timings.")
    parser.add_argument("--output-dir", default="output/api", help="Directory containing API task outputs.")
    parser.add_argument("--limit", type=int, default=8, help="Show the most recent N tasks.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    summaries = iter_summaries(Path(args.output_dir))
    if args.limit > 0:
        summaries = summaries[-args.limit :]

    if args.format == "json":
        print(json.dumps([item.as_dict() for item in summaries], ensure_ascii=False, indent=2))
    else:
        print(render_markdown(summaries))
        print()
        print("Notes:")
        print("- total is measured from the first progress event to task_completed.")
        print("- pre-agent discovery is the best available search/enrichment estimate before SDK agent start.")
        print("- agent window is measured from first to last timestamped agent.log record.")
        print("- first transcript, article write, and finalize are derived from progress phase boundaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
