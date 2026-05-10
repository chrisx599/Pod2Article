from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
from pathlib import Path
import shutil
import threading
import traceback
from typing import Callable
from urllib.parse import parse_qs, urlparse
import uuid

from agents.podcast_article_agent import run_agent
from api.progress_log import PROGRESS_PHASE_MESSAGES, ProgressLog


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_TASK_ROOT = Path("output") / "api"
API_PREFIX = "/video-deep-research/api/tasks"
DEFAULT_QUESTION = "请基于这个视频生成一篇结构化深度研究文章。"
ProgressSink = Callable[[dict[str, object]], None]
TaskRunner = Callable[[str, str, Path, ProgressSink | None], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _default_runner(
    input_value: str,
    question: str,
    task_dir: Path,
    progress_sink: ProgressSink | None = None,
) -> None:
    asyncio.run(
        run_agent(
            input_value=input_value,
            question=question,
            output_dir=str(task_dir),
            log_path=task_dir / "agent.log",
            progress_sink=progress_sink,
        )
    )


@dataclass
class TaskRecord:
    task_id: str
    input_value: str
    question: str
    task_dir: Path
    research_mode: str = "auto"
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    error_message: str = ""
    article_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "research_mode": self.research_mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "article_available": self.article_path is not None and self.article_path.exists(),
            "article_path": str(self.article_path) if self.article_path else "",
            "error_message": self.error_message,
        }


class TaskStore:
    def __init__(self, task_root: Path = DEFAULT_TASK_ROOT, runner: TaskRunner | None = None) -> None:
        self.task_root = Path(task_root)
        self.runner = runner or _default_runner
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()
        self.task_root.mkdir(parents=True, exist_ok=True)

    def create_task(self, *, input_value: str, question: str) -> TaskRecord:
        record = self._create_queued_record(input_value=input_value, question=question)
        response_record = TaskRecord(
            task_id=record.task_id,
            input_value=record.input_value,
            question=record.question,
            task_dir=record.task_dir,
            research_mode=record.research_mode,
        )
        self._save(record)
        self.append_progress(record.task_id, "phase_started", "prepare")

        thread = threading.Thread(target=self._run_task, args=(record.task_id,), daemon=True)
        thread.start()
        return response_record

    def run_task_sync(self, *, input_value: str, question: str) -> TaskRecord:
        record = self._create_queued_record(input_value=input_value, question=question)
        self._save(record)
        self.append_progress(record.task_id, "phase_started", "prepare")
        self._run_task(record.task_id)
        synced_record = self.get_task(record.task_id)
        if synced_record is None:
            raise RuntimeError(f"Task {record.task_id} disappeared during synchronous execution")
        return synced_record

    def _create_queued_record(self, *, input_value: str, question: str) -> TaskRecord:
        task_id = _new_task_id()
        task_dir = self.task_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        research_mode = "wide" if input_value.strip() == question.strip() else "auto"
        return TaskRecord(
            task_id=task_id,
            input_value=input_value,
            question=question,
            task_dir=task_dir,
            research_mode=research_mode,
        )

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            record = self._tasks.pop(task_id, None)
        if record is None:
            return False
        shutil.rmtree(record.task_dir, ignore_errors=True)
        return True

    def read_article(self, task_id: str) -> str | None:
        record = self.get_task(task_id)
        if record is None or record.article_path is None or not record.article_path.exists():
            return None
        return record.article_path.read_text(encoding="utf-8")

    def append_progress(
        self,
        task_id: str,
        event_type: str,
        phase: str,
        message: str | None = None,
        *,
        data: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        record = self.get_task(task_id)
        if record is None:
            return None
        resolved_message = message or PROGRESS_PHASE_MESSAGES.get(phase, phase)
        return ProgressLog(record.task_dir / "progress.jsonl", self._lock).append(
            event_type,
            phase,
            resolved_message,
            data=data,
        )

    def read_progress(
        self,
        task_id: str,
        *,
        after_seq: int = 0,
        limit: int = 100,
    ) -> list[dict[str, object]] | None:
        record = self.get_task(task_id)
        if record is None:
            return None
        return ProgressLog(record.task_dir / "progress.jsonl", self._lock).read(
            after_seq=after_seq,
            limit=limit,
        )

    def _run_task(self, task_id: str) -> None:
        record = self.get_task(task_id)
        if record is None:
            return

        self._update(task_id, status="running")
        try:
            self._call_runner(record)
            article_path = self._find_latest_article(record.task_dir)
            if article_path is None:
                message = "article.md was not generated"
                self._update(task_id, status="failed", error_message=message)
                self.append_progress(task_id, "task_failed", "failed", data={"error_message": message})
            else:
                self._update(task_id, status="completed", article_path=article_path)
                self.append_progress(task_id, "task_completed", "completed")
        except Exception as exc:  # pragma: no cover - detail is stored for API consumers
            error_path = record.task_dir / "error.log"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            self._update(task_id, status="failed", error_message=str(exc))
            self.append_progress(task_id, "task_failed", "failed", data={"error_message": str(exc)})

    def _call_runner(self, record: TaskRecord) -> None:
        def progress_sink(event: dict[str, object]) -> None:
            data = event.get("data")
            self.append_progress(
                record.task_id,
                str(event.get("type", "phase_progress")),
                str(event.get("phase", "prepare")),
                str(event.get("message", "")),
                data=data if isinstance(data, dict) else {},
            )

        if self._runner_accepts_progress_sink():
            self.runner(record.input_value, record.question, record.task_dir, progress_sink)
        else:
            self.runner(record.input_value, record.question, record.task_dir)  # type: ignore[misc, call-arg]

    def _runner_accepts_progress_sink(self) -> bool:
        try:
            signature = inspect.signature(self.runner)
        except (TypeError, ValueError):
            return True

        positional_count = 0
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                return True
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                positional_count += 1
        return positional_count >= 4

    def _save(self, record: TaskRecord) -> None:
        with self._lock:
            self._tasks[record.task_id] = record
        self._write_status(record)

    def _update(
        self,
        task_id: str,
        *,
        status: str,
        error_message: str = "",
        article_path: Path | None = None,
    ) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.status = status
            record.updated_at = _utc_now()
            record.error_message = error_message
            if article_path is not None:
                record.article_path = article_path
        self._write_status(record)

    def _write_status(self, record: TaskRecord) -> None:
        status_path = record.task_dir / "status.json"
        status_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _find_latest_article(self, task_dir: Path) -> Path | None:
        articles = sorted(task_dir.rglob("article.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        return articles[0] if articles else None


class VideoDeepResearchRequestHandler(BaseHTTPRequestHandler):
    store: TaskStore

    def do_POST(self) -> None:
        if self.path == f"{API_PREFIX}/sync":
            self._handle_sync_post()
            return

        if self.path != API_PREFIX:
            self._send_json(404, {"error": "not_found"})
            return

        payload = self._read_json()
        question = str(payload.get("question", "") or DEFAULT_QUESTION).strip()
        input_value = _request_input_value(payload, question=question)
        if not question:
            self._send_json(400, {"error": "invalid_request", "message": "question must not be empty"})
            return

        record = self.store.create_task(input_value=input_value, question=question)
        self._send_json(202, record.to_dict())

    def _handle_sync_post(self) -> None:
        payload = self._read_json()
        question = str(payload.get("question", "") or DEFAULT_QUESTION).strip()
        input_value = _request_input_value(payload, question=question)
        if not question:
            self._send_json(400, {"error": "invalid_request", "message": "question must not be empty"})
            return

        record = self.store.run_task_sync(input_value=input_value, question=question)
        article = self.store.read_article(record.task_id)
        if record.status != "completed" or article is None:
            self._send_json(500, record.to_dict())
            return

        response = {
            **record.to_dict(),
            "article_markdown": article,
            "article_path": str(record.article_path),
        }
        self._send_json(200, response)

    def do_GET(self) -> None:
        task_id, action = self._parse_task_path()
        if not task_id:
            self._send_json(404, {"error": "not_found"})
            return

        if action == "status":
            record = self.store.get_task(task_id)
            if record is None:
                self._send_json(404, {"error": "task_not_found"})
                return
            self._send_json(200, record.to_dict())
            return

        if action == "article":
            record = self.store.get_task(task_id)
            if record is None:
                self._send_json(404, {"error": "task_not_found"})
                return
            article = self.store.read_article(task_id)
            if article is None:
                self._send_json(404, {"error": "article_not_found", "status": record.status})
                return
            self._send_json(
                200,
                {
                    "task_id": task_id,
                    "status": record.status,
                    "article_markdown": article,
                    "article_path": str(record.article_path),
                },
            )
            return

        if action == "progress":
            query = self._read_progress_query()
            if query is None:
                self._send_json(
                    400,
                    {"error": "invalid_request", "message": "after_seq must be a non-negative integer"},
                )
                return
            after_seq, limit = query
            record = self.store.get_task(task_id)
            if record is None:
                self._send_json(404, {"error": "task_not_found"})
                return
            events = self.store.read_progress(task_id, after_seq=after_seq, limit=limit)
            if events is None:
                self._send_json(404, {"error": "task_not_found"})
                return
            next_after_seq = int(events[-1]["seq"]) if events else after_seq
            more_events = self.store.read_progress(task_id, after_seq=next_after_seq, limit=1)
            self._send_json(
                200,
                {
                    "task_id": task_id,
                    "status": record.status,
                    "next_after_seq": next_after_seq,
                    "has_more": bool(more_events),
                    "events": events,
                },
            )
            return

        self._send_json(404, {"error": "not_found"})

    def do_DELETE(self) -> None:
        task_id, action = self._parse_task_path()
        if not task_id or action:
            self._send_json(404, {"error": "not_found"})
            return

        if not self.store.delete_task(task_id):
            self._send_json(404, {"error": "task_not_found"})
            return
        self._send_json(200, {"task_id": task_id, "status": "deleted"})

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _parse_task_path(self) -> tuple[str, str]:
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 4 and parts[:3] == ["video-deep-research", "api", "tasks"]:
            return parts[3], ""
        if len(parts) == 5 and parts[:3] == ["video-deep-research", "api", "tasks"]:
            return parts[3], parts[4]
        return "", ""

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _read_progress_query(self) -> tuple[int, int] | None:
        query = parse_qs(urlparse(self.path).query)
        try:
            after_seq = int(query.get("after_seq", ["0"])[0])
            limit = int(query.get("limit", ["100"])[0])
        except ValueError:
            return None
        if after_seq < 0 or limit < 1:
            return None
        return after_seq, min(limit, 500)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _request_input_value(payload: dict[str, object], *, question: str) -> str:
    return str(
        payload.get("input", "")
        or payload.get("url", "")
        or payload.get("query", "")
        or payload.get("topic", "")
        or question
    ).strip()


def create_server(address: tuple[str, int], store: TaskStore) -> ThreadingHTTPServer:
    class Handler(VideoDeepResearchRequestHandler):
        pass

    Handler.store = store
    return ThreadingHTTPServer(address, Handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Video Deep Research HTTP API.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = TaskStore(Path(args.task_root))
    server = create_server((args.host, args.port), store)
    print(f"Video Deep Research API listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
