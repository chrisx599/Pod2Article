from __future__ import annotations

import http.client
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from api.progress_log import ProgressLog
from api.video_deep_research_api import TaskStore, create_server


def _request(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(raw) if raw else {}


def _wait_for_status(port: int, task_id: str, status: str) -> dict:
    deadline = time.time() + 5
    while time.time() < deadline:
        code, data = _request(port, "GET", f"/video-deep-research/api/tasks/{task_id}/status")
        if code == 200 and data.get("status") == status:
            return data
        time.sleep(0.05)
    raise AssertionError(f"Task {task_id} did not reach {status}")


class ProgressLogTests(unittest.TestCase):
    def test_append_assigns_monotonic_seq(self) -> None:
        with TemporaryDirectory() as tmpdir:
            progress = ProgressLog(Path(tmpdir) / "progress.jsonl", threading.Lock())

            first = progress.append("phase_started", "prepare", "任务准备中")
            second = progress.append("phase_started", "source_fetch", "正在获取视频转录上下文")

            self.assertEqual(first["seq"], 1)
            self.assertEqual(second["seq"], 2)
            self.assertEqual(progress.read(after_seq=1)[0]["phase"], "source_fetch")


class VideoDeepResearchApiTests(unittest.TestCase):
    def test_default_runner_invokes_agent(self) -> None:
        with TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "task"
            task_dir.mkdir()
            with mock.patch.object(
                __import__("api.video_deep_research_api", fromlist=["run_agent"]),
                "run_agent",
            ) as run_agent:
                from api.video_deep_research_api import _default_runner

                _default_runner("hmtuvNfytjM", "写文章", task_dir)

        run_agent.assert_called_once()
        kwargs = run_agent.call_args.kwargs
        self.assertEqual(kwargs["input_value"], "hmtuvNfytjM")
        self.assertEqual(kwargs["question"], "写文章")
        self.assertEqual(kwargs["output_dir"], str(task_dir))
        self.assertEqual(kwargs["log_path"], task_dir / "agent.log")

    def test_task_lifecycle_returns_status_article_and_delete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            def fake_runner(input_value: str, question: str, task_dir: Path) -> None:
                article_dir = task_dir / "hmtuvNfytjM" / "articles" / "article-20260510T000000Z"
                article_dir.mkdir(parents=True)
                (article_dir / "article.md").write_text(
                    f"# Demo\n\ninput={input_value}\nquestion={question}\n",
                    encoding="utf-8",
                )

            store = TaskStore(Path(tmpdir), runner=fake_runner)
            server = create_server(("127.0.0.1", 0), store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                code, created = _request(
                    port,
                    "POST",
                    "/video-deep-research/api/tasks",
                    {"input": "hmtuvNfytjM", "question": "写文章"},
                )

                self.assertEqual(code, 202)
                self.assertEqual(created["status"], "queued")
                task_id = created["task_id"]

                completed = _wait_for_status(port, task_id, "completed")
                self.assertTrue(completed["article_available"])

                code, article = _request(port, "GET", f"/video-deep-research/api/tasks/{task_id}/article")
                self.assertEqual(code, 200)
                self.assertIn("# Demo", article["article_markdown"])
                self.assertIn("hmtuvNfytjM", article["article_markdown"])

                code, progress = _request(port, "GET", f"/video-deep-research/api/tasks/{task_id}/progress")
                self.assertEqual(code, 200)
                self.assertTrue(progress["events"])

                code, deleted = _request(port, "DELETE", f"/video-deep-research/api/tasks/{task_id}")
                self.assertEqual(code, 200)
                self.assertEqual(deleted["status"], "deleted")
            finally:
                server.shutdown()
                server.server_close()

    def test_create_task_accepts_url_alias_and_question_only_wide_search(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = TaskStore(Path(tmpdir), runner=lambda _input, _question, _task_dir: None)
            server = create_server(("127.0.0.1", 0), store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                code, data = _request(
                    server.server_address[1],
                    "POST",
                    "/video-deep-research/api/tasks",
                    {"question": "写文章"},
                )
                self.assertEqual(code, 202)
                self.assertEqual(data["research_mode"], "wide")

                code, created = _request(
                    server.server_address[1],
                    "POST",
                    "/video-deep-research/api/tasks",
                    {"url": "https://www.youtube.com/watch?v=hmtuvNfytjM"},
                )
                self.assertEqual(code, 202)
                self.assertEqual(created["status"], "queued")
            finally:
                server.shutdown()
                server.server_close()

    def test_sync_task_returns_completed_article(self) -> None:
        with TemporaryDirectory() as tmpdir:
            def fake_runner(input_value: str, question: str, task_dir: Path) -> None:
                article_dir = task_dir / "demo" / "articles"
                article_dir.mkdir(parents=True)
                (article_dir / "article.md").write_text(
                    f"# Demo\n\ninput={input_value}\nquestion={question}\n",
                    encoding="utf-8",
                )

            store = TaskStore(Path(tmpdir), runner=fake_runner)
            server = create_server(("127.0.0.1", 0), store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                code, data = _request(
                    server.server_address[1],
                    "POST",
                    "/video-deep-research/api/tasks/sync",
                    {"input": "hmtuvNfytjM", "question": "写文章"},
                )

                self.assertEqual(code, 200)
                self.assertEqual(data["status"], "completed")
                self.assertIn("# Demo", data["article_markdown"])
                self.assertTrue(data["article_path"].endswith("article.md"))
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
