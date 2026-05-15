from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "api" / "analyze_task_timings.py"
SPEC = importlib.util.spec_from_file_location("analyze_task_timings", SCRIPT_PATH)
assert SPEC is not None
analyze_task_timings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["analyze_task_timings"] = analyze_task_timings
SPEC.loader.exec_module(analyze_task_timings)


class AnalyzeTaskTimingsTests(unittest.TestCase):
    def write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_summarize_task_combines_progress_log_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "20260515T000000Z-test"
            workspace = task_dir / "source"
            progress = [
                {"ts": "2026-05-15T00:00:00+00:00", "type": "phase_started", "phase": "prepare", "message": "prepare"},
                {
                    "ts": "2026-05-15T00:00:01+00:00",
                    "type": "phase_started",
                    "phase": "source_fetch",
                    "message": "source",
                },
                {
                    "ts": "2026-05-15T00:00:02+00:00",
                    "type": "phase_progress",
                    "phase": "source_fetch",
                    "message": "search",
                },
                {
                    "ts": "2026-05-15T00:02:30+00:00",
                    "type": "phase_progress",
                    "phase": "source_fetch",
                    "message": "已获取转录上下文",
                },
                {
                    "ts": "2026-05-15T00:02:31+00:00",
                    "type": "phase_started",
                    "phase": "article_write",
                    "message": "write",
                },
                {
                    "ts": "2026-05-15T00:05:00+00:00",
                    "type": "phase_progress",
                    "phase": "article_write",
                    "message": "已写入深度文章",
                },
                {"ts": "2026-05-15T00:05:10+00:00", "type": "task_completed", "phase": "completed", "message": "done"},
            ]
            task_dir.mkdir(parents=True)
            (task_dir / "progress.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in progress),
                encoding="utf-8",
            )
            (task_dir / "agent.log").write_text(
                "\n".join(
                    [
                        "2026-05-15T00:00:10Z | INFO     | AGENT START",
                        "2026-05-15T00:00:11Z | INFO     | PROMPT READY",
                        "2026-05-15T00:00:13Z | INFO     | SDK MESSAGE",
                        "2026-05-15T00:05:08Z | INFO     | SDK MESSAGE",
                    ]
                ),
                encoding="utf-8",
            )
            self.write_json(
                workspace / "run-manifest.json",
                {"status": "completed", "research_mode": "wide", "input": "sample"},
            )
            self.write_json(
                workspace / "search-results" / "search-manifest.json",
                {"search_count": 2, "searches": [{"candidate_count": 10}, {"candidate_count": 12}]},
            )
            self.write_json(workspace / "selection-manifest.json", {"candidate_count": 15})
            self.write_json(workspace / "quality-report.json", {"status": "passed", "transcript_count": 3})

            summary = analyze_task_timings.summarize_task(task_dir)

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary.total_seconds, 310)
            self.assertEqual(summary.pre_agent_discovery_seconds, 8)
            self.assertEqual(summary.agent_to_first_transcript_seconds, 140)
            self.assertEqual(summary.article_write_seconds, 149)
            self.assertEqual(summary.finalize_seconds, 10)
            self.assertEqual(summary.agent_window_seconds, 298)
            self.assertEqual(summary.first_sdk_delay_seconds, 3)
            self.assertEqual(summary.search_count, 2)
            self.assertEqual(summary.search_candidate_count, 22)
            self.assertEqual(summary.selection_candidate_count, 15)
            self.assertEqual(summary.transcript_count, 3)

    def test_negative_durations_are_hidden(self) -> None:
        start = datetime(2026, 5, 15, 0, 1, tzinfo=timezone.utc)
        end = datetime(2026, 5, 15, 0, 0, tzinfo=timezone.utc)

        self.assertIsNone(analyze_task_timings.seconds_between(start, end))


if __name__ == "__main__":
    unittest.main()
