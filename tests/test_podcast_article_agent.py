from __future__ import annotations

import asyncio
import json
import os
import shlex
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agents.podcast_article_agent import (
    DEFAULT_OUTPUT_DIR,
    build_agent_options,
    build_article_dir,
    build_prompt,
    build_run_metadata,
    build_source_id,
    build_workspace_paths,
    default_log_path,
    extract_youtube_video_id,
    find_transcript_context,
    load_env_file,
    resolve_model,
    run_agent,
    serialize_message,
)


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict


@dataclass
class FakeMessage:
    content: list
    session_id: str = "session-1"


class PodcastArticleAgentTests(unittest.TestCase):
    def test_default_agent_outputs_live_under_output_agent(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT_DIR, "output/agent")
        self.assertEqual(default_log_path().parent, Path("output") / "agent" / "logs")

    def test_build_prompt_requests_skill_and_exact_outputs(self) -> None:
        prompt = build_prompt(
            input_value="https://www.youtube.com/watch?v=hmtuvNfytjM",
            question="请写一篇深度文章",
            workspace_dir="output/run/workspace",
            search_dir="output/run/search-results",
            transcript_dir="output/run/transcripts",
            article_path="output/run/articles/article.md",
        )

        self.assertIn("podcast-to-article", prompt)
        self.assertIn("https://www.youtube.com/watch?v=hmtuvNfytjM", prompt)
        self.assertIn("请写一篇深度文章", prompt)
        self.assertIn("podcast-to-article/scripts/fetch_transcript.py", prompt)
        self.assertIn("'https://www.youtube.com/watch?v=hmtuvNfytjM'", prompt)
        self.assertIn("--output-dir output/run/transcripts", prompt)
        self.assertIn("Write the final Markdown article only to this exact path", prompt)
        self.assertIn("output/run/articles/article.md", prompt)
        self.assertIn(".transcript.json", prompt)

    def test_wide_prompt_requires_derived_search_query(self) -> None:
        question = "中国的 AI 行业大佬对当前AI发展情况的判断是怎么样的，请帮我调研"
        prompt = build_prompt(
            input_value=question,
            question=question,
            workspace_dir="output/run/workspace",
            search_dir="output/run/search-results",
            transcript_dir="output/run/transcripts",
            article_path="output/run/articles/article.md",
            research_mode="wide",
        )

        self.assertIn("Derive 2-3 concise, complementary YouTube search queries", prompt)
        self.assertIn('search_youtube.py "<derived-search-query>"', prompt)
        self.assertIn("Enforce source diversity", prompt)
        self.assertIn("Do not count third-party media analysis", prompt)
        self.assertIn("Avoid broad English queries", prompt)
        self.assertIn("Do not let one long transcript dominate", prompt)
        self.assertIn("search_queries: <derived search queries>", prompt)
        self.assertNotIn(f"search_youtube.py {shlex.quote(question)}", prompt)

    def test_build_article_dir_adds_uuid_suffix(self) -> None:
        fixed_now = datetime(2026, 5, 10, 12, 30, tzinfo=timezone.utc)
        with patch("agents.podcast_article_agent.uuid.uuid4") as fake_uuid:
            with patch("agents.podcast_article_agent.datetime") as fake_datetime:
                fake_datetime.now.return_value = fixed_now
                fake_uuid.return_value.hex = "abcdef1234567890"
                article_dir = build_article_dir(Path("output/demo/articles"))

        self.assertEqual(article_dir, Path("output/demo/articles/article-20260510T123000Z-abcdef12"))

    def test_build_agent_options_loads_project_skill_and_tools(self) -> None:
        options = build_agent_options(Path("/tmp/pod2article"), model="claude-sonnet")

        self.assertEqual(options.cwd, "/tmp/pod2article")
        self.assertEqual(options.setting_sources, ["project"])
        self.assertEqual(options.model, "claude-sonnet")
        self.assertIn("Skill", options.allowed_tools)
        self.assertIn("Bash", options.allowed_tools)
        self.assertIn("Write", options.allowed_tools)

    def test_extract_youtube_video_id_and_source_id(self) -> None:
        self.assertEqual(
            extract_youtube_video_id("https://www.youtube.com/watch?v=hmtuvNfytjM&list=demo"),
            "hmtuvNfytjM",
        )
        self.assertEqual(build_source_id("hmtuvNfytjM"), "hmtuvNfytjM")
        self.assertEqual(build_source_id("Sam Altman GPT 5 interview"), "sam-altman-gpt-5-interview")

    def test_build_workspace_paths(self) -> None:
        paths = build_workspace_paths(Path("output/agent"), "hmtuvNfytjM")

        self.assertEqual(paths["workspace_dir"], Path("output/agent/hmtuvNfytjM"))
        self.assertEqual(paths["search_dir"], Path("output/agent/hmtuvNfytjM/search-results"))
        self.assertEqual(paths["transcript_dir"], Path("output/agent/hmtuvNfytjM/transcripts"))
        self.assertEqual(paths["articles_root"], Path("output/agent/hmtuvNfytjM/articles"))

    def test_load_env_file_and_resolve_model(self) -> None:
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("DEFAULT_MODEL=base\nCLAUDE_AGENT_MODEL=override\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                values = load_env_file(env_path)
                resolved = resolve_model(values)

        self.assertEqual(values["DEFAULT_MODEL"], "base")
        self.assertEqual(resolved, "override")

    def test_load_env_file_overrides_existing_environment(self) -> None:
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic\n"
                "ANTHROPIC_API_KEY=deepseek-key\n"
                "CLAUDE_AGENT_MODEL=deepseek-v4-pro\n",
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "ANTHROPIC_AUTH_TOKEN": "stale-claude-code-token",
                    "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                    "CLAUDE_AGENT_MODEL": "claude-sonnet",
                },
                clear=True,
            ):
                values = load_env_file(env_path)
                resolved = resolve_model(values)

                self.assertEqual(os.environ["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic")
                self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "deepseek-key")
                self.assertNotIn("ANTHROPIC_AUTH_TOKEN", os.environ)
                self.assertEqual(resolved, "deepseek-v4-pro")

    def test_build_run_metadata_masks_secrets(self) -> None:
        metadata = build_run_metadata(
            input_value="hmtuvNfytjM",
            question="写文章",
            output_dir="output/demo",
            model="claude-sonnet",
            env_values={"SERPAPI_API_KEY": "secret-value", "DEFAULT_MODEL": "claude-sonnet"},
        )

        self.assertEqual(metadata["env"]["SERPAPI_API_KEY"], "<set>")
        self.assertNotIn("secret-value", str(metadata))

    def test_serialize_message_masks_secret_tool_input(self) -> None:
        message = FakeMessage(
            content=[
                FakeToolUse(
                    id="toolu_1",
                    name="Bash",
                    input={"cmd": "echo hi", "SERPAPI_API_KEY": "secret-value"},
                )
            ]
        )

        payload = serialize_message(message)

        self.assertEqual(payload["type"], "FakeMessage")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["content"][0]["input"]["SERPAPI_API_KEY"], "<set>")
        self.assertNotIn("secret-value", str(payload))

    def test_find_transcript_context_returns_latest_non_empty_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            transcript_dir = Path(tmpdir)
            (transcript_dir / "empty.transcript.json").write_text("", encoding="utf-8")
            ready = transcript_dir / "ready.transcript.json"
            ready.write_text("{}", encoding="utf-8")

            self.assertEqual(find_transcript_context(transcript_dir), ready)

    def test_run_agent_writes_article_and_emits_progress(self) -> None:
        events: list[dict[str, object]] = []

        async def fake_query(prompt: str, options: object):
            transcript_dir = Path(prompt.split("Use this exact transcript output directory:")[1].splitlines()[1])
            article_path = Path(prompt.split("Write the final Markdown article only to this exact path:")[1].splitlines()[1])
            transcript_dir.mkdir(parents=True, exist_ok=True)
            (transcript_dir / "demo.transcript.json").write_text(
                json.dumps({"segments": [{"text": "hello"}]}),
                encoding="utf-8",
            )
            article_path.parent.mkdir(parents=True, exist_ok=True)
            article_path.write_text("# Demo\n\n[`00:00`](https://www.youtube.com/watch?v=hmtuvNfytjM&t=0s)\n", encoding="utf-8")
            await asyncio.sleep(0.05)
            yield object()

        with patch("agents.podcast_article_agent.query", fake_query):
            with patch("agents.podcast_article_agent.ARTIFACT_PROGRESS_POLL_SECONDS", 0.01):
                with TemporaryDirectory() as tmpdir:
                    article_path = asyncio.run(
                        run_agent(
                            input_value="https://youtu.be/hmtuvNfytjM",
                            question="写文章",
                            output_dir=tmpdir,
                            log_path=Path(tmpdir) / "agent.log",
                            progress_sink=events.append,
                        )
                    )
                    article_exists = article_path.exists()

        self.assertTrue(article_exists)
        phases = [event["phase"] for event in events]
        self.assertIn("source_fetch", phases)
        self.assertIn("article_write", phases)


if __name__ == "__main__":
    unittest.main()
