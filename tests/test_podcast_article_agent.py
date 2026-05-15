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
    normalize_youtube_timestamp_link_text,
    resolve_model,
    run_agent,
    serialize_message,
    should_prepare_discovery,
    _sdk_error_message,
    write_run_manifest,
    write_sources_manifest,
)
from agents.log_format import sanitize_for_log


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict


@dataclass
class FakeMessage:
    content: list
    session_id: str = "session-1"


@dataclass
class FakeResultMessage:
    is_error: bool = True
    api_error_status: int | None = 402
    result: str = "API Error: 402 Insufficient Balance"


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
            run_id="article-20260510T123000Z-abcdef12",
            research_mode="wide",
        )

        self.assertIn("Required adaptive wide-search workflow", prompt)
        self.assertIn("There is no fixed transcript count target", prompt)
        self.assertIn('fetch_transcript.py "<selected-video-url>"', prompt)
        self.assertIn("--run-id article-20260510T123000Z-abcdef12", prompt)
        self.assertIn("Enforce source diversity", prompt)
        self.assertIn("Use third-party analysis only as background context", prompt)
        self.assertIn("Do not let one long transcript dominate", prompt)
        self.assertIn("Open the prebuilt discovery artifacts", prompt)
        self.assertIn("search_queries: <prebuilt and supplemental search queries used>", prompt)
        self.assertNotIn(f"search_youtube.py {shlex.quote(question)}", prompt)

    def test_should_prepare_discovery_for_wide_or_search_query_deep(self) -> None:
        self.assertTrue(should_prepare_discovery("ai founder interviews", "deep"))
        self.assertTrue(should_prepare_discovery("ai founder interviews", "wide"))
        self.assertFalse(should_prepare_discovery("https://www.youtube.com/watch?v=hmtuvNfytjM", "deep"))

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

    def test_write_run_manifest_resets_created_at_for_new_run_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = build_workspace_paths(root, "demo")
            manifest_path = paths["workspace_dir"] / "run-manifest.json"
            article_path = paths["articles_root"] / "article-new" / "article.md"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps({"run_id": "old-run", "created_at": "2026-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )

            write_run_manifest(
                manifest_path,
                status="running",
                input_value="demo",
                question="写文章",
                source_id="demo",
                research_mode="deep",
                model="model",
                paths=paths,
                article_path=article_path,
                run_id="new-run",
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_id"], "new-run")
        self.assertNotEqual(payload["created_at"], "2026-01-01T00:00:00+00:00")

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

    def test_sdk_error_message_prefers_api_error_status(self) -> None:
        message = FakeResultMessage()

        self.assertEqual(_sdk_error_message(message), "API Error 402: Insufficient Balance")

    def test_log_sanitizer_preserves_usage_token_counts(self) -> None:
        payload = sanitize_for_log(
            {
                "ANTHROPIC_AUTH_TOKEN": "secret-token",
                "model_usage": {
                    "deepseek-v4-pro": {
                        "inputTokens": 123,
                        "outputTokens": 45,
                        "cacheReadInputTokens": 67,
                    }
                },
            }
        )

        self.assertEqual(payload["ANTHROPIC_AUTH_TOKEN"], "<set>")
        self.assertEqual(payload["model_usage"]["deepseek-v4-pro"]["inputTokens"], 123)
        self.assertEqual(payload["model_usage"]["deepseek-v4-pro"]["outputTokens"], 45)
        self.assertEqual(payload["model_usage"]["deepseek-v4-pro"]["cacheReadInputTokens"], 67)

    def test_find_transcript_context_returns_latest_non_empty_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            transcript_dir = Path(tmpdir)
            (transcript_dir / "empty.transcript.json").write_text("", encoding="utf-8")
            ready = transcript_dir / "ready.transcript.json"
            ready.write_text("{}", encoding="utf-8")

            self.assertEqual(find_transcript_context(transcript_dir), ready)

    def test_normalize_youtube_timestamp_link_text_rewrites_visible_text_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                "\n".join(
                    [
                        "[▶ 01:29](https://www.youtube.com/watch?v=hmtuvNfytjM&t=89s)",
                        "这是一句很长的正文说明，不应该被当成短介绍。[00:02:00](https://www.youtube.com/watch?v=hmtuvNfytjM&t=120s)",
                        "短句但不是介绍。[00:03:00](https://www.youtube.com/watch?v=hmtuvNfytjM&t=180s)",
                        "[姚顺宇谈模型同质化 07:05](https://www.youtube.com/watch?v=hmtuvNfytjM&amp;t=3723s)",
                        "[source](https://www.youtube.com/watch?v=hmtuvNfytjM)",
                    ]
                ),
                encoding="utf-8",
            )

            replacement_count = normalize_youtube_timestamp_link_text(
                article_path,
                sources_manifest={
                    "sources": [
                        {
                            "video_id": "hmtuvNfytjM",
                            "title": "140. 对姚顺宇的4小时访谈：请允许我小疯一下",
                            "channel": "Zhang Xiaojun Podcast",
                        }
                    ]
                },
            )
            text = article_path.read_text(encoding="utf-8")

        self.assertEqual(replacement_count, 4)
        self.assertIn("(姚顺宇访谈 [00:01:29](https://www.youtube.com/watch?v=hmtuvNfytjM&t=89s))", text)
        self.assertIn("(姚顺宇访谈 [00:02:00](https://www.youtube.com/watch?v=hmtuvNfytjM&t=120s))", text)
        self.assertIn("(姚顺宇访谈 [00:03:00](https://www.youtube.com/watch?v=hmtuvNfytjM&t=180s))", text)
        self.assertIn(
            "(姚顺宇谈模型同质化 [01:02:03](https://www.youtube.com/watch?v=hmtuvNfytjM&amp;t=3723s))",
            text,
        )
        self.assertIn("[source](https://www.youtube.com/watch?v=hmtuvNfytjM)", text)

    def test_normalize_youtube_timestamp_link_text_is_idempotent_after_added_cue(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                '- 一条很长的列表正文，后面已有短介绍 (姚顺宇访谈 [00:09:23](https://www.youtube.com/watch?v=hmtuvNfytjM&t=563s))',
                encoding="utf-8",
            )

            replacement_count = normalize_youtube_timestamp_link_text(
                article_path,
                sources_manifest={
                    "sources": [
                        {
                            "video_id": "hmtuvNfytjM",
                            "title": "140. 对姚顺宇的4小时访谈：请允许我小疯一下",
                        }
                    ]
                },
            )
            text = article_path.read_text(encoding="utf-8")

        self.assertEqual(replacement_count, 0)
        self.assertEqual(text.count("姚顺宇访谈"), 1)

    def test_normalize_youtube_timestamp_link_text_wraps_existing_short_cue(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                '已有短介绍 姚顺宇访谈 [00:09:23](https://www.youtube.com/watch?v=hmtuvNfytjM&t=563s)',
                encoding="utf-8",
            )

            replacement_count = normalize_youtube_timestamp_link_text(article_path)
            text = article_path.read_text(encoding="utf-8")

        self.assertEqual(replacement_count, 1)
        self.assertIn("(姚顺宇访谈 [00:09:23](https://www.youtube.com/watch?v=hmtuvNfytjM&t=563s))", text)

    def test_write_sources_manifest_filters_by_run_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            search_dir = root / "search-results"
            transcript_dir = root / "transcripts"
            search_dir.mkdir()
            transcript_dir.mkdir()
            (search_dir / "old.search.json").write_text(
                json.dumps(
                    {
                        "run_id": "old-run",
                        "query": "old query",
                        "candidates": [{"video_id": "old12345678", "title": "Old"}],
                    }
                ),
                encoding="utf-8",
            )
            (search_dir / "new.search.json").write_text(
                json.dumps(
                    {
                        "run_id": "new-run",
                        "query": "new query",
                        "candidates": [{"video_id": "new12345678", "title": "New", "score": 2.0}],
                    }
                ),
                encoding="utf-8",
            )
            (transcript_dir / "old.transcript.json").write_text(
                json.dumps({"run_id": "old-run", "video": {"video_id": "old12345678"}}),
                encoding="utf-8",
            )
            (transcript_dir / "new.transcript.json").write_text(
                json.dumps({"run_id": "new-run", "video": {"video_id": "new12345678"}, "coverage": {"segments_count": 1}}),
                encoding="utf-8",
            )
            article_path = root / "article.md"
            article_path.write_text("[`00:00`](https://www.youtube.com/watch?v=new12345678&t=0s)", encoding="utf-8")

            manifest = write_sources_manifest(
                root / "sources-manifest.json",
                search_dir=search_dir,
                transcript_dir=transcript_dir,
                article_path=article_path,
                run_id="new-run",
            )

        self.assertEqual(manifest["search_count"], 1)
        self.assertEqual(manifest["transcript_count"], 1)
        self.assertEqual([source["video_id"] for source in manifest["sources"]], ["new12345678"])

    def test_run_agent_writes_article_and_emits_progress(self) -> None:
        events: list[dict[str, object]] = []

        async def fake_query(prompt: str, options: object):
            transcript_dir = Path(prompt.split("Use this exact transcript output directory:")[1].splitlines()[1])
            article_path = Path(prompt.split("Write the final Markdown article only to this exact path:")[1].splitlines()[1])
            run_id = prompt.split("Use this exact run id:")[1].splitlines()[1]
            transcript_dir.mkdir(parents=True, exist_ok=True)
            (transcript_dir / "demo.transcript.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "video": {
                            "video_id": "hmtuvNfytjM",
                            "title": "Demo",
                            "channel": "Demo Channel",
                            "url": "https://www.youtube.com/watch?v=hmtuvNfytjM",
                        },
                        "source_kind": "transcript",
                        "coverage": {"segments_count": 1},
                        "segments": [{"text": "hello"}],
                    }
                ),
                encoding="utf-8",
            )
            article_path.parent.mkdir(parents=True, exist_ok=True)
            article_path.write_text(
                "# Demo\n\n[▶ start](https://www.youtube.com/watch?v=hmtuvNfytjM&t=0s)\n",
                encoding="utf-8",
            )
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
                    workspace_dir = article_path.parents[2]
                    run_manifest = json.loads((workspace_dir / "run-manifest.json").read_text(encoding="utf-8"))
                    sources_manifest = json.loads((workspace_dir / "sources-manifest.json").read_text(encoding="utf-8"))
                    article_manifest = json.loads((article_path.parent / "article-manifest.json").read_text(encoding="utf-8"))
                    quality_report = json.loads((workspace_dir / "quality-report.json").read_text(encoding="utf-8"))
                    article_text = article_path.read_text(encoding="utf-8")

        self.assertTrue(article_exists)
        self.assertEqual(run_manifest["status"], "completed")
        self.assertEqual(run_manifest["article_path"], str(article_path))
        self.assertEqual(run_manifest["artifacts"]["article_manifest"], str(article_path.parent / "article-manifest.json"))
        self.assertEqual(run_manifest["artifacts"]["quality_report"], str(workspace_dir / "quality-report.json"))
        self.assertEqual(run_manifest["artifact_summary"]["quality_status"], "passed")
        self.assertEqual(sources_manifest["transcript_count"], 1)
        self.assertEqual(sources_manifest["sources"][0]["video_id"], "hmtuvNfytjM")
        self.assertTrue(sources_manifest["sources"][0]["referenced_in_article"])
        self.assertEqual(article_manifest["timestamp_link_count"], 1)
        self.assertEqual(article_manifest["timestamp_link_text_issue_count"], 0)
        self.assertEqual(article_manifest["referenced_video_ids"], ["hmtuvNfytjM"])
        self.assertEqual(quality_report["status"], "passed")
        self.assertEqual(quality_report["issue_count"], 0)
        self.assertIn("[00:00:00]", article_text)
        phases = [event["phase"] for event in events]
        self.assertIn("source_fetch", phases)
        self.assertIn("article_write", phases)


if __name__ == "__main__":
    unittest.main()
