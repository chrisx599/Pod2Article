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
    build_evidence_prompt,
    build_initial_search_query_prompt,
    build_prompt,
    build_query_planner_prompt,
    build_run_metadata,
    build_source_id,
    build_workspace_paths,
    build_wide_article_prompt,
    default_log_path,
    extract_youtube_video_id,
    find_transcript_context,
    load_env_file,
    normalize_youtube_timestamp_link_text,
    resolve_evidence_model,
    resolve_model,
    run_agent,
    serialize_message,
    should_prepare_discovery,
    _extract_json_object_from_text,
    _sdk_error_message,
    write_run_manifest,
    write_evidence_manifest,
    write_web_evidence_cards,
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

        self.assertIn("Plan only supplemental YouTube and Web search queries", prompt)
        self.assertIn("Do not run Bash, Read, Write", prompt)
        self.assertIn("article-20260510T123000Z-abcdef12", prompt)
        self.assertIn("supplemental_youtube_queries", prompt)
        self.assertIn("supplemental_web_queries", prompt)
        self.assertIn("The Python runner, not you, will write the transcript fetch plan", prompt)
        self.assertIn("transcript-fetch-plan.json", prompt)
        self.assertIn("Every query must be a search-engine query", prompt)
        self.assertNotIn("Write a coherent Markdown article", prompt)
        self.assertNotIn(f"search_youtube.py {shlex.quote(question)}", prompt)

    def test_initial_search_query_prompt_rejects_task_sentences(self) -> None:
        prompt = build_initial_search_query_prompt(
            input_value="搜集近三个月以来 ai 行业的重要访谈，播客，写一份该行业的研判报告 interview talk",
            question="搜集近三个月以来 ai 行业的重要访谈，播客，写一份该行业的研判报告 interview talk",
        )

        self.assertIn("Generate initial YouTube search queries", prompt)
        self.assertIn("Each query must be a search-engine query", prompt)
        self.assertIn("Bad query:", prompt)
        self.assertIn("Good queries:", prompt)
        self.assertIn("AI industry interview podcast 2026", prompt)

    def test_query_planner_prompt_includes_candidate_summary_without_tool_use(self) -> None:
        with TemporaryDirectory() as tmpdir:
            selection_path = Path(tmpdir) / "selection-manifest.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "selected_candidates": [
                            {
                                "video_id": "abc12345678",
                                "title": "AI leader interview",
                                "channel": "Demo",
                                "published_date": "1 month ago",
                                "score": 9.5,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            prompt = build_query_planner_prompt(question="调研 AI", selection_manifest_path=selection_path)

        self.assertIn("AI leader interview", prompt)
        self.assertIn("Return only compact JSON", prompt)
        self.assertIn("Do not use tools", prompt)

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
        self.assertEqual(paths["evidence_dir"], Path("output/agent/hmtuvNfytjM/evidence"))
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

    def test_resolve_evidence_model_uses_specific_override(self) -> None:
        with patch.dict("os.environ", {"EVIDENCE_AGENT_MODEL": "deepseek-v4-flash"}, clear=True):
            self.assertEqual(resolve_evidence_model({}, "deepseek-v4-pro"), "deepseek-v4-flash")

        self.assertEqual(resolve_evidence_model({}, "deepseek-v4-pro"), "deepseek-v4-pro")

    def test_load_env_file_overrides_existing_environment(self) -> None:
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic\n"
                "ANTHROPIC_API_KEY=deepseek-key\n"
                "CLAUDE_AGENT_MODEL=deepseek-v4-pro\n"
                "EVIDENCE_AGENT_MODEL=deepseek-v4-flash\n",
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
                self.assertEqual(os.environ["EVIDENCE_AGENT_MODEL"], "deepseek-v4-flash")
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

    def test_build_evidence_and_wide_article_prompts_use_compact_cards(self) -> None:
        evidence_prompt = build_evidence_prompt(
            question="调研 AI 行业判断",
            transcript_path=Path("output/run/transcripts/demo.transcript.json"),
            evidence_path=Path("output/run/evidence/demo.evidence.json"),
        )
        article_prompt = build_wide_article_prompt(
            question="调研 AI 行业判断",
            evidence_manifest_path=Path("output/run/evidence/evidence-manifest.json"),
            web_evidence_path=Path("output/run/web-search/web-evidence.json"),
            article_path=Path("output/run/articles/article.md"),
            sources_manifest_path=Path("output/run/sources-manifest.json"),
        )

        self.assertIn("Extract compact, question-focused evidence cards", evidence_prompt)
        self.assertIn('"cards"', evidence_prompt)
        self.assertIn("Write one JSON object only to this exact path", evidence_prompt)
        self.assertIn("Read this evidence manifest first", article_prompt)
        self.assertIn("Use the evidence cards as the primary context", article_prompt)
        self.assertIn("Use web evidence only for background", article_prompt)
        self.assertIn("[马斯克 Lex Fridman 访谈 00:55:33]", article_prompt)

    def test_write_evidence_manifest_records_partial_failures(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "demo.transcript.json"
            transcript.write_text("{}", encoding="utf-8")

            manifest = write_evidence_manifest(
                root / "evidence-manifest.json",
                run_id="run-1",
                transcript_paths=[transcript],
                successes=[
                    {
                        "video_id": "hmtuvNfytjM",
                        "path": str(root / "hmtuvNfytjM.evidence.json"),
                        "card_count": 3,
                    }
                ],
                failures=[{"video_id": "failed12345", "error_message": "boom"}],
            )

        self.assertEqual(manifest["transcript_count"], 1)
        self.assertEqual(manifest["success_count"], 1)
        self.assertEqual(manifest["failed_count"], 1)

    def test_write_web_evidence_cards_collects_snippet_cards(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            web_dir = root / "web-search"
            web_dir.mkdir()
            (web_dir / "demo.web-search.json").write_text(
                json.dumps(
                    {
                        "run_id": "article-1",
                        "query": "ai agents market",
                        "results": [
                            {
                                "rank": 1,
                                "result_type": "organic_results",
                                "title": "AI agents market update",
                                "url": "https://example.com/ai-agents",
                                "source": "Example",
                                "date": "May 2026",
                                "snippet": "AI agent adoption is expanding.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = write_web_evidence_cards(
                web_search_dir=web_dir,
                web_evidence_path=web_dir / "web-evidence.json",
                run_id="article-1",
            )

        self.assertEqual(manifest["card_count"], 1)
        self.assertEqual(manifest["cards"][0]["source_kind"], "web")
        self.assertEqual(manifest["cards"][0]["url"], "https://example.com/ai-agents")

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

    def test_extract_json_object_from_text_handles_duplicated_sdk_json_messages(self) -> None:
        text = (
            '{"schema_version": 1, "supplemental_web_queries": [{"query": "ai timeline"}]}\n'
            '{"schema_version": 1, "supplemental_web_queries": [{"query": "ai timeline"}]}'
        )

        payload = _extract_json_object_from_text(text)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["supplemental_web_queries"][0]["query"], "ai timeline")

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
        self.assertIn("[姚顺宇访谈 00:01:29](https://www.youtube.com/watch?v=hmtuvNfytjM&t=89s)", text)
        self.assertIn("[姚顺宇访谈 00:02:00](https://www.youtube.com/watch?v=hmtuvNfytjM&t=120s)", text)
        self.assertIn("[姚顺宇访谈 00:03:00](https://www.youtube.com/watch?v=hmtuvNfytjM&t=180s)", text)
        self.assertIn(
            "[姚顺宇访谈 01:02:03](https://www.youtube.com/watch?v=hmtuvNfytjM&amp;t=3723s)",
            text,
        )
        self.assertIn("[source](https://www.youtube.com/watch?v=hmtuvNfytjM)", text)

    def test_normalize_youtube_timestamp_link_text_is_idempotent_after_added_cue(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                '- 一条很长的列表正文，后面已有短介绍 [姚顺宇访谈 00:09:23](https://www.youtube.com/watch?v=hmtuvNfytjM&t=563s)',
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

    def test_normalize_youtube_timestamp_link_text_collapses_existing_short_cue_wrapper(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                '已有短介绍 (姚顺宇访谈 [00:09:23](https://www.youtube.com/watch?v=hmtuvNfytjM&t=563s))',
                encoding="utf-8",
            )

            replacement_count = normalize_youtube_timestamp_link_text(article_path)
            text = article_path.read_text(encoding="utf-8")

        self.assertEqual(replacement_count, 2)
        self.assertIn("[姚顺宇访谈 00:09:23](https://www.youtube.com/watch?v=hmtuvNfytjM&t=563s)", text)
        self.assertNotIn("(姚顺宇访谈 [", text)

    def test_normalize_youtube_timestamp_link_text_collapses_nested_old_wrapper_with_consistent_video_cue(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                "（NVIDIA 全员使用 AI (编程 [00:46:45](https://www.youtube.com/watch?v=PirWDBZlrVg&t=2805s))）",
                encoding="utf-8",
            )

            replacement_count = normalize_youtube_timestamp_link_text(
                article_path,
                sources_manifest={
                    "sources": [
                        {
                            "video_id": "PirWDBZlrVg",
                            "title": "NVIDIA 开发者大会：黄仁勋谈 AI 编程",
                        }
                    ]
                },
            )
            text = article_path.read_text(encoding="utf-8")

        self.assertGreaterEqual(replacement_count, 2)
        self.assertIn("[NVIDIA 开发者大会 00:46:45](https://www.youtube.com/watch?v=PirWDBZlrVg&t=2805s)", text)
        self.assertNotIn("[00:46:45]", text)

    def test_normalize_youtube_timestamp_link_text_uses_more_complete_title_cue(self) -> None:
        with TemporaryDirectory() as tmpdir:
            article_path = Path(tmpdir) / "article.md"
            article_path.write_text(
                "[00:17:31](https://www.youtube.com/watch?v=cyZeAw8DLew&t=1051s)",
                encoding="utf-8",
            )

            replacement_count = normalize_youtube_timestamp_link_text(
                article_path,
                sources_manifest={
                    "sources": [
                        {
                            "video_id": "cyZeAw8DLew",
                            "title": "深度对话00后CEO，重新定义 AI 原生公司与产品",
                        }
                    ]
                },
            )
            text = article_path.read_text(encoding="utf-8")

        self.assertEqual(replacement_count, 1)
        self.assertIn(
            "[深度对话00后CEO，重新定义 AI 原生公司与产品 00:17:31](https://www.youtube.com/watch?v=cyZeAw8DLew&t=1051s)",
            text,
        )

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
        self.assertIn("[Demo 00:00:00]", article_text)
        phases = [event["phase"] for event in events]
        self.assertIn("source_fetch", phases)
        self.assertIn("article_write", phases)

    def test_run_agent_wide_extracts_evidence_before_article_write(self) -> None:
        events: list[dict[str, object]] = []
        prompts: list[str] = []
        discovery_inputs: dict[str, object] = {}

        def fake_prepare_discovery(
            *,
            input_value: str,
            question: str,
            research_mode: str,
            workspace_dir: Path,
            search_dir: Path,
            run_id: str,
            planned_queries: list[dict[str, object]] | None = None,
        ) -> dict[str, Path]:
            discovery_inputs["planned_queries"] = planned_queries
            search_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                (search_dir / f"demo-{index}.search.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "query": f"demo {index}",
                            "candidates": [
                                {
                                    "video_id": f"widevideo0{index}",
                                    "title": f"Demo {index}",
                                    "score": 1.0,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            research_plan = workspace_dir / "research-plan.json"
            video_enrichment = workspace_dir / "video-enrichment-manifest.json"
            selection = workspace_dir / "selection-manifest.json"
            research_plan.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            video_enrichment.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            selection.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "candidate_count": 2,
                        "search_round_count": 1,
                        "selected_candidates": [
                            {
                                "video_id": f"widevideo0{index}",
                                "title": f"Demo {index}",
                                "channel": "Demo Channel",
                                "url": f"https://www.youtube.com/watch?v=widevideo0{index}",
                                "score": 10 - index,
                            }
                            for index in range(2)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return {
                "research_plan": research_plan,
                "video_enrichment_manifest": video_enrichment,
                "selection_manifest": selection,
            }

        async def fake_query(prompt: str, options: object):
            prompts.append(prompt)
            if "Generate initial YouTube search queries" in prompt:
                await asyncio.sleep(0)
                yield FakeResultMessage(
                    is_error=False,
                    api_error_status=None,
                    result=json.dumps(
                        {
                            "schema_version": 1,
                            "youtube_search_queries": [
                                {
                                    "query": "AI industry interview podcast 2026",
                                    "reason": "broad English interview discovery",
                                    "language": "en",
                                },
                                {
                                    "query": "人工智能 行业 访谈 播客 2026",
                                    "reason": "Chinese interview discovery",
                                    "language": "zh",
                                },
                            ],
                        }
                    ),
                )
                return
            if "Plan only supplemental YouTube and Web search queries" in prompt:
                await asyncio.sleep(0)
                yield FakeResultMessage(
                    is_error=False,
                    api_error_status=None,
                    result=json.dumps(
                        {
                            "schema_version": 1,
                            "supplemental_youtube_queries": [],
                            "supplemental_web_queries": [],
                        }
                    ),
                )
                return
            elif "Extract compact, question-focused evidence cards" in prompt:
                evidence_path = Path(prompt.split("Write one JSON object only to this exact path:")[1].splitlines()[1])
                transcript_path = Path(prompt.split("Read this transcript JSON:")[1].splitlines()[1])
                transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
                video = transcript_payload["video"]
                start_sec = transcript_payload["segments"][0]["start_sec"]
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "video_id": video["video_id"],
                            "title": video["title"],
                            "channel": video["channel"],
                            "source_kind": "transcript",
                            "transcript_path": str(transcript_path),
                            "relevance": "high",
                            "coverage_note": "完整",
                            "excluded": False,
                            "exclusion_reason": "",
                            "cards": [
                                {
                                    "claim": "核心观点",
                                    "why_it_matters": "回答问题",
                                    "timestamp": "00:00:00",
                                    "start_sec": start_sec,
                                    "url": f"https://www.youtube.com/watch?v={video['video_id']}&t={start_sec}s",
                                    "quote_or_paraphrase": "核心观点",
                                    "source_cue": "Demo观点",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            elif "Write the final grounded wide video deep research article" in prompt:
                article_path = Path(prompt.split("Write the final Markdown article only to this exact path:")[1].splitlines()[1])
                article_path.parent.mkdir(parents=True, exist_ok=True)
                article_path.write_text(
                    "# Wide Demo\n\n行业判断来自多条证据。[Demo 0 00:00:00](https://www.youtube.com/watch?v=widevideo00&t=0s)\n",
                    encoding="utf-8",
                )
            await asyncio.sleep(0)
            yield object()

        def fake_fetch_transcript_context(raw_input: str, *, output_dir: Path, run_id: str, **kwargs: object) -> Path:
            video_id = raw_input.rsplit("=", 1)[-1]
            index = int(video_id[-1])
            output_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = output_dir / f"demo-{index}.transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "video": {
                            "video_id": video_id,
                            "title": f"Demo {index}",
                            "channel": "Demo Channel",
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                        },
                        "source_kind": "transcript",
                        "coverage": {"segments_count": 1},
                        "segments": [
                            {
                                "start_sec": index * 60,
                                "timestamp": f"00:0{index}:00",
                                "url": f"https://www.youtube.com/watch?v={video_id}&t={index * 60}s",
                                "text": "核心观点",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return transcript_path

        with patch("agents.podcast_article_agent.query", fake_query):
            with patch("agents.podcast_article_agent.ResultMessage", FakeResultMessage):
                with patch("agents.podcast_article_agent.prepare_research_discovery", fake_prepare_discovery):
                    with patch("agents.podcast_article_agent.fetch_transcript_context", fake_fetch_transcript_context):
                        with patch("agents.podcast_article_agent.ARTIFACT_PROGRESS_POLL_SECONDS", 0.01):
                            with TemporaryDirectory() as tmpdir:
                                article_path = asyncio.run(
                                    run_agent(
                                        input_value="调研 AI 行业判断",
                                        question="调研 AI 行业判断",
                                        output_dir=tmpdir,
                                        log_path=Path(tmpdir) / "agent.log",
                                        progress_sink=events.append,
                                    )
                                )
                                workspace_dir = article_path.parents[2]
                                run_manifest = json.loads((workspace_dir / "run-manifest.json").read_text(encoding="utf-8"))
                                query_plan = json.loads((workspace_dir / "query-plan.json").read_text(encoding="utf-8"))
                                fetch_manifest = json.loads(
                                    (workspace_dir / "transcript-fetch-manifest.json").read_text(encoding="utf-8")
                                )
                                evidence_manifest = json.loads(
                                    (workspace_dir / "evidence" / "evidence-manifest.json").read_text(encoding="utf-8")
                                )
                                quality_report = json.loads((workspace_dir / "quality-report.json").read_text(encoding="utf-8"))

        self.assertTrue(any("Plan only supplemental YouTube and Web search queries" in prompt for prompt in prompts))
        self.assertTrue(any("Generate initial YouTube search queries" in prompt for prompt in prompts))
        self.assertEqual(len([prompt for prompt in prompts if "Extract compact, question-focused evidence cards" in prompt]), 2)
        self.assertTrue(any("Write the final grounded wide video deep research article" in prompt for prompt in prompts))
        planned_queries = discovery_inputs["planned_queries"]
        self.assertIsInstance(planned_queries, list)
        self.assertEqual(planned_queries[0]["query"], "AI industry interview podcast 2026")
        self.assertEqual(query_plan["supplemental_queries"], [])
        self.assertEqual(query_plan["supplemental_web_queries"], [])
        self.assertEqual(evidence_manifest["success_count"], 2)
        self.assertEqual(fetch_manifest["success_count"], 2)
        self.assertEqual(evidence_manifest["failed_count"], 0)
        self.assertEqual(run_manifest["artifacts"]["query_plan"], str(workspace_dir / "query-plan.json"))
        self.assertEqual(run_manifest["artifacts"]["transcript_fetch_manifest"], str(workspace_dir / "transcript-fetch-manifest.json"))
        self.assertEqual(run_manifest["artifacts"]["evidence_manifest"], str(workspace_dir / "evidence" / "evidence-manifest.json"))
        self.assertEqual(quality_report["evidence_success_count"], 2)
        phases = [event["phase"] for event in events]
        self.assertIn("evidence_extract", phases)
        self.assertIn("article_write", phases)


if __name__ == "__main__":
    unittest.main()
