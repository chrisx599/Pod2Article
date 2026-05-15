from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from serpapi_client import SerpApiError  # noqa: E402
from youtube_sources import (  # noqa: E402
    extract_related_search_queries,
    fetch_transcript_context,
    normalize_video_enrichment,
    parse_metadata,
    prepare_research_discovery,
    search_candidates,
    search_youtube_context,
)


class FakeClient:
    def __init__(self, fixtures: dict[str, dict]) -> None:
        self.fixtures = fixtures

    def search(self, query: str) -> dict:
        return self.fixtures["search"]

    def metadata(self, video_id: str) -> dict:
        payload = dict(self.fixtures["metadata"])
        payload.setdefault(
            "related_videos",
            [
                {
                    "title": "Related AI Founder Interview",
                    "link": "https://www.youtube.com/watch?v=rel12345678",
                    "video_id": "rel12345678",
                    "channel": {"name": "Related Channel"},
                    "length": "45:00",
                }
            ],
        )
        payload.setdefault("transcript", {"link": "https://serpapi.com/search.json?engine=youtube_video_transcript"})
        return payload

    def fetch_best_timed_content(self, video_id: str, language_code: str = "en"):
        return type(
            "Probe",
            (),
            {
                "metadata": self.fixtures["metadata"],
                "content_payload": self.fixtures["transcript"],
                "source_kind": "transcript",
                "origin": "uploader_provided",
            },
        )()


class FailingClient(FakeClient):
    def fetch_best_timed_content(self, video_id: str, language_code: str = "en"):
        raise SerpApiError("No transcript or subtitles available.")


class FakeAliyunAsrClient:
    def transcribe_youtube_video(self, video_id: str) -> dict:
        return {
            "provider": "aliyun_asr",
            "engine": "dashscope_asr",
            "transcript": [
                {
                    "snippet": "阿里云转写提供了第一段时间戳文本。",
                    "start_ms": 0,
                    "end_ms": 3500,
                },
                {
                    "snippet": "当 SerpApi 没有字幕时，仍然可以生成上下文。",
                    "start_ms": 3500,
                    "end_ms": 9000,
                },
            ],
            "raw_result": {
                "transcripts": [
                    {
                        "sentences": [
                            {"begin_time": 0, "end_time": 3500, "text": "阿里云转写提供了第一段时间戳文本。"},
                            {"begin_time": 3500, "end_time": 9000, "text": "当 SerpApi 没有字幕时，仍然可以生成上下文。"},
                        ]
                    }
                ]
            },
        }


class YouTubeSourcesTestCase(unittest.TestCase):
    def _fixture(self, name: str) -> dict:
        path = Path(__file__).resolve().parent / "fixtures" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def _fixtures(self) -> dict[str, dict]:
        return {
            "search": self._fixture("search_payload.json"),
            "metadata": self._fixture("metadata_payload.json"),
            "transcript": self._fixture("transcript_payload.json"),
        }

    def test_search_youtube_context_writes_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = search_youtube_context(
                "ai podcast agents",
                output_dir=Path(tmpdir),
                run_id="article-1",
                client=FakeClient(self._fixtures()),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(destination.name.endswith(".search.json"))
            self.assertEqual(payload["run_id"], "article-1")
            self.assertEqual(payload["query"], "ai podcast agents")
            self.assertEqual(payload["canonical_query"], "ai podcast agents")
            self.assertIn("query_hash", payload)
            raw_path = Path(payload["raw_output_path"])
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertTrue(raw_path.name.endswith(".raw-search.json"))
            self.assertEqual(raw_payload["run_id"], "article-1")
            self.assertIn("payload", raw_payload)
            self.assertEqual(payload["candidates"][0]["video_id"], "abc123def45")
            self.assertIn("score_breakdown", payload["candidates"][0])
            manifest = json.loads((Path(tmpdir) / "search-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["search_count"], 1)
            self.assertEqual(manifest["searches"][0]["round"], 1)
            self.assertEqual(manifest["searches"][0]["run_id"], "article-1")
            self.assertEqual(manifest["searches"][0]["output_path"], str(destination))
            self.assertEqual(manifest["searches"][0]["raw_output_path"], str(raw_path))

    def test_search_youtube_context_preserves_colliding_query_slugs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            first = search_youtube_context(
                "中国 AI 领袖 访谈 人工智能 发展 2025",
                output_dir=output_dir,
                client=FakeClient(self._fixtures()),
            )
            second = search_youtube_context(
                "李开复 周鸿祎 王小川 AI 访谈 2025",
                output_dir=output_dir,
                client=FakeClient(self._fixtures()),
            )
            third = search_youtube_context(
                "李开复 周鸿祎 王小川 AI 访谈 2025",
                output_dir=output_dir,
                client=FakeClient(self._fixtures()),
            )

            self.assertNotEqual(first.name, second.name)
            self.assertNotEqual(second.name, third.name)
            self.assertEqual(len(list(output_dir.glob("*.search.json"))), 3)
            self.assertTrue(third.name.endswith("-2.search.json"))
            manifest = json.loads((output_dir / "search-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["search_count"], 3)
            self.assertEqual([item["round"] for item in manifest["searches"]], [1, 2, 3])

    def test_fetch_transcript_context_writes_complete_source_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = fetch_transcript_context(
                "https://www.youtube.com/watch?v=abc123def45",
                output_dir=Path(tmpdir),
                run_id="article-1",
                client=FakeClient(self._fixtures()),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(destination.name.endswith(".transcript.json"))
            self.assertEqual(payload["run_id"], "article-1")
            self.assertEqual(payload["video"]["video_id"], "abc123def45")
            self.assertEqual(payload["coverage"]["segments_count"], len(payload["segments"]))
            self.assertGreater(payload["coverage"]["words_count"], 0)
            self.assertIn("text", payload["segments"][0])

    def test_search_candidates_supports_serpapi_payload(self) -> None:
        payload = self._fixture("serpapi_search_payload.json")
        candidates = search_candidates(payload, "Sam Altman GPT 5")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].video_id, "hmtuvNfytjM")
        self.assertEqual(candidates[0].source_bucket, "video_results")
        self.assertIn("title_match", candidates[0].score_breakdown or {})

    def test_search_candidates_boosts_serpapi_channel_match(self) -> None:
        payload = self._fixture("serpapi_search_payload.json")
        candidates = search_candidates(payload, "Sam Altman GPT 5 Cleo Abram")
        self.assertEqual(candidates[0].video_id, "hmtuvNfytjM")
        self.assertEqual(candidates[0].channel, "Cleo Abram")
        self.assertEqual(candidates[0].url, "https://www.youtube.com/watch?v=hmtuvNfytjM")
        self.assertEqual(candidates[0].duration_sec, 3907)

    def test_search_candidates_rank_chinese_entity_matches_over_position(self) -> None:
        payload = {
            "video_results": [
                {
                    "position_on_page": 1,
                    "title": "科技新闻三分钟",
                    "link": "https://www.youtube.com/watch?v=news1234567",
                    "video_id": "news1234567",
                    "channel": {"name": "News Byte"},
                    "length": "2:30",
                    "views": "10万次观看",
                },
                {
                    "position_on_page": 2,
                    "title": "李开复 对谈 王小川：中国 AI 发展与大模型创业",
                    "link": "https://www.youtube.com/watch?v=deep1234567",
                    "video_id": "deep1234567",
                    "channel": {"name": "AI 访谈"},
                    "length": "1:12:00",
                    "views": "5万次观看",
                    "description": "人工智能 发展 访谈",
                },
            ]
        }
        candidates = search_candidates(payload, "李开复 王小川 AI 访谈 人工智能 发展")
        self.assertEqual(candidates[0].video_id, "deep1234567")
        self.assertGreater(candidates[0].score, candidates[1].score)

    def test_search_candidates_penalize_reaction_noise_when_original_is_available(self) -> None:
        payload = {
            "video_results": [
                {
                    "position_on_page": 1,
                    "title": "Sam Altman GPT-5 Podcast Reaction Clip",
                    "link": "https://www.youtube.com/watch?v=react123456",
                    "video_id": "react123456",
                    "channel": {"name": "Reaction Lab"},
                    "length": "4:00",
                    "views": "100K views",
                },
                {
                    "position_on_page": 2,
                    "title": "Sam Altman on GPT-5 and the Future of AI",
                    "link": "https://www.youtube.com/watch?v=orig1234567",
                    "video_id": "orig1234567",
                    "channel": {"name": "Cleo Abram"},
                    "length": "1:05:00",
                    "views": "4.5M views",
                    "description": "Full interview with Sam Altman about GPT-5.",
                },
            ]
        }
        candidates = search_candidates(payload, "Sam Altman GPT 5 interview")
        self.assertEqual(candidates[0].video_id, "orig1234567")

    def test_search_candidates_prefer_direct_chinese_leader_sources_over_analysis(self) -> None:
        payload = {
            "video_results": [
                {
                    "position_on_page": 1,
                    "title": "Is China Winning the A.I. Race?",
                    "link": "https://www.youtube.com/watch?v=analysis123",
                    "video_id": "analysis123",
                    "channel": {"name": "New York Times Podcasts"},
                    "length": "29:42",
                    "views": "200K views",
                    "description": "A news analysis about China's AI race.",
                },
                {
                    "position_on_page": 2,
                    "title": "Alibaba's Joe Tsai on China's AI Future",
                    "link": "https://www.youtube.com/watch?v=direct12345",
                    "video_id": "direct12345",
                    "channel": {"name": "All-In Podcast"},
                    "length": "27:07",
                    "views": "100K views",
                    "description": "Interview with Alibaba chair Joe Tsai.",
                },
            ]
        }
        candidates = search_candidates(payload, "China AI founder podcast 2025")
        self.assertEqual(candidates[0].video_id, "direct12345")

    def test_parse_metadata_supports_serpapi_video_payload(self) -> None:
        payload = self._fixture("serpapi_metadata_payload.json")
        parsed = parse_metadata(payload, "hmtuvNfytjM")
        self.assertEqual(parsed["title"], "Sam Altman Shows Me GPT 5... And What's Next")
        self.assertEqual(parsed["channel"], "Cleo Abram")
        self.assertEqual(parsed["language"], "en")
        self.assertEqual(parsed["chapters"][1]["start_time"], 964)

    def test_extract_related_search_queries_and_video_enrichment(self) -> None:
        search_payload = {
            "related_searches": [{"query": "sam altman full interview"}, {"title": "openai keynote"}]
        }
        self.assertEqual(
            extract_related_search_queries(search_payload),
            ["sam altman full interview", "openai keynote"],
        )
        metadata_payload = self._fixture("serpapi_metadata_payload.json")
        metadata_payload["related_videos"] = [
            {
                "title": "Sam Altman Full Interview",
                "link": "https://www.youtube.com/watch?v=rel12345678",
                "video_id": "rel12345678",
                "channel": {"name": "Cleo Abram"},
                "length": "1:10:00",
            }
        ]
        metadata_payload["transcript_link"] = "https://serpapi.com/search.json?engine=youtube_video_transcript&v=hmtuvNfytjM"
        enrichment = normalize_video_enrichment("hmtuvNfytjM", metadata_payload)
        self.assertTrue(enrichment["has_transcript_link"])
        self.assertEqual(enrichment["related_video_ids"], ["rel12345678"])

    def test_prepare_research_discovery_writes_adaptive_artifacts(self) -> None:
        fixtures = self._fixtures()
        fixtures["search"] = {
            **fixtures["search"],
            "related_searches": [{"query": "ai agents founder interview"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = prepare_research_discovery(
                input_value="ai agents",
                question="请调研 AI agents 的行业访谈",
                research_mode="wide",
                workspace_dir=root,
                search_dir=root / "search-results",
                run_id="article-1",
                client=FakeClient(fixtures),
            )
            plan = json.loads(artifacts["research_plan"].read_text(encoding="utf-8"))
            enrichment = json.loads(artifacts["video_enrichment_manifest"].read_text(encoding="utf-8"))
            selection = json.loads(artifacts["selection_manifest"].read_text(encoding="utf-8"))

        self.assertEqual(plan["transcript_policy"]["mode"], "model_decides")
        self.assertGreaterEqual(len(plan["queries"]), 2)
        self.assertGreaterEqual(enrichment["enrichment_count"], 1)
        self.assertEqual(selection["transcript_policy"]["mode"], "adaptive")
        self.assertGreaterEqual(selection["candidate_count"], 1)

    def test_prepare_research_discovery_tolerates_related_video_without_duration(self) -> None:
        fixtures = self._fixtures()
        fixtures["metadata"] = {
            **fixtures["metadata"],
            "related_videos": [
                {
                    "title": "Related interview without duration",
                    "link": "https://www.youtube.com/watch?v=noduration1",
                    "video_id": "noduration1",
                    "channel": {"name": "Related Channel"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = prepare_research_discovery(
                input_value="ai agents",
                question="ai agents",
                research_mode="wide",
                workspace_dir=root,
                search_dir=root / "search-results",
                run_id="article-1",
                client=FakeClient(fixtures),
            )
            selection = json.loads(artifacts["selection_manifest"].read_text(encoding="utf-8"))

        self.assertGreaterEqual(selection["candidate_count"], 1)

    def test_missing_timed_content_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SerpApiError):
                fetch_transcript_context("abc123def45", output_dir=Path(tmpdir), client=FailingClient(self._fixtures()))

    def test_fetch_transcript_context_falls_back_to_aliyun_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = fetch_transcript_context(
                "https://www.youtube.com/watch?v=abc123def45",
                output_dir=Path(tmpdir),
                run_id="article-aliyun",
                client=FailingClient(self._fixtures()),
                asr_client=FakeAliyunAsrClient(),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["provider"], "aliyun_asr")
        self.assertEqual(payload["source_kind"], "asr")
        self.assertEqual(payload["origin"], "aliyun_asr")
        self.assertGreaterEqual(payload["coverage"]["segments_count"], 1)
        self.assertIn("阿里云转写", payload["segments"][0]["text"])


if __name__ == "__main__":
    unittest.main()
