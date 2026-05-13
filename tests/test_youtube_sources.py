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
from youtube_sources import fetch_transcript_context, parse_metadata, search_candidates, search_youtube_context  # noqa: E402


class FakeClient:
    def __init__(self, fixtures: dict[str, dict]) -> None:
        self.fixtures = fixtures

    def search(self, query: str) -> dict:
        return self.fixtures["search"]

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
                client=FakeClient(self._fixtures()),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(destination.name.endswith(".search.json"))
            self.assertEqual(payload["query"], "ai podcast agents")
            self.assertIn("query_hash", payload)
            self.assertEqual(payload["candidates"][0]["video_id"], "abc123def45")

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
            self.assertTrue(third.stem.endswith("-2.search") or third.name.endswith("-2.search.json"))

    def test_fetch_transcript_context_writes_complete_source_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = fetch_transcript_context(
                "https://www.youtube.com/watch?v=abc123def45",
                output_dir=Path(tmpdir),
                client=FakeClient(self._fixtures()),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(destination.name.endswith(".transcript.json"))
            self.assertEqual(payload["video"]["video_id"], "abc123def45")
            self.assertEqual(payload["coverage"]["segments_count"], len(payload["segments"]))
            self.assertGreater(payload["coverage"]["words_count"], 0)
            self.assertIn("text", payload["segments"][0])

    def test_search_candidates_supports_serpapi_payload(self) -> None:
        payload = self._fixture("serpapi_search_payload.json")
        candidates = search_candidates(payload, "Sam Altman GPT 5")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].video_id, "hmtuvNfytjM")

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

    def test_missing_timed_content_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SerpApiError):
                fetch_transcript_context("abc123def45", output_dir=Path(tmpdir), client=FailingClient(self._fixtures()))


if __name__ == "__main__":
    unittest.main()
