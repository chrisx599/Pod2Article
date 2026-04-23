from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_article import build_article, fetch_transcript_context, parse_metadata, search_candidates  # noqa: E402
from oxylabs_client import OxylabsError  # noqa: E402


class FakeClient:
    def __init__(self, fixtures: dict[str, dict]) -> None:
        self.fixtures = fixtures

    def search(self, query: str, source: str = "youtube_search", subtitles: bool = True) -> dict:
        return self.fixtures["search"]

    def metadata(self, video_id: str) -> dict:
        return self.fixtures["metadata"]

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


class FallbackClient(FakeClient):
    def fetch_best_timed_content(self, video_id: str, language_code: str = "en"):
        return type(
            "Probe",
            (),
            {
                "metadata": self.fixtures["metadata"],
                "content_payload": self.fixtures["subtitles"],
                "source_kind": "subtitles",
                "origin": "auto_generated",
            },
        )()


class FailingClient(FakeClient):
    def fetch_best_timed_content(self, video_id: str, language_code: str = "en"):
        raise OxylabsError("No transcript or subtitles available.")


class BuildArticleTestCase(unittest.TestCase):
    def _fixture(self, name: str) -> dict:
        path = Path(__file__).resolve().parent / "fixtures" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_build_article_from_url(self) -> None:
        fixtures = {
            "search": self._fixture("search_payload.json"),
            "metadata": self._fixture("metadata_payload.json"),
            "transcript": self._fixture("transcript_payload.json"),
            "subtitles": self._fixture("subtitles_payload.json"),
        }
        client = FakeClient(fixtures)
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = build_article(
                "https://www.youtube.com/watch?v=abc123def45",
                output_dir=Path(tmpdir),
                client=client,
            )
            content = destination.read_text(encoding="utf-8")
            self.assertTrue(destination.exists())
            self.assertIn("AI Podcast: Building Agents for Content Workflows - Article", content)
            self.assertIn("Source moment:", content)

    def test_build_article_from_search_query(self) -> None:
        fixtures = {
            "search": self._fixture("search_payload.json"),
            "metadata": self._fixture("metadata_payload.json"),
            "transcript": self._fixture("transcript_payload.json"),
            "subtitles": self._fixture("subtitles_payload.json"),
        }
        client = FakeClient(fixtures)
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = build_article(
                "ai podcast agents",
                output_dir=Path(tmpdir),
                client=client,
                search_source="youtube_search",
            )
            self.assertTrue(destination.exists())

    def test_fetch_transcript_context_returns_complete_source_payload(self) -> None:
        fixtures = {
            "search": self._fixture("search_payload.json"),
            "metadata": self._fixture("metadata_payload.json"),
            "transcript": self._fixture("transcript_payload.json"),
            "subtitles": self._fixture("subtitles_payload.json"),
        }
        client = FakeClient(fixtures)
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = fetch_transcript_context(
                "https://www.youtube.com/watch?v=abc123def45",
                output_dir=Path(tmpdir),
                client=client,
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(destination.name.endswith(".transcript.json"))
            self.assertEqual(payload["video"]["video_id"], "abc123def45")
            self.assertEqual(payload["coverage"]["segments_count"], len(payload["segments"]))
            self.assertGreater(payload["coverage"]["words_count"], 0)
            self.assertTrue(payload["agent_instructions"]["do_not_treat_as_article"])
            self.assertIn("text", payload["segments"][0])

    def test_search_candidates_supports_raw_oxylabs_payload(self) -> None:
        payload = self._fixture("raw_search_payload.json")
        candidates = search_candidates(payload, "Sam Altman GPT 5")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].video_id, "hmtuvNfytjM")
        self.assertEqual(candidates[0].channel, "Cleo Abram")
        self.assertEqual(candidates[0].duration_sec, 3907)

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

    def test_parse_metadata_supports_serpapi_video_payload(self) -> None:
        payload = self._fixture("serpapi_metadata_payload.json")
        parsed = parse_metadata(payload, "hmtuvNfytjM")
        self.assertEqual(parsed["title"], "Sam Altman Shows Me GPT 5... And What's Next")
        self.assertEqual(parsed["channel"], "Cleo Abram")
        self.assertEqual(parsed["language"], "en")
        self.assertEqual(parsed["chapters"][1]["start_time"], 964)

    def test_subtitles_fallback_path(self) -> None:
        fixtures = {
            "search": self._fixture("search_payload.json"),
            "metadata": self._fixture("metadata_payload.json"),
            "transcript": self._fixture("transcript_payload.json"),
            "subtitles": self._fixture("subtitles_payload.json"),
        }
        client = FallbackClient(fixtures)
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = build_article(
                "abc123def45",
                output_dir=Path(tmpdir),
                client=client,
            )
            content = destination.read_text(encoding="utf-8")
            self.assertIn("This fallback subtitle payload proves the builder can keep going", content)

    def test_missing_timed_content_raises_clear_error(self) -> None:
        fixtures = {
            "search": self._fixture("search_payload.json"),
            "metadata": self._fixture("metadata_payload.json"),
            "transcript": self._fixture("transcript_payload.json"),
            "subtitles": self._fixture("subtitles_payload.json"),
        }
        client = FailingClient(fixtures)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(OxylabsError):
                build_article("abc123def45", output_dir=Path(tmpdir), client=client)

    @unittest.skipUnless(
        os.environ.get("RUN_LIVE_API_TESTS") == "1"
        and "OXYLABS_USERNAME" in os.environ
        and "OXYLABS_PASSWORD" in os.environ,
        "Live Oxylabs smoke test requires RUN_LIVE_API_TESTS=1 and credentials.",
    )
    def test_live_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = build_article(
                "https://www.youtube.com/watch?v=abc123def45",
                output_dir=Path(tmpdir),
                provider="oxylabs",
            )
            self.assertTrue(destination.exists())


if __name__ == "__main__":
    unittest.main()
