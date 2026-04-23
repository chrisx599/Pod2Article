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
            self.assertEqual(payload["candidates"][0]["video_id"], "abc123def45")

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
