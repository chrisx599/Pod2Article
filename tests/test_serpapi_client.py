from __future__ import annotations

import unittest
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from serpapi_client import SerpApiClient  # noqa: E402


class StubSerpApiClient(SerpApiClient):
    def __init__(self) -> None:
        super().__init__("key")

    def metadata(self, video_id: str) -> dict:
        return {
            "search_metadata": {"status": "Success"},
            "title": "Demo",
            "channel": {"name": "Channel"},
        }

    def transcript(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict:
        return {
            "search_metadata": {"status": "Success"},
            "transcript": [
                {
                    "start_ms": 0,
                    "end_ms": 1200,
                    "snippet": "Hello world",
                }
            ],
        }


class SerpApiClientTestCase(unittest.TestCase):
    def test_fetch_best_timed_content_uses_transcript_payload(self) -> None:
        client = StubSerpApiClient()
        probe = client.fetch_best_timed_content("abc123def45", language_code="en")
        self.assertEqual(probe.source_kind, "transcript")
        self.assertEqual(probe.origin, "uploader_provided")
        self.assertIn("transcript", probe.content_payload)


if __name__ == "__main__":
    unittest.main()
