from __future__ import annotations

import unittest
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from oxylabs_client import OxylabsClient  # noqa: E402


class StubClient(OxylabsClient):
    def __init__(self) -> None:
        super().__init__("user", "pass")

    def metadata(self, video_id: str) -> dict:
        return {"results": [{"status_code": 200, "content": {"results": {"title": "Demo"}}}]}

    def transcript(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict:
        return {
            "results": [
                {
                    "status_code": 613,
                    "content": "",
                }
            ]
        }

    def subtitles(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict:
        return {
            "results": [
                {
                    "status_code": 200,
                    "content": {
                        "auto_generated": {
                            "en": {
                                "events": [
                                    {
                                        "tStartMs": 0,
                                        "dDurationMs": 1200,
                                        "segs": [{"utf8": "Hello"}, {"utf8": " world"}],
                                    }
                                ]
                            }
                        }
                    },
                }
            ]
        }


class OxylabsClientTestCase(unittest.TestCase):
    def test_fetch_best_timed_content_falls_back_to_subtitles(self) -> None:
        client = StubClient()
        probe = client.fetch_best_timed_content("abc123def45", language_code="en")
        self.assertEqual(probe.source_kind, "subtitles")
        self.assertEqual(probe.origin, "uploader_provided")


if __name__ == "__main__":
    unittest.main()
