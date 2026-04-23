from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from normalize import Segment, merge_timed_segments, normalize_timed_content  # noqa: E402


class NormalizeTestCase(unittest.TestCase):
    def _fixture(self, name: str) -> dict:
        path = Path(__file__).resolve().parent / "fixtures" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_normalize_transcript_payload(self) -> None:
        payload = self._fixture("transcript_payload.json")
        segments = normalize_timed_content(payload, video_id="abc123def45", source_kind="transcript", language="en")
        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0].label, "Why podcast repurposing matters")
        self.assertEqual(segments[0].start_sec, 0)
        self.assertIn("Long-form podcasts", segments[0].text)

    def test_normalize_subtitles_payload(self) -> None:
        payload = self._fixture("subtitles_payload.json")
        segments = normalize_timed_content(payload, video_id="abc123def45", source_kind="subtitles", language="en")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1].start_sec, 7)

    def test_normalize_prefers_accessibility_label_when_available(self) -> None:
        payload = self._fixture("transcript_accessibility_payload.json")
        segments = normalize_timed_content(payload, video_id="abc123def45", source_kind="transcript", language="en")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_sec, 84)
        self.assertTrue(segments[0].text.startswith("How are you? Great to meet you. Thanks for doing this."))

    def test_normalize_serpapi_transcript_payload(self) -> None:
        payload = self._fixture("serpapi_transcript_payload.json")
        segments = normalize_timed_content(payload, video_id="hmtuvNfytjM", source_kind="transcript", language="en")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].start_sec, 0)
        self.assertEqual(segments[0].end_sec, 7)
        self.assertEqual(segments[0].label, "What future are we headed for?")
        self.assertEqual(segments[1].label, "GPT-5 demo")

    def test_merge_timed_segments_combines_short_adjacent_segments(self) -> None:
        segments = [
            Segment(0, 3, "This is a short segment.", "transcript", "en", "abc123def45"),
            Segment(3, 7, "Another adjacent line follows.", "transcript", "en", "abc123def45"),
        ]
        merged = merge_timed_segments(segments)
        self.assertEqual(len(merged), 1)
        self.assertIn("This is a short segment. Another adjacent line follows.", merged[0].text)


if __name__ == "__main__":
    unittest.main()
