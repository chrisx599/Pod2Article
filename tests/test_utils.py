from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import (  # noqa: E402
    build_youtube_timestamp_url,
    detect_input_type,
    extract_video_id,
    format_timestamp,
    load_local_env,
    slugify,
)


class UtilsTestCase(unittest.TestCase):
    def test_detect_input_type(self) -> None:
        self.assertEqual(detect_input_type("https://www.youtube.com/watch?v=abc123def45"), "youtube_url")
        self.assertEqual(detect_input_type("abc123def45"), "video_id")
        self.assertEqual(detect_input_type("best ai podcast agents"), "search_query")

    def test_extract_video_id(self) -> None:
        self.assertEqual(extract_video_id("https://youtu.be/abc123def45"), "abc123def45")
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=abc123def45"), "abc123def45")
        self.assertIsNone(extract_video_id("https://example.com/video"))

    def test_helpers(self) -> None:
        self.assertEqual(slugify("AI Podcast: Building Agents"), "ai-podcast-building-agents")
        self.assertEqual(format_timestamp(754), "00:12:34")
        self.assertEqual(
            build_youtube_timestamp_url("abc123def45", 754),
            "https://www.youtube.com/watch?v=abc123def45&t=754s",
        )

    def test_load_local_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OXYLABS_USERNAME=test-user\nOXYLABS_PASSWORD=test-pass\n", encoding="utf-8")
            old_user = os.environ.pop("OXYLABS_USERNAME", None)
            old_pass = os.environ.pop("OXYLABS_PASSWORD", None)
            try:
                loaded = load_local_env(Path(tmpdir))
                self.assertEqual(loaded["OXYLABS_USERNAME"], "test-user")
                self.assertEqual(os.environ["OXYLABS_PASSWORD"], "test-pass")
            finally:
                if old_user is not None:
                    os.environ["OXYLABS_USERNAME"] = old_user
                else:
                    os.environ.pop("OXYLABS_USERNAME", None)
                if old_pass is not None:
                    os.environ["OXYLABS_PASSWORD"] = old_pass
                else:
                    os.environ.pop("OXYLABS_PASSWORD", None)


if __name__ == "__main__":
    unittest.main()
