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
    parse_serpapi_key,
    resolve_setting,
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
            env_path.write_text("SERPAPI_API_KEY=test-key\n", encoding="utf-8")
            old_key = os.environ.pop("SERPAPI_API_KEY", None)
            try:
                loaded = load_local_env(Path(tmpdir))
                self.assertEqual(loaded["SERPAPI_API_KEY"], "test-key")
                self.assertEqual(os.environ["SERPAPI_API_KEY"], "test-key")
            finally:
                if old_key is not None:
                    os.environ["SERPAPI_API_KEY"] = old_key
                else:
                    os.environ.pop("SERPAPI_API_KEY", None)

    def test_resolve_setting_prefers_config_over_environment_and_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pod2article.config").write_text("SERPAPI_API_KEY=config-key\n", encoding="utf-8")
            (root / ".env").write_text("SERPAPI_API_KEY=env-file-key\n", encoding="utf-8")
            old_key = os.environ.get("SERPAPI_API_KEY")
            os.environ["SERPAPI_API_KEY"] = "system-key"
            try:
                self.assertEqual(resolve_setting(("SERPAPI_API_KEY",), start_path=root), "config-key")
                self.assertEqual(parse_serpapi_key(root), "config-key")
            finally:
                if old_key is None:
                    os.environ.pop("SERPAPI_API_KEY", None)
                else:
                    os.environ["SERPAPI_API_KEY"] = old_key

    def test_resolve_setting_prefers_environment_over_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text("SERPAPI_API_KEY=env-file-key\n", encoding="utf-8")
            old_key = os.environ.get("SERPAPI_API_KEY")
            os.environ["SERPAPI_API_KEY"] = "system-key"
            try:
                self.assertEqual(resolve_setting(("SERPAPI_API_KEY",), start_path=root), "system-key")
            finally:
                if old_key is None:
                    os.environ.pop("SERPAPI_API_KEY", None)
                else:
                    os.environ["SERPAPI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
