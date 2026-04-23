from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from article_builder import build_outline_sections, render_article_markdown  # noqa: E402
from normalize import Segment, normalize_timed_content  # noqa: E402


class ArticleBuilderTestCase(unittest.TestCase):
    def _fixture(self, name: str) -> dict:
        path = Path(__file__).resolve().parent / "fixtures" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_render_article_markdown(self) -> None:
        transcript = self._fixture("transcript_payload.json")
        template_path = Path(__file__).resolve().parents[1] / "podcast-to-article" / "templates" / "article-template.md"
        template_text = template_path.read_text(encoding="utf-8")
        segments = normalize_timed_content(transcript, video_id="abc123def45", source_kind="transcript", language="en")
        sections = build_outline_sections(segments, target_sections=2)
        markdown = render_article_markdown(
            title="Demo Article",
            source_title="AI Podcast",
            channel="Agent Lab",
            video_url="https://www.youtube.com/watch?v=abc123def45",
            language="en",
            sections=sections,
            template_text=template_text,
        )
        self.assertIn("# Demo Article", markdown)
        self.assertIn("## TL;DR", markdown)
        self.assertIn("### Why podcast repurposing matters", markdown)
        self.assertIn("https://www.youtube.com/watch?v=abc123def45&t=0s", markdown)
        self.assertIn("## Source Timeline", markdown)

    def test_outline_sections_prefer_clean_leading_sentences(self) -> None:
        transcript = self._fixture("transcript_payload.json")
        segments = normalize_timed_content(transcript, video_id="abc123def45", source_kind="transcript", language="en")
        sections = build_outline_sections(segments, target_sections=2)
        self.assertEqual(sections[0].heading, "Why podcast repurposing matters")
        self.assertIn("Long-form podcasts contain dense ideas", sections[0].summary)
        self.assertIn("searchable, skimmable, and shareable", sections[0].summary)

    def test_outline_sections_use_all_chapters(self) -> None:
        segments = [
            Segment(
                start_sec=index * 60,
                end_sec=index * 60 + 30,
                text=f"Chapter {index} has enough transcript text to produce a useful article section summary.",
                source_kind="transcript",
                language="en",
                video_id="abc123def45",
            )
            for index in range(6)
        ]
        chapters = [{"title": f"Chapter {index}", "start_time": index * 60} for index in range(6)]

        sections = build_outline_sections(segments, target_sections=2, chapters=chapters)

        self.assertEqual([section.heading for section in sections], [f"Chapter {index}" for index in range(6)])
        self.assertEqual(sections[-1].start_sec, 300)


if __name__ == "__main__":
    unittest.main()
