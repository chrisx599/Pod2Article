from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from build_article import REPO_ROOT, fetch_transcript_context
else:
    from .build_article import REPO_ROOT, fetch_transcript_context


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch complete YouTube transcript context for an agent-written article."
    )
    parser.add_argument("input", help="A YouTube URL, a video ID, or a search query.")
    parser.add_argument("--output-dir", default="transcripts", help="Directory where transcript JSON should be saved.")
    parser.add_argument("--language-code", default="en", help="Preferred transcript/subtitle language code.")
    parser.add_argument(
        "--provider",
        choices=["auto", "serpapi", "oxylabs"],
        default="serpapi",
        help="API provider. SerpApi is the default; use oxylabs to force the legacy path.",
    )
    parser.add_argument(
        "--search-source",
        choices=["youtube_search", "youtube_search_max"],
        default="youtube_search",
        help="Oxylabs search source to use for query-based resolution. Ignored by SerpApi.",
    )
    parser.add_argument("--search-limit", type=int, default=5, help="Maximum number of ranked search candidates to probe.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    try:
        output_path = fetch_transcript_context(
            args.input,
            output_dir=output_dir,
            language_code=args.language_code,
            search_source=args.search_source,
            search_limit=args.search_limit,
            provider=args.provider,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
