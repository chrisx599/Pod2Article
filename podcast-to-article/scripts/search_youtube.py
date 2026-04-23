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
    from youtube_sources import REPO_ROOT, search_youtube_context
else:
    from .youtube_sources import REPO_ROOT, search_youtube_context


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search YouTube and save normalized candidate results.")
    parser.add_argument("query", help="YouTube search query.")
    parser.add_argument("--output-dir", default="search-results", help="Directory where search JSON should be saved.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    try:
        output_path = search_youtube_context(
            args.query,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "ok", "output_path": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
