from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from normalize import normalize_timed_content  # noqa: E402
from serpapi_client import SerpApiClient  # noqa: E402
from utils import load_local_env, parse_serpapi_key  # noqa: E402
from youtube_sources import parse_metadata  # noqa: E402


def build_client() -> SerpApiClient:
    return SerpApiClient(parse_serpapi_key(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache raw and normalized transcript payloads.")
    parser.add_argument("video_id")
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--output-dir", default="benchmark-results/transcript-cache")
    args = parser.parse_args()

    load_local_env(REPO_ROOT)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()
    probe = client.fetch_best_timed_content(args.video_id, language_code=args.language_code)
    metadata = parse_metadata(probe.metadata, args.video_id)
    segments = normalize_timed_content(
        probe.content_payload,
        video_id=args.video_id,
        source_kind=probe.source_kind,
        language=metadata["language"],
    )

    raw_path = raw_dir / f"serpapi-{args.video_id}.json"
    normalized_path = normalized_dir / f"serpapi-{args.video_id}.json"
    raw_path.write_text(
        json.dumps(
            {
                "provider": "serpapi",
                "video_id": args.video_id,
                "source_kind": probe.source_kind,
                "origin": probe.origin,
                "metadata": probe.metadata,
                "content_payload": probe.content_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
        json.dumps(
            {
                "provider": "serpapi",
                "video_id": args.video_id,
                "source_kind": probe.source_kind,
                "origin": probe.origin,
                "metadata": metadata,
                "segments": [asdict(segment) for segment in segments],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs = {"serpapi": {"raw": str(raw_path), "normalized": str(normalized_path)}}

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
