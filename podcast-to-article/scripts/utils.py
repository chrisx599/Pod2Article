from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import parse_qs, urlparse

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def load_local_env(start_path: Optional[Path] = None) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    start = Path(start_path or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        env_path = candidate / ".env"
        if not env_path.exists():
            continue
        values: Dict[str, str] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
            if key:
                values[key] = value
        return values
    return {}


def parse_credentials() -> Tuple[str, str]:
    username = os.environ.get("OXYLABS_USERNAME")
    password = os.environ.get("OXYLABS_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Missing Oxylabs credentials. Set OXYLABS_USERNAME and "
            "OXYLABS_PASSWORD or provide them in a local .env file."
        )
    return username, password


def parse_serpapi_key() -> str:
    api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise ValueError(
            "Missing SerpApi credentials. Set SERPAPI_API_KEY or provide it in a local .env file."
        )
    return api_key


def detect_input_type(raw_input: str) -> str:
    value = raw_input.strip()
    if not value:
        raise ValueError("Input cannot be empty.")
    if extract_video_id(value):
        return "youtube_url" if value.startswith(("http://", "https://")) else "video_id"
    return "search_query"


def extract_video_id(value: str) -> Optional[str]:
    candidate = value.strip()
    if VIDEO_ID_RE.fullmatch(candidate):
        return candidate
    if not candidate.startswith(("http://", "https://")):
        return None
    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        tail = parsed.path.strip("/").split("/", 1)[0]
        return tail if VIDEO_ID_RE.fullmatch(tail) else None
    if "youtube.com" in host:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
            return video_id if video_id and VIDEO_ID_RE.fullmatch(video_id) else None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return parts[1] if VIDEO_ID_RE.fullmatch(parts[1]) else None
    return None


def slugify(value: str, fallback: str = "article") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def format_timestamp(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_youtube_timestamp_url(video_id: str, seconds: int) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={max(int(seconds), 0)}s"


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z']+", text.lower()) if token not in STOPWORDS]


def keyword_frequencies(texts: Iterable[str]) -> Dict[str, int]:
    frequencies: Dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies
