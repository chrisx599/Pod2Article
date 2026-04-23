from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
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


CONFIG_FILENAMES = ("pod2article.config", ".pod2article.config", "config.env")


def _parse_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _find_upwards(start_path: Path, filenames: Sequence[str]) -> Optional[Path]:
    start = Path(start_path).resolve()
    if start.is_file():
        start = start.parent
    for candidate_dir in [start, *start.parents]:
        for filename in filenames:
            path = candidate_dir / filename
            if path.exists():
                return path
    return None


def load_local_env(start_path: Optional[Path] = None) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    env_path = _find_upwards(Path(start_path or Path.cwd()), (".env",))
    if env_path is None:
        return {}
    values = _parse_key_value_file(env_path)
    for key, value in values.items():
        if key not in os.environ:
            os.environ[key] = value
    return values


def load_config_file(start_path: Optional[Path] = None) -> Dict[str, str]:
    """Read project config without mutating process environment."""
    config_path = _find_upwards(Path(start_path or Path.cwd()), CONFIG_FILENAMES)
    if config_path is None:
        return {}
    return _parse_key_value_file(config_path)


def resolve_setting(
    keys: Sequence[str],
    *,
    start_path: Optional[Path] = None,
    local_env: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a setting from config, system env, then local .env."""
    config_values = load_config_file(start_path)
    for key in keys:
        value = config_values.get(key)
        if value:
            return value
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    env_values = local_env if local_env is not None else load_local_env(start_path)
    for key in keys:
        value = env_values.get(key)
        if value:
            return value
    return None


def parse_serpapi_key(start_path: Optional[Path] = None) -> str:
    env_values = load_local_env(start_path)
    api_key = resolve_setting(("SERPAPI_API_KEY", "SERPAPI_KEY"), start_path=start_path, local_env=env_values)
    if not api_key:
        raise ValueError(
            "Missing SerpApi credentials. Set SERPAPI_API_KEY in pod2article.config, "
            "the system environment, or a local .env file."
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
