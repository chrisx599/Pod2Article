from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
import time


SECRET_KEYS = {"ANTHROPIC_API_KEY", "SERPAPI_API_KEY", "api_key", "authorization", "Authorization"}
LOG_SEPARATOR = "=" * 80
LOG_SUB_SEPARATOR = "-" * 80


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pod2article_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def sanitize_for_log(value: object) -> object:
    if is_dataclass(value):
        return sanitize_for_log(asdict(value))
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in SECRET_KEYS or "key" in key_text.lower() or "token" in key_text.lower():
                sanitized[key_text] = "<set>" if item else None
            else:
                sanitized[key_text] = sanitize_for_log(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _format_event_name(event: str) -> str:
    return event.replace("_", " ").upper()


def _indent_text(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def _format_log_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(sanitize_for_log(value), ensure_ascii=False, sort_keys=True, indent=2)
    return str(value)


def format_log_event(event: str, payload: dict[str, object] | None = None) -> str:
    lines = [LOG_SEPARATOR, _format_event_name(event), LOG_SUB_SEPARATOR]
    for key, value in (payload or {}).items():
        if isinstance(value, (dict, list, tuple)):
            lines.append(f"{key}:")
            lines.append(_indent_text(_format_log_value(value)))
        else:
            lines.append(f"{key}: {_format_log_value(value)}")
    return "\n".join(lines)


def format_log_text_block(event: str, text: str) -> str:
    return "\n".join(
        [
            LOG_SEPARATOR,
            _format_event_name(event),
            LOG_SUB_SEPARATOR,
            _indent_text(text),
        ]
    )

