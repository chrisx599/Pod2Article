#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/agent/run_podcast_article_agent.sh [--input <youtube-url|video-id|search-query>] --question <request> [--mode auto|deep|wide] [--output-dir <dir>] [--log-file <path>]

Examples:
  scripts/agent/run_podcast_article_agent.sh \
    --input "https://www.youtube.com/watch?v=hmtuvNfytjM" \
    --question "请写一篇关于这期访谈核心观点的深度文章"

Options:
  --input       YouTube URL, video ID, or search query. Optional in wide search.
  --question    Research request.
  --mode        auto, deep, or wide. Defaults to auto.
  --output-dir  Output root. Defaults to output/agent.
  --log-file    Optional explicit log file path.
  -h, --help    Show this help.
EOF
}

INPUT_VALUE=""
QUESTION=""
MODE="auto"
OUTPUT_DIR="output/agent"
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_VALUE="${2:-}"
      shift 2
      ;;
    --question)
      QUESTION="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --log-file)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$QUESTION" ]]; then
  echo "--question must not be empty." >&2
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python3. Activate a Python environment or set PYTHON_BIN." >&2
  exit 1
fi

ARGS=(
  "-m" "agents.podcast_article_agent"
  "--input" "$INPUT_VALUE"
  "--question" "$QUESTION"
  "--mode" "$MODE"
  "--output-dir" "$OUTPUT_DIR"
)

if [[ -n "$LOG_FILE" ]]; then
  ARGS+=("--log-file" "$LOG_FILE")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
