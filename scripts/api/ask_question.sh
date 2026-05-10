#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/api/ask_question.sh [--input <youtube-url|video-id|search-query>] --question <request> [options]

Options:
  --base-url  API base URL. Defaults to http://127.0.0.1:8090.
  --input     YouTube URL, video ID, or search query. Optional; omitted input triggers wide search from question.
  --question  Research request.
  --sync      Use /video-deep-research/api/tasks/sync. This is the default.
  --async     Use /video-deep-research/api/tasks and return a task id immediately.
  --raw       Print raw JSON instead of pretty JSON.
  -h, --help  Show this help.
EOF
}

BASE_URL="http://127.0.0.1:8090"
INPUT_VALUE=""
QUESTION=""
MODE="sync"
RAW="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --input)
      INPUT_VALUE="${2:-}"
      shift 2
      ;;
    --question)
      QUESTION="${2:-}"
      shift 2
      ;;
    --sync)
      MODE="sync"
      shift
      ;;
    --async)
      MODE="async"
      shift
      ;;
    --raw)
      RAW="1"
      shift
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

if [[ -z "$BASE_URL" || -z "$QUESTION" ]]; then
  echo "--base-url and --question must not be empty." >&2
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

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing curl." >&2
  exit 1
fi

BASE_URL="${BASE_URL%/}"
if [[ "$MODE" == "sync" ]]; then
  ENDPOINT="$BASE_URL/video-deep-research/api/tasks/sync"
else
  ENDPOINT="$BASE_URL/video-deep-research/api/tasks"
fi

PAYLOAD="$(
  VDR_INPUT="$INPUT_VALUE" VDR_QUESTION="$QUESTION" "$PYTHON_BIN" - <<'PY'
import json
import os

print(
    json.dumps(
        {
            **({"input": os.environ["VDR_INPUT"]} if os.environ["VDR_INPUT"] else {}),
            "question": os.environ["VDR_QUESTION"],
        },
        ensure_ascii=False,
    )
)
PY
)"

RESPONSE_FILE="$(mktemp)"
trap 'rm -f "$RESPONSE_FILE"' EXIT

HTTP_CODE=$(curl -s -w "%{http_code}" -o "$RESPONSE_FILE" \
  -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

if [[ "$RAW" == "1" ]]; then
  cat "$RESPONSE_FILE"
  printf '\n'
else
  "$PYTHON_BIN" -m json.tool "$RESPONSE_FILE"
fi

if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "202" ]]; then
  exit 1
fi
