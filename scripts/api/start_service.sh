#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/api/start_service.sh [options]

Examples:
  scripts/api/start_service.sh
  scripts/api/start_service.sh --background

Options:
  --host        Host to bind. Defaults to 127.0.0.1.
  --port        Port to bind. Defaults to 8090.
  --task-root   Runtime task directory. Defaults to output/api.
  --log-file    Background log path. Defaults to output/api/server.log.
  --background  Start with nohup and print the process id.
  -h, --help    Show this help.
EOF
}

HOST="127.0.0.1"
PORT="8090"
TASK_ROOT="output/api"
LOG_FILE=""
BACKGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --task-root)
      TASK_ROOT="${2:-}"
      shift 2
      ;;
    --log-file)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    --background)
      BACKGROUND="1"
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

if [[ -z "$HOST" || -z "$PORT" || -z "$TASK_ROOT" ]]; then
  echo "--host, --port, and --task-root must not be empty." >&2
  exit 2
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "--port must be an integer." >&2
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

mkdir -p "$TASK_ROOT"

ARGS=(
  "-m" "api.video_deep_research_api"
  "--host" "$HOST"
  "--port" "$PORT"
  "--task-root" "$TASK_ROOT"
)

if [[ "$BACKGROUND" == "1" ]]; then
  if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$TASK_ROOT/server.log"
  fi
  mkdir -p "$(dirname "$LOG_FILE")"
  nohup "$PYTHON_BIN" "${ARGS[@]}" >"$LOG_FILE" 2>&1 &
  PID="$!"
  echo "Video Deep Research API starting at http://$HOST:$PORT"
  echo "pid=$PID"
  echo "log=$LOG_FILE"
  exit 0
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
