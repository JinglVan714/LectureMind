#!/usr/bin/env bash
# scripts/dev_up.sh — one-shot local dev launcher (Linux / macOS / WSL).
#
# Steps:
#   1. ensure .env exists (auto-copy from .env.example with a warning)
#   2. mkdir data/ + run scripts/init_db.py to materialise the SQLite schema
#   3. start uvicorn on 0.0.0.0:8000 (override via APP_HOST/APP_PORT in .env)
#
# Usage:
#   ./scripts/dev_up.sh                       # plain `python` from PATH
#   CONDA_ENV=myagent ./scripts/dev_up.sh     # use conda env
#   RELOAD=1 ./scripts/dev_up.sh              # uvicorn --reload

set -euo pipefail

# Reject native Windows shells (PowerShell / cmd) where this bash script
# either silently no-ops or partially executes through Git Bash with
# unexpected pathing.  Send the user to the PowerShell launcher instead.
case "${OS:-}${MSYSTEM:-}${WT_SESSION:-}" in
    Windows_NT*|MINGW*|MSYS*)
        if [[ -z "${BASH_VERSION:-}" || "${TERM_PROGRAM:-}" == "PowerShell" ]]; then
            echo "[dev_up.sh] Windows / PowerShell detected — please run scripts/dev_up.ps1 instead." >&2
            echo "    pwsh> ./scripts/dev_up.ps1               # uses conda env 'myagent' by default" >&2
            echo "    pwsh> ./scripts/dev_up.ps1 -NoConda      # uses python on PATH" >&2
            exit 2
        fi
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "[bootstrap] .env created from .env.example — fill DASHSCOPE_API_KEY etc. before real runs" >&2
    else
        echo "[error] neither .env nor .env.example exists" >&2
        exit 1
    fi
fi

mkdir -p data

PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

echo "==> python -m scripts.init_db"
"${PYTHON_CMD[@]}" -m scripts.init_db

UVICORN_ARGS=(-m uvicorn app.main:app --host 0.0.0.0 --port 8000)
if [[ "${RELOAD:-0}" == "1" ]]; then
    UVICORN_ARGS+=(--reload)
fi

echo "==> uvicorn app.main:app  (host/port from .env, override via APP_HOST/APP_PORT)"
exec "${PYTHON_CMD[@]}" "${UVICORN_ARGS[@]}"
