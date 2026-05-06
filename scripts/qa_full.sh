#!/usr/bin/env bash
# scripts/qa_full.sh — full automated regression for LectureMind.
#
# Same gate as scripts/qa_full.ps1, for Linux / macOS / WSL / CI.
#
# Usage (from repo root):
#   ./scripts/qa_full.sh                 # plain `python` from PATH
#   CONDA_ENV=myagent ./scripts/qa_full.sh
#       → runs `conda run --no-capture-output -n $CONDA_ENV python ...`

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

step() { echo; echo "==> $*"; }

if command -v node >/dev/null 2>&1; then
    step "node --check app/static/copilot.js"
    node --check app/static/copilot.js
else
    echo "[warn] node not found; skipping copilot.js syntax check" >&2
fi

step "python -m compileall app tests scripts"
"${PYTHON_CMD[@]}" -m compileall app tests scripts

step "pytest tests/ -v"
"${PYTHON_CMD[@]}" -m pytest tests/ -v

echo
echo "OK · full automated regression passed"
