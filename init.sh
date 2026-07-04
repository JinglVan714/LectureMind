#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/.tmp/pytest"
export TMPDIR="$ROOT/.tmp/pytest"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

PYTHON_BIN=""
for candidate in python python3 /mnt/d/anaconda/envs/myagent/python.exe; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c "import fastapi, pytest" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  elif [ -x "$candidate" ]; then
    if "$candidate" -c "import fastapi, pytest" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "No suitable Python interpreter with fastapi and pytest was found." >&2
  exit 1
fi

echo "==> LectureMind agent init"
echo "==> repo root: $ROOT"
echo "==> temp root: $TMPDIR"
echo "==> python executable: $PYTHON_BIN"
echo "==> python version: $("$PYTHON_BIN" --version 2>&1)"

if [ ! -f ".env" ] && [ ! -f ".env.example" ]; then
  echo "Missing both .env and .env.example" >&2
  exit 1
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "[warn] .env is missing; copy .env.example before running real model-backed flows."
fi

command -v ffmpeg >/dev/null 2>&1 || echo "[warn] ffmpeg not found on PATH"
command -v yt-dlp >/dev/null 2>&1 || echo "[warn] yt-dlp not found on PATH"

echo "==> python -m compileall app tests scripts"
"$PYTHON_BIN" -m compileall app tests scripts

echo "==> python -m pytest tests/test_harness_workspace.py -q"
"$PYTHON_BIN" -m pytest tests/test_harness_workspace.py -q

echo "==> python -m pytest tests/test_skill_harness.py -q"
"$PYTHON_BIN" -m pytest tests/test_skill_harness.py -q

echo "==> python -m pytest tests/test_smoke.py -q"
"$PYTHON_BIN" -m pytest tests/test_smoke.py -q

echo "==> init complete"
