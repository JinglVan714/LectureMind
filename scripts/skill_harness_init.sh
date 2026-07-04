#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/.tmp/pytest"
export TMPDIR="$ROOT/.tmp/pytest"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "==> Media skill harness init"
echo "==> temp root: $TMPDIR"

./init.sh

PYTHON_BIN=""
for candidate in python python3 /mnt/d/anaconda/envs/myagent/python.exe; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  elif [ -x "$candidate" ]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "No Python interpreter found for skill harness check." >&2
  exit 1
fi

echo "==> python scripts/skill_harness_check.py"
"$PYTHON_BIN" scripts/skill_harness_check.py

echo "==> media skill harness init complete"
