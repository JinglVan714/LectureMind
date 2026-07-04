from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_skill_harness_check_passes():
    result = subprocess.run(
        [sys.executable, "scripts/skill_harness_check.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skill harness check passed" in result.stdout


def test_skill_harness_files_define_restart_path():
    harness = _read_text("skill-harness.md")
    progress = _read_text("skill-progress.md")
    handoff = _read_text("skill-session-handoff.md")

    assert "skill_harness_init.ps1" in harness
    assert "Next Work" in progress
    assert "Reusable Handoff Prompt" in handoff
    assert "Do not copy backend pipeline code" in harness


def test_feature_list_tracks_skill_harness_item():
    data = json.loads(_read_text("feature_list.json"))
    features = data["features"]

    matches = [feature for feature in features if feature["id"] == "skill-harness-001"]
    assert len(matches) == 1
    assert matches[0]["status"] in {"in_progress", "passing"}
    assert matches[0]["verification"]
    assert matches[0]["evidence"]


def test_init_scripts_run_skill_harness_tests_and_local_temp():
    ps1 = _read_text("init.ps1")
    sh = _read_text("init.sh")

    assert "tests/test_skill_harness.py" in ps1
    assert "tests/test_skill_harness.py" in sh
    assert ".tmp\\pytest" in ps1
    assert ".tmp/pytest" in sh
