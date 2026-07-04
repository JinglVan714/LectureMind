from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _read_json(relative_path: str) -> dict:
    return json.loads(_read_text(relative_path))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        "skill-harness.md",
        "skill-progress.md",
        "skill-session-handoff.md",
        "docs/superpowers/specs/2026-07-04-evidence-grounded-media-understanding-skill-design.md",
        "scripts/skill_harness_init.ps1",
        "scripts/skill_harness_init.sh",
    ]
    for relative_path in required_files:
        if not (ROOT / relative_path).exists():
            errors.append(f"missing required skill harness file: {relative_path}")

    if not errors:
        harness = _read_text("skill-harness.md")
        progress = _read_text("skill-progress.md")
        handoff = _read_text("skill-session-handoff.md")
        spec = _read_text(
            "docs/superpowers/specs/2026-07-04-evidence-grounded-media-understanding-skill-design.md"
        )

        harness_must_mention = [
            "Evidence-Grounded Media Understanding Skill",
            "feature_list.json",
            "skill-progress.md",
            "skill-session-handoff.md",
            "media_artifact",
            "Do not copy backend pipeline code",
        ]
        for phrase in harness_must_mention:
            if phrase not in harness:
                errors.append(f"skill-harness.md must mention: {phrase}")

        progress_must_mention = [
            "Current Phase",
            "Next Work",
            "Verification Evidence",
            "ffmpeg",
            "yt-dlp",
        ]
        for phrase in progress_must_mention:
            if phrase not in progress:
                errors.append(f"skill-progress.md must mention: {phrase}")

        handoff_must_mention = [
            "Current Objective",
            "Verification Evidence",
            "Reusable Handoff Prompt",
            "skill_harness_init.ps1",
        ]
        for phrase in handoff_must_mention:
            if phrase not in handoff:
                errors.append(f"skill-session-handoff.md must mention: {phrase}")

        spec_must_mention = [
            "Follow-Up Directions To Preserve",
            "Quality",
            "media_artifact",
            "Do not claim Tencent Meeting",
        ]
        for phrase in spec_must_mention:
            if phrase not in spec:
                errors.append(f"replacement spec must mention: {phrase}")

    try:
        feature_list = _read_json("feature_list.json")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"feature_list.json is not valid JSON: {exc}")
    else:
        features = feature_list.get("features")
        if not isinstance(features, list):
            errors.append("feature_list.json must contain a top-level features list")
        else:
            in_progress = [item.get("id") for item in features if item.get("status") == "in_progress"]
            if len(in_progress) > 1:
                errors.append(f"feature_list.json has multiple in_progress features: {in_progress}")
            if not any(item.get("id") == "skill-harness-001" for item in features):
                errors.append("feature_list.json must include skill-harness-001")

    media_skill = ROOT / "media-understanding"
    if media_skill.exists():
        expected = [
            media_skill / "SKILL.md",
            media_skill / "references" / "artifact_schema.md",
        ]
        for path in expected:
            if not path.exists():
                errors.append(f"existing media-understanding skill is missing: {path.relative_to(ROOT)}")
    else:
        warnings.append("media-understanding/ does not exist yet; this is expected before Phase 1 implementation.")

    for script in ["init.ps1", "init.sh"]:
        if (ROOT / script).exists() and "test_skill_harness.py" not in _read_text(script):
            errors.append(f"{script} must run tests/test_skill_harness.py")

    if errors:
        print("skill harness check failed:")
        for error in errors:
            print(f"- {error}")
        if warnings:
            print("warnings:")
            for warning in warnings:
                print(f"- {warning}")
        return 1

    print("skill harness check passed")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
