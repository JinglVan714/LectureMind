from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _required_path(relative_path: str) -> Path:
    path = ROOT / relative_path
    assert path.exists(), f"Expected workspace harness file at repo root: {relative_path}"
    return path


def _read_required_text(relative_path: str) -> str:
    return _required_path(relative_path).read_text(encoding="utf-8")


def _read_required_json(relative_path: str) -> dict:
    data = json.loads(_read_required_text(relative_path))
    assert isinstance(data, dict), f"{relative_path} must contain a top-level JSON object"
    return data


def _count_top_level_tests(relative_path: str) -> int:
    module = ast.parse(_read_required_text(relative_path), filename=relative_path)
    return sum(
        1
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def test_workspace_harness_core_files_exist():
    required = [
        "AGENTS.md",
        "CLAUDE.md",
        "ARCHITECTURE.md",
        "feature_list.json",
        "claude-progress.md",
        "clean-state-checklist.md",
        "init.ps1",
        "init.sh",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert missing == [], f"Missing workspace harness files: {missing}"


def test_feature_list_has_single_in_progress_rule():
    data = _read_required_json("feature_list.json")

    assert "features" in data, "feature_list.json must expose a top-level 'features' array"
    assert isinstance(data["features"], list), "feature_list.json 'features' must be a list"

    in_progress = []
    for index, feature in enumerate(data["features"]):
        assert isinstance(feature, dict), f"features[{index}] must be a JSON object"
        assert "id" in feature, f"features[{index}] must define an 'id'"
        assert "status" in feature, f"features[{index}] must define a 'status'"
        if feature["status"] == "in_progress":
            in_progress.append(feature["id"])

    assert len(in_progress) <= 1, (
        "feature_list.json allows at most one in_progress feature; "
        f"found {len(in_progress)}: {in_progress}"
    )


def test_passing_feature_items_record_evidence():
    data = _read_required_json("feature_list.json")
    count = _count_top_level_tests("tests/test_harness_workspace.py")
    progress_body = _read_required_text("claude-progress.md")

    for index, feature in enumerate(data["features"]):
        if feature.get("status") == "passing":
            evidence = feature.get("evidence")
            assert isinstance(evidence, list) and evidence, (
                f"features[{index}] is passing but does not record verification evidence"
            )
            if feature.get("id") == "harness-001":
                joined_evidence = "\n".join(evidence)
                assert (
                    f"tests/test_harness_workspace.py -v -> {count} passed." in joined_evidence
                ), "feature_list.json must record the current harness test count in verification evidence"
                assert re.search(
                    rf"tests/test_harness_workspace\.py -v`[^\n]*\b{count}/{count}\b",
                    progress_body,
                ), "claude-progress.md baseline section must record the current harness test count"
                assert re.search(
                    rf"`tests/test_harness_workspace\.py`[^\n]*\b{count}/{count}\b",
                    progress_body,
                ), "claude-progress.md recent-completed section must record the current harness test count"


def test_agents_md_links_progress_and_feature_sources():
    body = _read_required_text("AGENTS.md")

    assert "claude-progress.md" in body, "AGENTS.md must point agents to claude-progress.md"
    assert "feature_list.json" in body, "AGENTS.md must point agents to feature_list.json"
    assert "init.ps1" in body or "init.sh" in body, "AGENTS.md must point agents to init scripts"


def test_claude_md_contains_quick_reference_sections():
    body = _read_required_text("CLAUDE.md")

    assert "## Technical Stack" in body, "CLAUDE.md must include a Technical Stack section"
    assert "## Validation Commands" in body, "CLAUDE.md must include a Validation Commands section"


def test_init_scripts_do_not_boot_uvicorn():
    ps1 = _read_required_text("init.ps1")
    sh = _read_required_text("init.sh")

    assert "uvicorn" not in ps1.lower(), "init.ps1 must not boot uvicorn or any web server"
    assert "uvicorn" not in sh.lower(), "init.sh must not boot uvicorn or any web server"
    assert "python3" in sh, "init.sh must support python3-based Unix environments"
    assert "tests/test_harness_workspace.py" in ps1, "init.ps1 must validate the workspace harness contract"
    assert "tests/test_harness_workspace.py" in sh, "init.sh must validate the workspace harness contract"


def test_progress_file_declares_single_active_task_section():
    body = _read_required_text("claude-progress.md")

    active_task_headings = re.findall(r"^##\s+.*Active Task\s*$", body, flags=re.MULTILINE)
    assert len(active_task_headings) == 1, (
        "claude-progress.md must declare exactly one markdown section for the active task; "
        f"found {len(active_task_headings)} headings"
    )

    data = _read_required_json("feature_list.json")
    in_progress = [feature["id"] for feature in data["features"] if feature.get("status") == "in_progress"]
    assert len(in_progress) <= 1, (
        "feature_list.json allows at most one in_progress feature; "
        f"found {len(in_progress)}"
    )
    if in_progress:
        assert in_progress[0] in body, (
            "claude-progress.md active task section must name the in_progress feature id"
        )
    else:
        assert "没有新的 active implementation task" in body or "No active implementation task" in body, (
            "claude-progress.md must explicitly state when no active implementation task is selected"
        )


def test_architecture_doc_declares_layer_boundaries():
    body = _read_required_text("ARCHITECTURE.md")

    assert "## Layers" in body, "ARCHITECTURE.md must define a Layers section"
    assert "Ingest" in body, "ARCHITECTURE.md must mention the Ingest layer"
    assert "Understand" in body, "ARCHITECTURE.md must mention the Understand layer"
    assert "Render" in body, "ARCHITECTURE.md must mention the Render layer"
    assert "Copilot" in body, "ARCHITECTURE.md must mention the Copilot layer"
    assert "Storage" in body, "ARCHITECTURE.md must mention the Storage layer"
    assert "Forbidden Dependencies" in body, (
        "ARCHITECTURE.md must define forbidden cross-layer dependencies"
    )


def test_readme_mentions_agent_quick_start():
    body = _read_required_text("README.md")

    assert "Agent Quick Start" in body, "README.md must include an Agent Quick Start section"
    assert "AGENTS.md" in body, "README.md Agent Quick Start must point to AGENTS.md"
    assert "claude-progress.md" in body, "README.md Agent Quick Start must point to claude-progress.md"
    assert "feature_list.json" in body, "README.md Agent Quick Start must point to feature_list.json"
    assert "init.ps1" in body or "init.sh" in body, "README.md Agent Quick Start must mention init scripts"


def test_gitignore_keeps_docs_private_by_default():
    body = _read_required_text(".gitignore")

    assert "/docs/*" in body, ".gitignore must keep deep docs private by default"
    assert "!/docs/deploy.md" in body, ".gitignore must keep the deploy doc as the narrow allowlist"
