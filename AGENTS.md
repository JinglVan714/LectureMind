# AGENTS.md

This repository expects a coding agent to cold-start from a small, stable entry layer instead of long handoff notes.

## Start Here

1. Read `claude-progress.md` for the current verified state, active task, and known blockers.
2. Read `feature_list.json` for the single source of truth on task status and required verification.
3. Run `init.ps1` on Windows or `init.sh` on Linux/macOS/WSL.
4. If `feature_list.json` contains one `in_progress` item, pick that single active task. If none are active, choose the highest-priority non-passing item or stop and ask for the next task.
5. Work only within the chosen task until it is verified, explicitly marked blocked, or deliberately handed off.
6. For evidence-grounded media skill tasks, read `skill-harness.md` after the active task is selected.

## Working Rules

- Single-task constraint: only one task or feature may be active at a time.
- Do not widen scope unless the current blocker requires it.
- Treat repository files as the source of truth, not chat history.
- Adapt to parallel changes from other agents; do not overwrite unrelated edits.
- Use deeper docs only when the entry layer is insufficient:
  - `CLAUDE.md` for quick project pointers
  - `ARCHITECTURE.md` for stable boundaries and invariants
  - `skill-harness.md` for media skill planning, implementation, artifact, and demo work

## Validation

- Do not start from a broken baseline without recording the blocker.
- Run the required verification listed on the active item in `feature_list.json`.
- At minimum, re-run `init.ps1` or `init.sh` when your changes affect the standard startup path.
- Do not claim completion without recorded verification evidence.

## Done Definition

A task is done only when all of the following are true:

- The requested behavior is implemented.
- Required verification has been executed successfully.
- `claude-progress.md` is updated with the current status, evidence, and any remaining risks.
- `feature_list.json` is updated so the next agent can see the true state immediately.
- The repository can still be re-entered through `init.ps1` or `init.sh`.
