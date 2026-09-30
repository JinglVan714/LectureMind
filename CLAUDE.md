# CLAUDE.md

## Project Overview

LectureMind turns a long-form video into a readable lecture note, a traceable evidence graph, and a note-centered Copilot experience.

## Technical Stack

- Python 3.11
- FastAPI
- Pydantic v2
- SQLite
- pytest
- PowerShell + shell startup scripts

## Quick Start

1. Read `AGENTS.md`.
2. Read `claude-progress.md`.
3. Read `feature_list.json`.
4. Run `init.ps1` or `init.sh`.
5. Open `ARCHITECTURE.md` before changing cross-layer behavior.
6. Use `scripts/preflight.ps1` or `scripts/qa_full.ps1` only when the active task calls for broader verification.

## Key Files

- `AGENTS.md`: startup order, working rules, done definition.
- `ARCHITECTURE.md`: stable layer boundaries and forbidden dependencies.
- `claude-progress.md`: current verified status, active task, blockers.
- `feature_list.json`: machine-readable task list and verification contract.
- `init.ps1` / `init.sh`: standard startup checks.
- `app/pipeline.py`: end-to-end orchestration entry.
- `app/understand/ir_builder.py`: builds `Lecture Note IR`.
- `app/understand/evidence_index.py`: builds `Evidence Index`.
- `app/render/renderer.py`: renders note artifacts.
- `app/copilot/`: note-centered retrieval, answering, and MCP surface.
- `app/storage/db.py`: persistence boundary.
- `docs/superpowers/specs/2026-06-09-lecturemind-agent-workspace-harness-design.md`: workspace harness design source of truth.

## Validation Commands

- `D:\anaconda\envs\myagent\python.exe -m pytest tests/test_harness_workspace.py -v`
- `powershell -ExecutionPolicy Bypass -File .\init.ps1`
- `wsl.exe bash -lc 'cd /mnt/d/Diet_Agent_NEW && ./init.sh'`
- `scripts/preflight.ps1 -SkipTests`

## Architecture Rules

- `Lecture Note IR` is the reader-facing structured source of truth.
- `Evidence Index` is the system-facing traceability source of truth.
- HTML 不是结构真源; it is a render projection, not the canonical data model.
- Copilot is `note-first / evidence-second`, not raw-HTML-first.
- New behavior should preserve the boundaries in `ARCHITECTURE.md` instead of re-encoding logic in UI or render code.
