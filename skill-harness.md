# Media Skill Harness

This file is the development harness for the Evidence-Grounded Media Understanding Skill work. Read it after the root entry layer when the active task involves the media skill, its implementation plan, artifact schemas, wrappers, or demos.

## Source Of Truth

Use these files in order:

1. `AGENTS.md`
2. `claude-progress.md`
3. `feature_list.json`
4. `ARCHITECTURE.md` for cross-layer boundaries
5. `docs/superpowers/specs/2026-07-04-evidence-grounded-media-understanding-skill-design.md`
6. `skill-progress.md`
7. `skill-session-handoff.md`

The older `docs/superpowers/specs/2026-07-04-media-understanding-skill-suite-design.md` is a historical draft only.

## Quality Bar

Optimize for evidence reliability and output quality first, then user experience, then speed, then cost.

Do not accept a speed or cost improvement if it:

- hides uncertainty
- weakens evidence grounding
- creates unsupported claims
- caches invalid artifacts as successful results
- makes the skill look reusable while depending on unstated manual steps

## Scope Gates

Work on one feature at a time.

Phase order:

1. Skill harness and implementation plan.
2. Skill skeleton.
3. Artifact schema and validator.
4. Wrappers over existing LectureMind outputs.
5. Strong lecture demo with evidence audit.
6. Degraded-path demo.
7. Recap mode.
8. Meeting mode.
9. Live and sports highlights as separate future specs.

Stop if a task tries to add a new scene mode before artifact validation and evidence audit are stable.

## Boundary Rules

- Keep `LectureNoteIR`, `EvidenceIndex`, and `RunRecord` as internal truth.
- Treat `media_artifact` as a public projection, not a parallel truth model.
- Keep heavy media work in the LectureMind backend.
- Keep skill scripts as wrappers and validators only.
- Do not copy backend pipeline code into the skill.
- Do not make HTML, PDF, or UI pages canonical.

## Development Commands

Use the skill harness initializer before planning or implementing skill work:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\skill_harness_init.ps1
```

On Linux/macOS/WSL:

```bash
./scripts/skill_harness_init.sh
```

The initializer runs the normal project startup path and then validates the skill harness files.

For a fast harness-only check:

```powershell
D:\anaconda\envs\myagent\python.exe scripts\skill_harness_check.py
```

## Done Definition

A media skill task is done only when:

- the requested behavior is implemented or the requested plan/spec is written
- required verification has passed
- `feature_list.json` records the true task state
- `claude-progress.md` records evidence and remaining risks
- `skill-progress.md` records current phase and next direction
- `skill-session-handoff.md` can restart the next session
- the root startup path still passes

## Next Direction

The next implementation task should be Phase 1: create the `media-understanding` skill skeleton and artifact-schema reference. Do not start with backend expansion.
