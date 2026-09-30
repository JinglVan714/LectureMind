# ARCHITECTURE.md

This document describes stable system boundaries only. It does not track session state.

## Layers

- `Ingest` (`app/ingest/`)
  - Fetches and normalizes raw source material such as video metadata, subtitles, keyframes, covers, and related assets.
- `Understand` (`app/understand/`, coordinated by `app/pipeline.py`)
  - Converts source material into the project's canonical structured outputs: `Lecture Note IR` and `Evidence Index`.
- `Render` (`app/render/`)
  - Projects structured note data into reader-facing artifacts such as HTML.
- `Copilot` (`app/copilot/`)
  - Answers questions against note and evidence data, and exposes MCP-facing read surfaces.
- `Storage` (`app/storage/`, `data/`)
  - Owns persisted DB state, jobs, assets, chunks, and other runtime records.

## Dependency Direction

The stable flow is:

`Ingest` -> `Understand` -> `Render`

`Understand` -> `Copilot`

`Ingest`, `Understand`, `Render`, and `Copilot` may persist or read runtime state through `Storage`, but `Storage` does not define note semantics.

## Invariants

- `Lecture Note IR` is the canonical reader-facing note structure.
- `Evidence Index` is the canonical system-facing traceability structure.
- HTML is an output artifact from `Render`, not a source of truth.
- `Render` may format or project note data, but it must not redefine note semantics.
- `Copilot` must remain `note-first / evidence-second`; it should not treat raw HTML as its primary knowledge model.
- Cross-layer writes must respect ownership:
  - `Understand` owns semantic note/evidence construction.
  - `Render` owns exported presentation artifacts.
  - `Storage` owns persisted DB and runtime records.

## Forbidden Dependencies

- `Ingest` must not depend on `Render` or `Copilot`.
- `Understand` must not depend on rendered HTML for semantic decisions.
- `Render` must not rebuild or reinterpret business logic that belongs in `Understand`.
- `Copilot` must not bypass `Lecture Note IR` and `Evidence Index` by answering directly from rendered HTML alone.
- Modules outside `Storage` must not invent ad hoc persistence paths that bypass the storage boundary.
