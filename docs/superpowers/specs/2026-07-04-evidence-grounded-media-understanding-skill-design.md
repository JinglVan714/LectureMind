# Evidence-Grounded Media Understanding Skill Design

Date: 2026-07-04

## Status

Approved replacement direction. This specification supersedes `docs/superpowers/specs/2026-07-04-media-understanding-skill-suite-design.md`.

This document is a product and skill architecture specification only. It does not claim that the new skill package, generic media API, meeting mode, sports highlight mode, or live ingestion mode already exist.

## Decision

Keep the product direction, but rewrite the specification and boundary.

The project should not become a generic media platform or a renamed lecture library. The durable product should be:

> Evidence-Grounded Media Understanding Skill: an agent-callable capability that turns long media into structured summaries, timestamped evidence, and reusable artifacts, backed by LectureMind's existing long-video pipeline and runtime observability.

The first credible version must optimize for:

1. Evidence reliability and output quality.
2. User experience for the calling agent and downstream reader.
3. Execution speed.
4. Cost.

Cost and latency optimizations are invalid if they hide uncertainty, weaken evidence grounding, or produce artifacts that cannot be audited.

## Why This Replaces The Earlier Spec

The earlier `Media Understanding Skill Suite` draft had the right instinct: move away from a library-like product shell and expose the valuable media understanding ability as agent-callable skills.

It also had three structural problems:

1. It introduced a broad "suite" before one vertical capability was proven.
2. It risked creating parallel truth models beside existing `LectureNoteIR`, `EvidenceIndex`, and `RunRecord`.
3. It described backend pipeline responsibilities as if they belonged inside a skill.

This replacement keeps the repositioning but narrows the engineering shape:

- start with one production-grade path for long-form learning/recap media
- expose a small set of agent tools
- make structured artifacts the reusable contract
- keep heavy media processing in the service/backend
- treat meeting, live, sports, and short-form highlights as scene profiles or future specializations, not as all-at-once v1 promises

## Repository Reality

### Existing Capability

LectureMind already has reusable assets worth preserving:

- `Ingest -> Understand -> Render` layering.
- Bilibili-oriented ingestion for metadata, subtitles, covers, keyframes, and cookies.
- Subtitle-first processing with ASR fallback paths.
- Long-video segmentation, chapter planning, cache-aware map-reduce behavior, VLM tiering, and VLM cache concepts.
- `LectureNoteIR` as reader-facing structured truth.
- `EvidenceIndex` as system-facing traceability truth.
- HTML rendering as a projection, not the canonical data model.
- Copilot/RAG/MCP read surfaces over note and evidence data.
- A default-hidden `summarize_video` MCP write tool.
- Runtime harness records with `trace_id`, `RunContract`, `PolicySnapshot`, emitted artifacts, and `RunVerdict`.

### Current Gaps

The current project is not yet a generic media skill:

- Ingestion remains Bilibili-first.
- `ffmpeg` and `yt-dlp` were not found on PATH during the latest `init.ps1` run, so real video demos must address environment readiness.
- Meeting, live, sports, and highlight extraction do not have production-grade contracts or evaluation yet.
- Existing HTTP endpoints are lecture/job oriented, not a stable `/media/jobs` contract.
- UI and lecture-library residue can still make the project look like a personal knowledge app.

### Verification Baseline

`powershell -ExecutionPolicy Bypass -File .\init.ps1` initially failed because pytest could not create directories under the default Windows Temp path. With `TEMP` and `TMP` pointed to `.tmp/pytest` inside the repository, the startup path passed:

- `compileall app tests scripts` passed
- `tests/test_harness_workspace.py`: 10 passed
- `tests/test_smoke.py`: 107 passed

Warnings remained:

- `ffmpeg not found on PATH`
- `yt-dlp not found on PATH`

Those warnings must be treated as skill-readiness risks, not ignored.

## Product Definition

The skill helps an upstream agent or business workflow answer:

- What happened in this long media item?
- Which claims are supported by which timestamps, transcript spans, or frames?
- What structured artifact can I reuse in a report, Q&A flow, recap, or later analysis?
- What degraded, failed, or relied on weaker evidence?

The product is not a standalone library UI. The first screen or calling surface should be a tool/API invocation and its artifact result, not a lecture collection dashboard.

## Non-Goals

Do not build these in the first implementation slice:

- a new generic agent framework
- a new all-purpose media platform
- a new canonical data model that replaces `LectureNoteIR` and `EvidenceIndex`
- a full multi-tenant SaaS permission model
- real-time live stream ingestion
- production sports highlight detection
- production meeting diarization and speaker-attributed minutes
- LaTeX/PDF as the canonical output
- a large frontend product shell

## Skill Boundary

The skill is a thin, reusable agent-facing package. It should contain:

- concise instructions
- tool/API calling rules
- validation scripts
- artifact schemas
- failure/degradation policies
- quality rubrics
- demo artifacts

The skill must not contain:

- video download implementation
- ASR implementation
- VLM orchestration
- cache implementation
- database ownership
- RAG index implementation
- job scheduler implementation
- a copied version of LectureMind's backend

Heavy media work belongs in the LectureMind service/backend. The skill should teach an agent how to call that service, validate the returned artifacts, and reuse evidence safely.

## Architecture

```text
Agent / Business Caller
        |
        v
media-understanding skill
  - short SKILL.md
  - wrappers
  - schemas
  - quality and failure policies
        |
        v
Tool/API Adapter Layer
  - MCP tools for agents
  - HTTP job adapter for services
  - structured outputs and artifact refs
        |
        v
LectureMind Media Understanding Service
  - ingest
  - transcript and ASR fallback
  - segmentation
  - frame sampling and VLM tiering
  - summary compilation
  - evidence indexing
  - runtime harness
        |
        v
Artifact Store
  - media artifacts
  - evidence index projections
  - frame and asset refs
  - run records
        |
        v
Evaluation Harness
  - schema validation
  - evidence audit
  - degraded replay
  - scenario benchmarks
```

Boundary rule:

- `LectureNoteIR` and `EvidenceIndex` remain the current internal truth.
- `media_artifact` is an adapter/projection for agent reuse.
- `RunRecord` remains the runtime observability truth.
- HTML/PDF remain render projections.

## Public Tool Surface

The v1 tool surface should be small. Tool names should describe tasks, not internal pipeline stages.

### `summarize_media`

Purpose: submit or run a media analysis job and return an artifact reference.

Inputs:

```json
{
  "source": {
    "type": "url | local_file | transcript_ref | artifact_ref",
    "value": "string",
    "platform_hint": "bilibili | youtube | meeting | unknown"
  },
  "scene_mode": "lecture | recap",
  "quality_policy": "standard | high_quality",
  "reuse_policy": "reuse_valid | force_refresh",
  "async": true
}
```

Output:

```json
{
  "status": "queued | running | succeeded | degraded | failed",
  "job_ref": "string",
  "artifact_ref": "string | null",
  "summary_preview": "string | null",
  "degradations": [],
  "run_record_ref": "string | null"
}
```

Rules:

- Default to async for real video and ASR-heavy runs.
- Return `artifact_ref` only when a schema-valid artifact exists.
- Surface degraded status when subtitles, frames, VLM, or ASR are missing or weakened.
- Do not hide environment failures such as missing `ffmpeg` or `yt-dlp`.

### `get_media_artifact`

Purpose: retrieve a structured artifact for reuse.

Inputs:

```json
{
  "artifact_ref": "string",
  "projection": "full | summary | segments | evidence_outline",
  "include_asset_refs": true
}
```

Output:

```json
{
  "media_artifact": {},
  "run_record_ref": "string | null"
}
```

Rules:

- This is the preferred follow-up tool after `summarize_media`.
- Callers should not reprocess media when an artifact is already valid.

### `locate_evidence`

Purpose: find timestamped support for a question, claim, topic, or summary paragraph.

Inputs:

```json
{
  "artifact_ref": "string",
  "query": "string",
  "claim": "string | null",
  "filters": {
    "modalities": ["transcript", "frame", "summary_block"],
    "time_range": {"start_ms": 0, "end_ms": null}
  },
  "top_k": 5
}
```

Output:

```json
{
  "evidence_pack": {},
  "coverage_notes": [],
  "run_record_ref": "string | null"
}
```

Rules:

- Evidence results must include timestamp anchors when available.
- Visual evidence must identify frame references and the selection reason.
- Low-confidence or partial matches must be marked as such.

### `extract_highlights`

Status: experimental and hidden by default.

Purpose: produce highlight candidates for recap or reuse.

Inputs:

```json
{
  "artifact_ref": "string",
  "scene_mode": "recap | lecture",
  "strategy": "summary_salience | visual_change | transcript_density | mixed",
  "max_candidates": 10
}
```

Output:

```json
{
  "highlight_candidates": {},
  "experimental": true,
  "limitations": [],
  "run_record_ref": "string | null"
}
```

Rules:

- Do not market this as sports/live highlight extraction until it has dedicated signals and evaluation.
- Every candidate must include a reason label and supporting evidence refs.

## HTTP Surface

Short term: add a thin adapter over the existing app and job model only when it does not duplicate current behavior.

Long term target:

- `POST /media/jobs`
- `GET /media/jobs/{job_id}`
- `GET /media/jobs/{job_id}/artifact`
- `GET /media/jobs/{job_id}/events`
- `POST /media/artifacts/{artifact_id}/evidence:locate`

Do not implement the long-term HTTP shape before the artifact contract and MCP tool contract are stable. A parallel `/media` API that bypasses the current pipeline would be a regression.

## Artifact Contract

The canonical public artifact is `media_artifact`. It is not a replacement for existing internal IR. It is a stable projection over current and future media understanding outputs.

### `media_artifact`

Required fields:

```json
{
  "schema_version": "1.0",
  "artifact_id": "string",
  "media_id": "string",
  "source": {
    "type": "url | local_file | transcript_ref",
    "value": "string",
    "platform": "string | null",
    "input_digest": "string"
  },
  "scene_mode": "lecture | recap | meeting | live_highlight | sports_highlight",
  "status": "succeeded | degraded | failed",
  "title": "string",
  "duration_ms": 0,
  "summary": {
    "short": "string",
    "key_points": [],
    "chapters": []
  },
  "evidence_index_ref": "string",
  "assets": {
    "cover_ref": "string | null",
    "frame_refs": []
  },
  "provenance": {
    "transcript_source": "platform_cc | asr | provided_transcript | unavailable",
    "asr_model": "string | null",
    "frame_sampling_policy": "string",
    "vlm_policy": "string",
    "cache_policy": "string"
  },
  "quality": {
    "coverage_notes": [],
    "degradations": [],
    "warnings": [],
    "evidence_audit_status": "not_run | passed | needs_review | failed"
  },
  "run_record_ref": "string"
}
```

### `chapter`

Each chapter or segment should include:

```json
{
  "chapter_id": "string",
  "title": "string",
  "start_ms": 0,
  "end_ms": 0,
  "summary": "string",
  "key_points": [],
  "evidence_refs": []
}
```

### `evidence_item`

Each evidence item should include:

```json
{
  "evidence_id": "string",
  "modality": "transcript | frame | summary_block | metadata",
  "text": "string",
  "quote": "string",
  "start_ms": 0,
  "end_ms": 0,
  "frame_ref": "string | null",
  "source_object_ref": "string",
  "score": 0.0,
  "confidence": "high | medium | low",
  "reason": "string"
}
```

### `evidence_pack`

```json
{
  "artifact_ref": "string",
  "query": "string",
  "claim": "string | null",
  "items": [],
  "ranking_policy": "string",
  "coverage_notes": [],
  "run_record_ref": "string"
}
```

### `highlight_candidates`

```json
{
  "artifact_ref": "string",
  "scene_mode": "string",
  "strategy": "string",
  "candidates": [
    {
      "highlight_id": "string",
      "start_ms": 0,
      "end_ms": 0,
      "title": "string",
      "reason_labels": [],
      "representative_frame_refs": [],
      "supporting_evidence_refs": [],
      "score": 0.0,
      "confidence": "high | medium | low"
    }
  ],
  "limitations": [],
  "run_record_ref": "string"
}
```

## Scene Modes

Scene modes are output profiles first. They should become separate skills only when their inputs, quality bar, and evaluation diverge enough to justify a new package.

### V1 Production Mode: `lecture`

Use for long-form educational videos, tutorials, technical talks, and lectures.

Quality bar:

- coherent concept structure
- chapter-level summaries
- timestamped evidence
- key frame evidence for visual claims
- explicit degradation notes

This is the first mode that should be made genuinely strong.

### V1 Optional Mode: `recap`

Use for long-form content recap when transcript and visual evidence are available but the output does not need a teaching-note structure.

Quality bar:

- narrative digest
- major event/section coverage
- evidence refs for important claims
- no unsupported editorial claims

### Later Mode: `meeting`

Do not claim production meeting minutes until the system supports at least:

- speaker or participant metadata when available
- decisions
- action items
- disagreements
- follow-ups
- uncertainty when diarization or attribution is missing

Meeting evaluation should use meeting-summary datasets and human review before resume claims become strong.

### Later Mode: `live_highlight`

Do not claim real-time live highlight extraction until the system has:

- streaming or near-real-time ingestion
- incremental transcript and frame windows
- event/state buffering
- latency-aware artifact updates
- replayable highlight evidence

### Later Mode: `sports_highlight`

Sports highlight extraction is not just generic video summarization. It needs domain signals such as scoreboard changes, commentary peaks, player/team entities, event types, replay shots, and platform-specific recap style.

Keep it as a future specialization unless those signals and evaluations exist.

## Skill Package Design

The skill should follow progressive disclosure:

- metadata decides when the skill triggers
- `SKILL.md` gives only the essential workflow
- references carry schemas and detailed policies
- scripts perform deterministic wrapper and validation work
- assets contain example artifacts and templates

Recommended directory:

```text
media-understanding/
  SKILL.md
  agents/
    openai.yaml
  scripts/
    summarize_media.py
    get_media_artifact.py
    locate_evidence.py
    validate_artifact.py
    inspect_run_record.py
  references/
    service_contract.md
    artifact_schema.md
    mcp_tools.md
    failure_and_degradation.md
    scene_modes.md
    quality_rubric.md
    evaluation.md
  assets/
    examples/
      lecture_media_artifact.json
      lecture_evidence_pack.json
      degraded_run_record.json
```

Do not add `README.md`, `INSTALLATION_GUIDE.md`, `QUICK_REFERENCE.md`, or changelog files inside the skill unless a future distribution system explicitly requires them.

### `SKILL.md` Scope

`SKILL.md` should contain:

- when to use the skill
- preferred call sequence
- how to choose `scene_mode`
- when to read each reference file
- how to validate artifacts before relying on them
- how to answer follow-up user questions using evidence refs
- how to report degradation honestly

`SKILL.md` should not contain:

- full JSON schemas
- long benchmark explanations
- implementation details of video download, ASR, VLM, cache, RAG, or storage
- a copied tutorial from `bilibili-render-pdf`

### Scripts

Scripts should be thin and deterministic:

- `summarize_media.py`: call MCP/HTTP adapter, normalize response, print refs.
- `get_media_artifact.py`: fetch artifact by ref and optionally select a projection.
- `locate_evidence.py`: call evidence tool and print structured result.
- `validate_artifact.py`: validate artifact schema and required evidence links.
- `inspect_run_record.py`: summarize verdict, warnings, degradations, and missing dependencies.

Scripts should not reimplement the media pipeline.

## Relationship To `bilibili-render-pdf`

Keep these ideas:

- high pedagogical quality bar
- long-video segmentation
- subtitle-first and ASR fallback discipline
- frame selection by teaching value, not arbitrary quota
- direct visual inspection standard for important frames
- timestamp provenance for visual claims
- final synthesis quality

Do not keep these as canonical behavior:

- Bilibili-only assumptions
- PDF/LaTeX as the main artifact
- manual frame inspection as the normal runtime path
- monolithic `SKILL.md`
- lack of machine-validatable artifact contracts

The right migration is to turn `bilibili-render-pdf` into quality references:

- `quality_rubric.md`
- `scene_modes.md`
- `failure_and_degradation.md`
- optional render templates outside the canonical artifact contract

## Ingestion And Understanding Quality Policy

Quality is the main differentiator. The system should spend effort where it improves evidence reliability and reader usefulness.

### Source Acquisition

Preferred order:

1. Platform metadata and subtitles.
2. Downloadable audio/video assets.
3. ASR fallback when subtitles are unavailable.
4. User-provided transcript when platform access is unavailable.

Every artifact must record what source path was used.

### Transcript Policy

Prefer timestamped subtitles over flattened transcript text. If ASR is used, record model, language, confidence if available, and known limitations.

### Frame Policy

Use scene detection and transcript-aligned windows to maximize recall before expensive VLM selection. Important visual claims should have frame refs or explicitly state that visual evidence was unavailable.

### VLM Policy

Use VLM calls for high-value frames and ambiguous visual claims. Cache results by content digest and policy so that later calls reuse the same evidence rather than regenerate inconsistent descriptions.

### Degradation Policy

Degradation is acceptable only when it is visible:

- no subtitles -> ASR fallback or transcript-unavailable warning
- no video frames -> transcript-only artifact
- VLM unavailable -> visual evidence unavailable warning
- missing `ffmpeg`/`yt-dlp` -> dependency failure or provided-transcript mode
- invalid schema -> failed artifact, not cached success

## Evaluation And Proof That It Is Not A Toy

The project should prove three things:

1. The skill can be called by an agent without reading the backend code.
2. The artifact can be reused in later calls without reprocessing the media.
3. Important claims can be audited against timestamps, transcript spans, or frames.

### Required V1 Demo

Build one strong lecture demo:

- one real long-form video source
- complete `media_artifact`
- `evidence_pack` for at least five user-like follow-up questions
- `run_record` with policy snapshot and verdict
- a degraded-path example, such as transcript-only or missing VLM
- a small rendered view only after JSON artifacts are valid

### Manual Evidence Audit

For every serious demo, sample at least 20 claims:

- claim text
- artifact location
- evidence refs
- timestamp accuracy
- whether transcript/frame supports the claim
- audit verdict: supported, partially supported, unsupported

Unsupported claims must become either corrected output or explicit limitations.

### Automated Checks

Minimum automated checks:

- skill folder validates
- `SKILL.md` frontmatter is valid
- artifact JSON validates against schema
- every chapter evidence ref resolves
- every evidence item has a modality and source object
- degraded artifacts include visible warnings
- `init.ps1` or equivalent startup path still passes

### Benchmarks

Use benchmarks only when they match the scene:

- lecture/long-video: build a small project-specific gold set first
- meeting: use meeting summarization datasets and human review
- generic video summary/highlights: use video summarization datasets as directional references, not as proof of sports or live capability
- sports/live: create domain-specific event and highlight evaluation before claiming production readiness

Metrics should include:

- evidence precision
- timestamp accuracy
- coverage of major segments
- unsupported-claim rate
- artifact schema validity
- degradation honesty
- follow-up reuse success
- reader usefulness

## Implementation Roadmap

### Phase 0: Spec And Boundary Lock

Goal: prevent the project from drifting back into a platform shell.

Deliverables:

- this replacement spec
- old spec marked superseded
- feature/progress state updated
- startup verification recorded

### Phase 1: Skill Skeleton

Goal: create a high-quality skill package without duplicating the backend.

Deliverables:

- `media-understanding/SKILL.md`
- `agents/openai.yaml`
- references for artifact schema, tools, failure, scenes, quality, evaluation
- scripts for calling and validating artifacts
- no copied backend pipeline code

Validation:

- skill validation script passes
- at least one forward test with a fresh agent-like prompt
- artifact validator runs on sample artifacts

### Phase 2: Adapter Over Existing LectureMind

Goal: expose current lecture path through the new skill surface.

Deliverables:

- `summarize_media` maps to existing pipeline or MCP write tool
- `get_media_artifact` maps current IR/evidence/run records to `media_artifact`
- `locate_evidence` maps existing evidence tools to `evidence_pack`
- dependency readiness check for `ffmpeg` and `yt-dlp`

Validation:

- existing tests pass
- MCP tool contract tests added
- schema round-trip tests added
- degraded dependency tests added

### Phase 3: Demo And Evaluation

Goal: show a non-toy, auditable capability.

Deliverables:

- one strong lecture demo
- one degraded demo
- evidence audit table
- artifact examples in skill assets
- concise interview walkthrough

Validation:

- artifact schema validation
- evidence ref resolution
- manual audit pass recorded
- startup path passes with documented dependencies

### Phase 4: Scene Expansion

Goal: add scene modes only after they have contracts and evaluation.

Order:

1. `recap` from transcript-rich long videos
2. `meeting` with speaker/action-item contract
3. `highlight` candidate extraction for non-real-time recap
4. sports/live only as dedicated specializations

Each scene expansion needs its own quality rubric and benchmark plan.

## Interview And Resume Positioning

Use truthful language that distinguishes built capability from target shape.

Safe project description:

> I built LectureMind from a long-form learning-video scenario, then reframed the valuable part into an evidence-grounded media understanding skill. The important shift was from a note/library product to an agent-callable capability: structured media artifacts, timestamped evidence, reusable follow-up tools, and runtime records for degraded long-media processing.

Safe bullets after Phase 1/2:

- Designed an evidence-grounded media understanding skill that exposes long-video summary and evidence lookup through structured artifacts rather than a lecture-library UI.
- Reused LectureMind's existing `LectureNoteIR`, `EvidenceIndex`, MCP tools, and runtime harness as the service core instead of duplicating backend logic inside the skill.
- Defined a public `media_artifact` contract with provenance, degradation, evidence refs, and run-record links so downstream agents can reuse results and audit claims.

Only after implementation and verification should stronger bullets mention:

- runnable `summarize_media` / `locate_evidence` wrappers
- validated demo artifacts
- ASR fallback behavior
- VLM/frame evidence policy
- meeting or highlight modes

Do not claim Tencent Meeting, live stream, sports highlight, or production multi-user service before there is a working path and evaluation evidence.

## Risks

### Risk: Rebranding Without Capability

Mitigation: make the demo artifact, evidence audit, and tool trace the center of the presentation.

### Risk: Parallel Data Models

Mitigation: keep `LectureNoteIR`, `EvidenceIndex`, and `RunRecord` as internal truth; make `media_artifact` a public projection.

### Risk: Skill Becomes A Backend Dump

Mitigation: keep `SKILL.md` short and move schemas/policies to references; scripts call the service and validate outputs only.

### Risk: Overclaiming Scene Modes

Mitigation: mark meeting/live/sports as future modes until prerequisites and benchmarks exist.

### Risk: Environment-Dependent Demo Failure

Mitigation: add dependency checks and transcript-only fallback demos; record `ffmpeg`/`yt-dlp` readiness explicitly.

## Follow-Up Directions To Preserve

Do these in order unless a blocker forces a change:

1. Build the skill package skeleton.
2. Write the artifact schema reference before wrappers.
3. Write the validator before trusting sample artifacts.
4. Adapt existing LectureMind outputs into `media_artifact`.
5. Add `summarize_media`, `get_media_artifact`, and `locate_evidence` wrappers.
6. Produce one strong lecture demo with evidence audit.
7. Add degraded-path demos.
8. Only then consider `recap` mode.
9. Only after recap is stable, design meeting mode.
10. Keep live and sports highlights as separate future design specs.

Memory rule:

> If a future task tries to add a new scene mode before artifact validation and evidence audit are stable, stop and return to the quality bar.

## Done Definition For The Skill

The skill is not done when the folder exists. It is done only when:

- another agent can trigger it from the metadata description
- `SKILL.md` stays concise and points to the right references
- wrappers call the service instead of reimplementing it
- artifact validation passes
- evidence refs resolve
- degraded outputs are honest
- one real demo and one degraded demo are reproducible
- the repo startup path remains passing
- resume wording matches verified behavior

## Spec Self-Check

- No placeholder sections remain.
- The first implementation slice is narrow enough for one plan.
- Existing repository truth models are preserved.
- Scene modes are scoped by maturity.
- Quality priority is explicit.
- The spec distinguishes existing capability, short-term work, and long-term goals.
- The next step should be an implementation plan for Phase 1, not immediate backend expansion.
