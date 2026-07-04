# Media Understanding Skill Suite Design

> Superseded on 2026-07-04 by `docs/superpowers/specs/2026-07-04-evidence-grounded-media-understanding-skill-design.md`.
> Keep this file as a historical draft only. Do not use it as the active implementation source of truth.

Date: 2026-07-04

## Status

Approved for design capture. This document records the agreed direction before implementation planning.

## Problem

LectureMind started as a learning-video summarization and Q&A project. That entry point made the project concrete, but it also made the outer product look like a personal lecture-note or library-management system.

The stronger business value is the underlying media understanding capability:

- summarize long-form multimedia content
- locate evidence with timestamps, source snippets, and key frames
- extract highlight candidates for recap or secondary distribution
- expose the capability to agents and business systems without forcing a custom UI

The upgraded direction is to keep LectureMind as the historical prototype and reposition the active work as a reusable skill suite:

> Media Understanding Skill Suite: a multimedia understanding skill suite exposed through MCP tools and HTTP APIs, backed by asynchronous jobs, structured artifacts, and runtime harness observability.

## Goals

1. Reframe the project from a lecture-note product into reusable media understanding capabilities.
2. Provide a mature skill-oriented shape that can be called by agent platforms such as Codex-like or OpenClaw-like systems.
3. Keep the valuable backend engineering work: job execution, resource boundaries, degradation, recovery, and observability.
4. Produce stable structured artifacts instead of treating HTML pages as the primary output.
5. Support business-facing language for meeting minutes, long-video recap, learning-video summary, and live/highlight recap scenarios.

## Non-Goals

The first implementation pass must avoid widening back into a platform project.

- Do not build a full frontend product.
- Do not build a generic agent framework.
- Do not implement a complete multi-tenant permission system.
- Do not implement real-time live stream ingestion end to end.
- Do not rewrite the existing lecture generation workflow from scratch.
- Do not make HTML handouts the canonical output format.
- Do not claim production-grade concurrent multi-user service before resource governance and deployment work exist.

## Product Shape

The project should be described as a skill suite with a shared capability core.

### Shared Capability Core

The shared core is not marketed as a user-facing feature. It supports all exposed skills.

- media acquisition: fetch video metadata, subtitles, cover image, audio, and key frames
- transcript handling: prefer platform subtitles; fall back to ASR when subtitles are unavailable
- segmentation: split by chapters, time windows, scene changes, or transcript boundaries
- visual sampling: use scene detection and tiered scoring before expensive VLM calls
- evidence indexing: model text chunks, frame references, timestamps, and summary blocks as retrievable evidence
- artifact management: emit structured JSON artifacts for downstream agents, APIs, and demo renderers
- runtime envelope: record job state, policy snapshot, stage events, fallback path, verdict, and artifact references

### Public Skills

The first public skill set contains three tools.

#### summarize_media

Purpose: generate structured media summaries.

Inputs:

- media URL, local media path, audio path, or existing transcript
- optional scene mode: `meeting`, `lecture`, `recap`, `live_highlight`
- optional quality policy: `fast`, `standard`, `high_quality`

Outputs:

- summary artifact
- chapters or segments
- key points
- action items when scene mode is `meeting`
- evidence references for important claims
- job or artifact reference for follow-up calls

#### locate_evidence

Purpose: locate evidence for a question, claim, topic, or summary paragraph.

Inputs:

- query, claim, or topic
- media job reference or artifact reference
- optional evidence type filters: transcript, segment, key frame, timestamp, summary block

Outputs:

- evidence pack
- source snippets
- timestamp anchors
- key frame references
- confidence or ranking signals
- explanation of why each evidence item was selected

#### extract_highlights

Purpose: extract highlight candidates for recap or content reuse.

Inputs:

- media URL or existing job reference
- optional scene mode
- optional highlight strategy

Outputs:

- highlight candidates
- start and end timestamps
- representative key frames
- reason labels
- supporting transcript snippets or scene signals

This tool is a differentiator, but it should not block the first useful version. It can start as a simple candidate generator using transcript density, scene changes, summary salience, and key frame quality.

### Second-Phase Skill

#### qa_media

Purpose: answer multi-turn questions about a processed media artifact with evidence grounding.

This should remain second phase because it depends on good summary artifacts and evidence packs. The first phase should make `summarize_media` and `locate_evidence` reliable before adding conversational behavior.

## Interface Design

The project exposes one capability core through two interface families.

### MCP Tools

MCP is the agent-facing interface. It should expose small, task-shaped tools rather than the full internal pipeline.

Tools:

- `summarize_media`
- `locate_evidence`
- `extract_highlights`

Each tool should return structured content and a lightweight `job_ref` or `artifact_ref`. Follow-up agent calls should reuse those references instead of reprocessing the media.

### HTTP API

HTTP is the business-service interface. It should use a job model for media-heavy tasks.

Minimal endpoints:

- `POST /media/jobs`
- `GET /media/jobs/{job_id}`
- `GET /media/jobs/{job_id}/result`
- `GET /media/jobs/{job_id}/events`

Job states:

- `queued`
- `running`
- `succeeded`
- `degraded`
- `failed`

Design rule:

- lightweight tasks may return synchronously
- video and ASR-heavy tasks should run asynchronously through jobs

This lets the same system be discussed as both an agent skill suite and a deployable backend service.

## Artifact Contract

HTML is a rendering view, not the source of truth.

Canonical output should be structured JSON artifacts.

### summary_artifact

Fields:

- `artifact_id`
- `media_id`
- `source`
- `scene_mode`
- `title`
- `duration`
- `summary`
- `chapters`
- `key_points`
- `action_items`
- `evidence_refs`
- `created_at`
- `run_record_ref`

### evidence_pack

Fields:

- `artifact_id`
- `media_id`
- `query`
- `evidence_items`
- `ranking_policy`
- `coverage_notes`
- `run_record_ref`

Each evidence item should contain:

- `evidence_id`
- `evidence_type`
- `text`
- `start_time`
- `end_time`
- `frame_ref`
- `source_segment_id`
- `score`
- `reason`

### highlight_candidates

Fields:

- `artifact_id`
- `media_id`
- `scene_mode`
- `candidates`
- `selection_policy`
- `run_record_ref`

Each candidate should contain:

- `highlight_id`
- `start_time`
- `end_time`
- `title`
- `reason_labels`
- `representative_frame_refs`
- `supporting_evidence_refs`
- `score`

### run_record

Fields:

- `trace_id`
- `job_id`
- `skill_name`
- `input_digest`
- `policy_snapshot`
- `stage_events`
- `cost_usage`
- `fallback_path`
- `artifact_refs`
- `verdict`

## Runtime Harness

The runtime harness should be explained as the skill execution contract, not as a separate platform.

Each significant call should produce a run record. The run record exists for:

- debugging failed runs
- explaining degraded outputs
- avoiding cache pollution from failed artifacts
- comparing fast, standard, and high-quality policies
- supporting future multi-user service operation

Minimum stage events:

- `ingest_started`
- `ingest_completed`
- `transcript_selected`
- `asr_fallback_started`
- `frame_sampling_completed`
- `segmentation_completed`
- `summary_completed`
- `evidence_index_completed`
- `artifact_validated`
- `job_finalized`

Failure handling rules:

- network and temporary provider failures may retry with bounded backoff
- missing subtitles should fall back to ASR
- long videos should route to segmentation or map-reduce style processing
- VLM budget pressure should lower visual inspection depth before failing the whole job
- invalid artifacts should not be cached as successful results
- degraded jobs must return a clear `degraded` verdict and record what was skipped or approximated

## Scene Modes

The same capability core should support different business language through scene modes.

### meeting

Focus:

- decisions
- action items
- disagreements
- follow-ups
- speaker-attributed evidence when available

### lecture

Focus:

- concepts
- chapter structure
- formulas, diagrams, and code
- learning-oriented explanations
- timestamped evidence for review

### recap

Focus:

- narrative summary
- major events
- content structure
- user-facing digest

### live_highlight

Focus:

- scene changes
- emotional or semantic peaks
- dense commentary windows
- visually representative frames
- short clips for secondary distribution

The first implementation does not need perfect scene-specific algorithms. It does need explicit mode labels and mode-aware output fields so the project no longer reads as a single learning-note system.

## Two-Day Implementation Scope

The immediate goal is not to rebuild the entire system. It is to create a convincing skill-suite shape and one narrow runnable path.

### Must Do

1. Add or prepare a `media-understanding` skill directory.
2. Write a concise `SKILL.md` for the suite.
3. Add `references/artifact_schema.md`.
4. Add `references/failure_and_degradation.md`.
5. Add `references/scene_modes.md`.
6. Produce one demo output containing `summary_artifact` and `evidence_pack`.
7. Update resume and interview wording so LectureMind is described as the prototype, not the final product shape.

### Should Do

1. Add a thin script wrapper for `summarize_media`.
2. Add a thin script wrapper for `locate_evidence`.
3. Add an artifact validator for required fields.
4. Add a demo HTML renderer only after the JSON artifact exists.

### Could Do

1. Add `extract_highlights` as a lightweight candidate generator.
2. Add one non-lecture demo case.
3. Add MCP tool metadata examples.
4. Add HTTP endpoint examples if the current codebase already has a compatible service layer.

## Interview Positioning

Use this positioning:

> I initially built LectureMind from the learning-video scenario, but after reviewing the business fit I realized the durable value was not the lecture-note UI. I refactored the direction into a Media Understanding Skill Suite. The suite exposes `summarize_media`, `locate_evidence`, and `extract_highlights` through MCP tools for agents and HTTP APIs for business systems. The core output is structured artifacts with summaries, evidence, timestamps, key frames, and run records. The backend keeps asynchronous jobs, fallback policies, and runtime harness observability so long media tasks can degrade, recover, and be debugged.

This positioning preserves the real engineering work while avoiding the impression that the project is only a personal knowledge-base or library-management product.

## Resume Positioning

Recommended project title:

> Media Understanding Skill Suite: 多媒体总结、证据定位与高光提取能力套装

Recommended description:

> 面向会议纪要、长视频 recap、学习视频总结等场景，将原 LectureMind 学习讲义系统抽象为可被 Agent / 业务系统调用的多媒体理解能力。

Recommended bullets:

- 设计 `summarize_media`、`locate_evidence`、`extract_highlights` 三类核心 skill，对外通过 MCP tools 与 HTTP API 暴露统一能力；支持视频链接输入，输出结构化摘要、章节、证据片段、时间锚点与关键帧 artifact。
- 构建多模态处理链路：并行获取字幕、封面与关键帧；无字幕时回退 ASR；基于场景检测与分级评分筛选关键帧，降低 VLM 调用成本。
- 设计结构化 artifact contract，将 HTML 讲义降级为 demo rendering，核心结果以 `summary_artifact`、`evidence_pack`、`highlight_candidates` 和 `run_record` 形式供上层 Agent 复用。
- 引入 runtime harness 管理长任务执行：通过 `trace_id`、`policy_snapshot`、`stage_events`、`fallback_path` 和 `verdict` 记录任务状态，支持分治降级、失败恢复和可观测排查。
- 基于 SQLite、sqlite-vec、FTS5 与 RRF 实现本地混合检索，支持围绕片段、关键帧、时间锚点和摘要块定位证据，为多轮问答与外部 Agent 调用提供检索基础。

## External References

- Agent Skills: https://agentskills.io/
- Agent Skills best practices: https://agentskills.io/skill-creation/best-practices
- Agent Skills script guidance: https://agentskills.io/skill-creation/using-scripts
- Anthropic public skills repository: https://github.com/anthropics/skills
- MCP tool specification: https://modelcontextprotocol.io/specification/2025-06-18/server/tools

## Spec Self-Check

- No placeholder sections remain.
- Scope is intentionally limited to a skill-suite redesign and a two-day execution slice.
- The design distinguishes skill interface, service interface, artifact contract, and runtime harness.
- HTML output is explicitly non-canonical.
- The design does not claim completed production multi-user deployment.
- The next step should be implementation planning only after user review.
