"""Copilot + MCP Tool Registry.

This module exposes a single, normalised surface area that **both** the
in-process Copilot Agent (Stage 4) **and** the external MCP server
(Stage 7) consume.  Each tool is:

1. A pure async function taking a :class:`ToolContext` as its first
   argument plus typed keyword arguments.
2. Documented with a Chinese-English one-liner on the first line of the
   docstring (so ``fastmcp`` auto-extracts a useful description) and a
   detailed parameter / return block beneath.
3. Validated by Pydantic models (one per ``Input`` / ``Output``) so
   LangChain ``@tool`` and fastmcp ``@mcp.tool`` can both pull the JSON
   schema directly without re-declaration.
4. Free of FastAPI globals — every external dependency is reached
   through ``ToolContext`` (database, RAG store, optional pipeline).

Failures are translated into :class:`ToolError` with a short ``code`` so
the agent layer can decide rendering (``"not_found"`` becomes a polite
"该章节不存在", ``"invalid_arg"`` is surfaced as a 400, etc.).

Stage-3 ships the underlying functions only.  Stage 4 wraps the five
Copilot tools with LangChain ``@tool`` (closing over a captured
``ToolContext``); Stage 7 wraps the MCP-only tools with fastmcp
``@mcp.tool`` similarly.  No state lives here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_settings
from app.storage.db import Database, SummaryRow
from app.understand.schema import (
    Chapter,
    ContentBlock,
    EvidenceObject,
    EvidenceRelation,
    Frame,
    KnowledgeUnitView,
    LectureJSON,
    TeachingUnit,
)

from .rag import RAGStore

if TYPE_CHECKING:
    from app.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


ToolErrorCode = Literal[
    "not_found",
    "invalid_arg",
    "unsupported",
    "internal",
]


class ToolError(Exception):
    """Uniform error type for every tool in the registry.

    The agent layer (Stage 4) catches this and turns it into a tool-call
    response that the LLM can reason about; the MCP server (Stage 7)
    maps it to a structured error envelope.  Always raise with a
    machine-friendly ``code`` and a user-facing ``message``.
    """

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Bundle of dependencies every tool can ask for.

    ``pipeline`` is optional because most tools are read-only;
    ``summarize_video`` is the only tool that needs it, and it raises
    :class:`ToolError` ``"unsupported"`` when ``pipeline is None``
    (e.g. running the MCP server in stdio mode without the FastAPI app
    process).
    """

    db: Database
    rag: RAGStore
    pipeline: "Pipeline | None" = None


# ---------------------------------------------------------------------------
# Input / Output models (Pydantic)
# ---------------------------------------------------------------------------


_StrictModel = ConfigDict(extra="forbid", populate_by_name=True)


class SearchLectureInput(BaseModel):
    model_config = _StrictModel
    bv: str = Field(..., description="Bilibili BV id of the lecture to search within.")
    query: str = Field(..., min_length=1, description="Free-form natural-language query.")
    top_k: int = Field(5, ge=1, le=20, description="Maximum chunks to return (1-20).")


class SearchHit(BaseModel):
    chunk_id: int
    bv_id: str
    kind: str
    chapter_idx: int | None
    t_start: int | None
    t_end: int | None
    text: str
    quote: str = ""
    score: float
    meta: dict[str, Any] = Field(default_factory=dict)
    evidence_id: str = ""
    evidence_kind: str = ""
    note_node_id: str = ""
    source_path: str = ""

    @model_validator(mode="after")
    def _lift_backlinks_from_meta(self) -> "SearchHit":
        meta = self.meta or {}
        if not self.evidence_id:
            self.evidence_id = str(meta.get("evidence_id") or "")
        if not self.evidence_kind:
            self.evidence_kind = str(meta.get("evidence_kind") or "")
        if not self.note_node_id:
            self.note_node_id = str(meta.get("note_node_id") or "")
        if not self.source_path:
            self.source_path = str(meta.get("path") or "")
        return self


class SearchLectureOutput(BaseModel):
    chunks: list[SearchHit]


class SearchEvidenceInput(BaseModel):
    model_config = _StrictModel
    bv: str = Field(..., description="Bilibili BV id of the lecture to search within.")
    query: str = Field(..., min_length=1, description="Free-form evidence lookup query.")
    top_k: int = Field(5, ge=1, le=20, description="Maximum evidence hits to return (1-20).")


class SearchEvidenceHit(BaseModel):
    evidence_id: str
    kind: str
    title: str = ""
    summary: str = ""
    chapter_idx: int | None = None
    t_start: int | None = None
    t_end: int | None = None
    note_node_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    matched_text: str = ""
    anchors: dict[str, Any] = Field(default_factory=dict)
    quote: str = ""
    path: str = ""


class SearchEvidenceOutput(BaseModel):
    hits: list[SearchEvidenceHit] = Field(default_factory=list)


class GetNoteUnitInput(BaseModel):
    model_config = _StrictModel
    bv: str
    unit_id: str = Field(..., min_length=1, description="Teaching unit id or content block id.")


class GetNoteUnitOutput(BaseModel):
    bv_id: str
    requested_node_id: str
    resolved_node_type: str = "teaching_unit"
    unit: TeachingUnit
    focused_block: ContentBlock | None = None
    related_evidence_ids: list[str] = Field(default_factory=list)
    primary_ref_kind: str = "note"
    primary_ref_id: str = ""
    primary_evidence_id: str = ""


class GetEvidenceObjectInput(BaseModel):
    model_config = _StrictModel
    bv: str
    evidence_id: str = Field(..., min_length=1)


class GetEvidenceObjectOutput(BaseModel):
    bv_id: str
    evidence: EvidenceObject
    relations: list[EvidenceRelation] = Field(default_factory=list)
    related_note_unit_ids: list[str] = Field(default_factory=list)
    primary_ref_kind: str = "evidence"
    primary_ref_id: str = ""
    primary_note_node_id: str = ""


class WebSearchInput(BaseModel):
    model_config = _StrictModel
    query: str = Field(..., min_length=1, description="Web search query.")
    count: int = Field(5, ge=1, le=10, description="Maximum web results to return.")


class WebSearchResult(BaseModel):
    title: str = ""
    url: str = ""
    hostname: str = ""
    snippet: str = ""


class WebSearchOutput(BaseModel):
    query: str
    results: list[WebSearchResult] = Field(default_factory=list)
    error: dict[str, str] | None = None


class GetChapterInput(BaseModel):
    model_config = _StrictModel
    bv: str
    chapter_idx: int = Field(..., ge=1, description="1-based chapter index.")


class ChapterFrameView(BaseModel):
    frame_id: int
    ts: int
    path: str
    caption: str = ""
    ocr_text: str = ""
    insight: str = ""
    visual_type: str = ""


class ChapterPointView(BaseModel):
    text: str
    ts: int
    quote: str = ""


class GetChapterOutput(BaseModel):
    bv_id: str
    chapter_idx: int
    title: str
    start: int
    end: int
    summary: str
    learning_goal: str = ""
    teaching_notes: list[str] = Field(default_factory=list)
    points: list[ChapterPointView] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    frames: list[ChapterFrameView] = Field(default_factory=list)
    note_unit_id: str = ""
    note_unit_title: str = ""
    related_evidence_ids: list[str] = Field(default_factory=list)
    related_note_node_ids: list[str] = Field(default_factory=list)
    compatibility_anchor_kind: str = "chapter"
    compatibility_anchor_id: str = ""
    primary_ref_kind: str = ""
    primary_ref_id: str = ""
    primary_evidence_id: str = ""
    primary_note_node_id: str = ""
    primary_note_unit_id: str = ""
    primary_note_node_type: str = ""
    primary_note_title: str = ""
    primary_evidence_kind: str = ""


class GetFrameInput(BaseModel):
    model_config = _StrictModel
    bv: str
    frame_id: int = Field(..., ge=1, description="1-based global frame index (matches [F7] anchor).")


class GetFrameOutput(BaseModel):
    bv_id: str
    frame_id: int
    ts: int
    chapter_idx: int | None
    path: str
    caption: str = ""
    ocr_text: str = ""
    insight: str = ""
    visual_type: str = ""
    importance_score: float = 0.0
    related_evidence_ids: list[str] = Field(default_factory=list)
    related_note_node_ids: list[str] = Field(default_factory=list)
    compatibility_anchor_kind: str = "frame"
    compatibility_anchor_id: str = ""
    primary_ref_kind: str = ""
    primary_ref_id: str = ""
    primary_evidence_id: str = ""
    primary_note_node_id: str = ""
    primary_note_unit_id: str = ""
    primary_note_node_type: str = ""
    primary_note_title: str = ""
    primary_evidence_kind: str = ""


class GetQuoteContextInput(BaseModel):
    model_config = _StrictModel
    bv: str
    quote: str = Field(..., min_length=1)


class NeighborQuoteView(BaseModel):
    text: str
    ts: int
    quote: str = ""


class GetQuoteContextOutput(BaseModel):
    bv_id: str
    quote: str
    matched_quote: str
    t_start: int
    t_end: int
    chapter_idx: int
    neighbor_quotes: list[NeighborQuoteView] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    related_note_node_ids: list[str] = Field(default_factory=list)
    compatibility_anchor_kind: str = "quote"
    compatibility_anchor_id: str = ""
    primary_ref_kind: str = ""
    primary_ref_id: str = ""
    primary_evidence_id: str = ""
    primary_note_node_id: str = ""
    primary_note_unit_id: str = ""
    primary_note_node_type: str = ""
    primary_note_title: str = ""
    primary_evidence_kind: str = ""


class ExplainFrameInput(BaseModel):
    model_config = _StrictModel
    bv: str
    frame_id: int = Field(..., ge=1)


class ExplainFrameOutput(BaseModel):
    bv_id: str
    frame_id: int
    ts: int
    chapter_idx: int | None
    path: str
    caption: str = ""
    ocr_text: str = ""
    why_useful: str = ""
    context_quotes: list[NeighborQuoteView] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    related_note_node_ids: list[str] = Field(default_factory=list)
    compatibility_anchor_kind: str = "frame"
    compatibility_anchor_id: str = ""
    primary_ref_kind: str = ""
    primary_ref_id: str = ""
    primary_evidence_id: str = ""
    primary_note_node_id: str = ""
    primary_note_unit_id: str = ""
    primary_note_node_type: str = ""
    primary_note_title: str = ""
    primary_evidence_kind: str = ""


class SummarizeVideoInput(BaseModel):
    model_config = _StrictModel
    url: str = Field(..., description="Bilibili video URL (https://www.bilibili.com/video/BV...).")
    force_refresh: bool = Field(False, description="When true, re-runs the pipeline even if cached.")


class SummarizeVideoOutput(BaseModel):
    bv: str
    report_url: str
    status: str
    taxonomy: dict[str, Any] | None = None


class SearchLecturesInput(BaseModel):
    model_config = _StrictModel
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    domain: str | None = None
    direction: str | None = None


class SearchLecturesOutput(BaseModel):
    chunks: list[SearchHit]


class GetKnowledgeUnitsInput(BaseModel):
    model_config = _StrictModel
    bv: str
    kind: str | None = Field(default=None, description="Optional unit type filter (e.g. concept / procedure / formula).")


class KnowledgeUnitOutItem(BaseModel):
    id: str = ""
    kind: str
    title: str
    explanation: str
    t_start: int
    t_end: int
    chapter_idx: int
    quote: str = ""


class GetKnowledgeUnitsOutput(BaseModel):
    bv_id: str
    units: list[KnowledgeUnitOutItem]


class ListLecturesInput(BaseModel):
    model_config = _StrictModel
    limit: int = Field(50, ge=1, le=500)
    domain: str | None = None
    direction: str | None = None


class LectureMetaItem(BaseModel):
    bv_id: str
    title: str = ""
    author: str = ""
    duration: int = 0
    cover_url: str = ""
    domain: str | None = None
    direction: str | None = None
    domain_tags: list[str] = Field(default_factory=list)
    report_path: str = ""
    updated_at: str = ""


class ListLecturesOutput(BaseModel):
    lectures: list[LectureMetaItem]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _load_lecture(ctx: ToolContext, bv: str) -> LectureJSON:
    row = await ctx.db.get_summary(bv)
    if row is None:
        raise ToolError("not_found", f"lecture {bv!r} not found in summaries")
    try:
        return LectureJSON.model_validate(row.summary_json)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("internal", f"summary_json for {bv} is corrupt: {exc}") from exc


def _evidence_objects(lecture: LectureJSON) -> list[Any]:
    evidence = lecture.evidence_index
    if evidence is None:
        return []
    return list(evidence.evidence_objects)


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _find_note_unit_for_chapter(lecture: LectureJSON, chapter_idx: int) -> tuple[str, str]:
    note = lecture.lecture_note_ir
    if note is not None:
        for unit in note.body.teaching_units:
            if chapter_idx in unit.source_chapter_refs:
                return unit.unit_id, unit.title
    for obj in _evidence_objects(lecture):
        if obj.kind == "chapter_evidence" and obj.chapter_index == chapter_idx and obj.note_node_ids:
            node_id = obj.note_node_ids[0]
            found = _find_note_node(lecture, node_id)
            if found is not None:
                return found[0].unit_id, found[0].title
            return node_id, ""
    return "", ""


def _explicit_chapter_primary_ref(
    lecture: LectureJSON,
    chapter_idx: int,
) -> tuple[str, str, str]:
    note = lecture.lecture_note_ir
    if note is not None:
        for unit in note.body.teaching_units:
            if chapter_idx in unit.source_chapter_refs:
                return "note", unit.unit_id, ""
    for obj in _evidence_objects(lecture):
        if obj.kind == "chapter_evidence" and obj.chapter_index == chapter_idx and obj.note_node_ids:
            return "evidence", obj.evidence_id, obj.evidence_id
    return "", "", ""


def _chapter_backlinks(lecture: LectureJSON, chapter_idx: int) -> tuple[list[str], list[str]]:
    evidence_ids = [
        obj.evidence_id for obj in _evidence_objects(lecture) if obj.chapter_index == chapter_idx
    ]
    note_unit_id, _ = _find_note_unit_for_chapter(lecture, chapter_idx)
    note_ids = [note_unit_id] if note_unit_id else []
    return _dedupe_strings(evidence_ids), _dedupe_strings(note_ids)


def _frame_backlinks(
    lecture: LectureJSON,
    frame: Frame,
    chapter_idx: int | None,
) -> tuple[list[str], list[str]]:
    evidence_ids: list[str] = []
    note_ids: list[str] = []
    ts = int(frame.ts)
    for obj in _evidence_objects(lecture):
        path_match = bool(frame.path and obj.path and obj.path == frame.path)
        time_match = (
            chapter_idx is not None
            and obj.chapter_index == chapter_idx
            and obj.ts is not None
            and int(obj.ts) == ts
        )
        if not path_match and not time_match:
            continue
        evidence_ids.append(obj.evidence_id)
        note_ids.extend(obj.note_node_ids)
    return _dedupe_strings(evidence_ids), _dedupe_strings(note_ids)


def _explicit_frame_primary_ref(
    lecture: LectureJSON,
    frame: Frame,
) -> tuple[str, str, str]:
    if not frame.path:
        return "", "", ""
    for obj in _evidence_objects(lecture):
        if obj.path and obj.path == frame.path:
            return "evidence", obj.evidence_id, obj.evidence_id
    return "", "", ""


def _quote_backlinks(
    lecture: LectureJSON,
    chapter_idx: int,
    needle: str,
) -> tuple[list[str], list[str]]:
    evidence_ids: list[str] = []
    note_ids: list[str] = []
    for obj in _evidence_objects(lecture):
        if obj.chapter_index != chapter_idx:
            continue
        haystacks = [
            str(obj.quote or ""),
            str(obj.text or ""),
            str(obj.source_payload.get("quote") or ""),
            str(obj.source_payload.get("point_text") or ""),
        ]
        if not any(needle in text for text in haystacks if text):
            continue
        evidence_ids.append(obj.evidence_id)
        note_ids.extend(obj.note_node_ids)
    return _dedupe_strings(evidence_ids), _dedupe_strings(note_ids)


def _explicit_quote_primary_ref(
    lecture: LectureJSON,
    chapter_idx: int,
    needle: str,
) -> tuple[str, str, str]:
    note = lecture.lecture_note_ir
    if note is None:
        return "", "", ""
    for unit in note.body.teaching_units:
        if chapter_idx not in unit.source_chapter_refs:
            continue
        for evidence_id in unit.evidence_refs:
            obj = _evidence_index_map(lecture).get(evidence_id)
            if obj is None:
                continue
            haystacks = [
                str(obj.quote or ""),
                str(obj.text or ""),
                str(obj.source_payload.get("quote") or ""),
                str(obj.source_payload.get("point_text") or ""),
            ]
            if any(needle in text for text in haystacks if text):
                return "evidence", obj.evidence_id, obj.evidence_id
        for block in unit.content_blocks:
            if chapter_idx not in block.source_chapter_refs:
                continue
            for evidence_id in block.evidence_refs:
                obj = _evidence_index_map(lecture).get(evidence_id)
                if obj is None:
                    continue
                haystacks = [
                    str(obj.quote or ""),
                    str(obj.text or ""),
                    str(obj.source_payload.get("quote") or ""),
                    str(obj.source_payload.get("point_text") or ""),
                ]
                if any(needle in text for text in haystacks if text):
                    return "evidence", obj.evidence_id, obj.evidence_id
    return "", "", ""


def _find_note_node(
    lecture: LectureJSON,
    node_id: str,
) -> tuple[TeachingUnit, ContentBlock | None] | None:
    note = lecture.lecture_note_ir
    if note is None:
        return None
    for unit in note.body.teaching_units:
        if unit.unit_id == node_id:
            return unit, None
        for block in unit.content_blocks:
            if block.block_id == node_id:
                return unit, block
    return None


def _related_evidence_for_note_node(lecture: LectureJSON, node_id: str) -> list[str]:
    matches: list[str] = []
    found = _find_note_node(lecture, node_id)
    if found is not None:
        unit, block = found
        matches.extend(unit.evidence_refs)
        if block is not None:
            matches.extend(block.evidence_refs)
    for obj in _evidence_objects(lecture):
        if node_id in obj.note_node_ids:
            matches.append(obj.evidence_id)
    return _dedupe_strings(matches)


def _evidence_index_map(lecture: LectureJSON) -> dict[str, EvidenceObject]:
    return {obj.evidence_id: obj for obj in _evidence_objects(lecture)}


def _note_unit_ids_for_evidence(lecture: LectureJSON, obj: EvidenceObject) -> list[str]:
    unit_ids: list[str] = []
    for node_id in obj.note_node_ids:
        found = _find_note_node(lecture, node_id)
        if found is not None:
            unit_ids.append(found[0].unit_id)
    return _dedupe_strings(unit_ids)


def _primary_note_ids(
    lecture: LectureJSON,
    *,
    related_note_node_ids: Iterable[str] = (),
    primary_evidence_id: str = "",
    primary_ref_kind: str = "",
    primary_ref_id: str = "",
) -> tuple[str, str]:
    node_ids = _dedupe_strings(related_note_node_ids)
    primary_note_node_id = node_ids[0] if node_ids else ""
    primary_note_unit_id = ""

    if primary_note_node_id:
        found = _find_note_node(lecture, primary_note_node_id)
        if found is not None:
            primary_note_unit_id = found[0].unit_id

    if not primary_note_node_id and primary_ref_kind == "note" and primary_ref_id:
        primary_note_node_id = primary_ref_id
        found = _find_note_node(lecture, primary_note_node_id)
        if found is not None:
            primary_note_unit_id = found[0].unit_id

    if not primary_note_unit_id and primary_ref_kind == "note" and primary_ref_id:
        found = _find_note_node(lecture, primary_ref_id)
        if found is not None:
            primary_note_unit_id = found[0].unit_id

    if primary_evidence_id:
        obj = _evidence_index_map(lecture).get(primary_evidence_id)
        if obj is not None:
            if not primary_note_node_id and obj.note_node_ids:
                primary_note_node_id = obj.note_node_ids[0]
            if not primary_note_unit_id:
                unit_ids = _note_unit_ids_for_evidence(lecture, obj)
                if unit_ids:
                    primary_note_unit_id = unit_ids[0]
            if not primary_note_unit_id and primary_note_node_id:
                found = _find_note_node(lecture, primary_note_node_id)
                if found is not None:
                    primary_note_unit_id = found[0].unit_id

    return primary_note_node_id, primary_note_unit_id


def _primary_note_meta(
    lecture: LectureJSON,
    *,
    primary_note_node_id: str = "",
    primary_note_unit_id: str = "",
) -> tuple[str, str]:
    if primary_note_node_id:
        found = _find_note_node(lecture, primary_note_node_id)
        if found is not None:
            unit, block = found
            if block is not None:
                return "content_block", block.title or unit.title
            return "teaching_unit", unit.title
    if primary_note_unit_id:
        found = _find_note_node(lecture, primary_note_unit_id)
        if found is not None:
            return "teaching_unit", found[0].title
    return "", ""


def _primary_evidence_kind(lecture: LectureJSON, primary_evidence_id: str) -> str:
    if not primary_evidence_id:
        return ""
    obj = _evidence_index_map(lecture).get(primary_evidence_id)
    return obj.kind if obj is not None else ""


def _canonical_primary_ref(
    *,
    primary_evidence_id: str = "",
    primary_note_node_id: str = "",
    compatibility_anchor_id: str = "",
) -> tuple[str, str]:
    if primary_evidence_id:
        return "evidence", primary_evidence_id
    if primary_note_node_id:
        return "note", primary_note_node_id
    return "", ""


def _score_text_match(query: str, text: str) -> float:
    query_lc = query.strip().lower()
    text_lc = text.strip().lower()
    if not query_lc or not text_lc:
        return 0.0
    if query_lc in text_lc:
        return 1.0
    tokens = [tok for tok in query_lc.split() if tok]
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if tok in text_lc)
    return hits / len(tokens)


def _flat_frames(lecture: LectureJSON) -> list[tuple[Frame, int | None]]:
    """Return ``[(frame, chapter_idx), ...]`` in the canonical anchor order.

    Order is *identical* to the renderer's ``_global_visual_evidence``
    output for the lecture-level slice (so ``[F7]`` anchors line up
    with the rendered HTML), then chapter-attached frames not yet seen
    by ``path`` are appended.  De-duplicated by ``path``.
    """
    out: list[tuple[Frame, int | None]] = []
    seen: set[str] = set()
    for f in lecture.visual_evidence:
        if not f.path or f.path in seen:
            continue
        seen.add(f.path)
        out.append((f, _chapter_for_ts(lecture, int(f.ts))))
    for ch in lecture.chapters:
        for f in ch.frames:
            if not f.path or f.path in seen:
                continue
            seen.add(f.path)
            out.append((f, ch.index))
    return out


def _chapter_for_ts(lecture: LectureJSON, ts: int) -> int | None:
    for ch in lecture.chapters:
        if ch.start <= ts <= ch.end:
            return ch.index
    return lecture.chapters[0].index if lecture.chapters else None


def _frame_view(idx: int, frame: Frame, chapter_idx: int | None) -> ChapterFrameView:
    return ChapterFrameView(
        frame_id=idx,
        ts=int(frame.ts),
        path=frame.path,
        caption=frame.caption,
        ocr_text=frame.ocr_text,
        insight=frame.insight,
        visual_type=frame.visual_type,
    )


def _summary_to_meta(s: SummaryRow, report_url_base: str = "/reports") -> LectureMetaItem:
    return LectureMetaItem(
        bv_id=s.bv_id,
        title=s.title or "",
        author=s.author or "",
        duration=int(s.duration or 0),
        cover_url=s.cover_url or "",
        domain=s.domain,
        direction=s.direction,
        domain_tags=list(s.domain_tags or []),
        report_path=s.report_path or "",
        updated_at=str(s.updated_at or ""),
    )


# ---------------------------------------------------------------------------
# Copilot internal tools
# ---------------------------------------------------------------------------


async def search_lecture(
    ctx: ToolContext,
    *,
    bv: str,
    query: str,
    top_k: int = 5,
) -> SearchLectureOutput:
    """单 BV 内 RAG 检索 / Hybrid retrieval scoped to one lecture.

    Runs vector top-20 + FTS5 top-20 → reciprocal rank fusion → top_k.
    Use this whenever the user's question is about content inside *the
    current lecture* (the common Copilot path).

    :param bv: BV id of the lecture (matches ``lecture.bv_id``).
    :param query: Free-form natural-language question.
    :param top_k: Number of hits to return (default 5, max 20).
    :returns: ``SearchLectureOutput`` with deterministically ordered hits.
    """
    args = SearchLectureInput(bv=bv, query=query, top_k=top_k)
    try:
        rows = await ctx.rag.search(args.query, bv_id=args.bv, top_k=args.top_k)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("internal", f"RAG search failed: {exc}") from exc
    try:
        lecture = await _load_lecture(ctx, args.bv)
    except ToolError:
        return SearchLectureOutput(chunks=[SearchHit(**r) for r in rows])
    fallback_rows = _lecture_evidence_fallback_rows(
        lecture,
        query=args.query,
        top_k=args.top_k,
        seen_chunk_ids={int(r["chunk_id"]) for r in rows if r.get("chunk_id") is not None},
        seen_signatures={_lecture_hit_signature(r) for r in rows},
    )
    if fallback_rows:
        rows = sorted([*rows, *fallback_rows], key=lambda r: float(r.get("score") or 0.0), reverse=True)[
            : args.top_k
        ]
    return SearchLectureOutput(chunks=[SearchHit(**r) for r in rows])


def _lecture_evidence_fallback_rows(
    lecture: LectureJSON,
    *,
    query: str,
    top_k: int,
    seen_chunk_ids: set[int] | None = None,
    seen_signatures: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    seen_chunk_ids = seen_chunk_ids or set()
    seen_signatures = seen_signatures or set()
    out: list[dict[str, Any]] = []
    evidence_objects = _evidence_objects(lecture)
    if not evidence_objects:
        return []

    for obj_idx, obj in enumerate(evidence_objects):
        evidence_haystack = "\n".join(
            piece
            for piece in (obj.title, obj.summary, obj.text, obj.quote)
            if piece
        )
        note_haystack = _note_text_for_evidence(lecture, obj)
        base_haystack = "\n".join(piece for piece in (evidence_haystack, note_haystack) if piece)
        for chunk_idx, rag_chunk in enumerate(obj.rag_chunks):
            text = (rag_chunk.text or "").strip()
            if not text:
                continue
            chunk_haystack = "\n".join(piece for piece in (text, base_haystack) if piece)
            score = _score_text_match(query, chunk_haystack)
            if score <= 0:
                continue
            chunk_id = -((obj_idx + 1) * 1000 + chunk_idx + 1)
            if chunk_id in seen_chunk_ids:
                continue
            t_start = int(rag_chunk.t_start) if rag_chunk.t_start is not None else None
            t_end = int(rag_chunk.t_end) if rag_chunk.t_end is not None else t_start
            note_node_id = rag_chunk.note_node_id or (obj.note_node_ids[0] if obj.note_node_ids else "")
            meta = dict(rag_chunk.meta or {})
            meta.setdefault("evidence_id", obj.evidence_id)
            meta.setdefault("note_node_id", note_node_id)
            meta.setdefault("evidence_kind", obj.kind)
            if obj.quote and "quote" not in meta:
                meta["quote"] = obj.quote
            if obj.path and "path" not in meta:
                meta["path"] = obj.path
            signature = (obj.evidence_id, note_node_id, text)
            if signature in seen_signatures:
                continue
            out.append(
                {
                    "chunk_id": chunk_id,
                    "bv_id": lecture.bv_id,
                    "kind": rag_chunk.kind,
                    "chapter_idx": rag_chunk.chapter_idx or obj.chapter_index,
                    "t_start": t_start,
                    "t_end": t_end,
                    "text": text,
                    "quote": meta.get("quote") or "",
                    "score": score,
                    "meta": meta,
                }
            )

    out.sort(key=lambda row: row["score"], reverse=True)
    return out[:top_k]


def _lecture_hit_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return (
        str(meta.get("evidence_id") or ""),
        str(meta.get("note_node_id") or ""),
        str(row.get("text") or ""),
    )


def _note_text_for_evidence(lecture: LectureJSON, obj: EvidenceObject) -> str:
    parts: list[str] = []
    for node_id in obj.note_node_ids:
        found = _find_note_node(lecture, node_id)
        if found is None:
            continue
        unit, block = found
        parts.extend([unit.title, unit.teaching_goal, unit.core_message])
        if block is not None:
            parts.extend([block.title, block.lead, *block.paragraphs])
    return "\n".join(piece for piece in parts if piece)


async def search_evidence(
    ctx: ToolContext,
    *,
    bv: str,
    query: str,
    top_k: int = 5,
) -> SearchEvidenceOutput:
    """搜证据对象 / Search evidence objects scoped to the current lecture.

    Evidence-native lookup over ``Evidence Index``. Preferred when the
    agent already knows it needs proof objects rather than broad lecture
    chunks.
    """
    args = SearchEvidenceInput(bv=bv, query=query, top_k=top_k)
    lecture = await _load_lecture(ctx, args.bv)
    evidence_by_id = _evidence_index_map(lecture)
    if not evidence_by_id:
        return SearchEvidenceOutput(hits=[])

    ranked: dict[str, tuple[float, str]] = {}
    try:
        rows = await ctx.rag.search(args.query, bv_id=args.bv, top_k=max(10, args.top_k * 4))
    except Exception:
        rows = []
    for row in rows:
        meta = row.get("meta") if isinstance(row, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        evidence_id = str(meta.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id not in evidence_by_id:
            continue
        score = float(row.get("score") or 0.0)
        matched_text = str(row.get("text") or "")
        prev = ranked.get(evidence_id)
        if prev is None or score > prev[0]:
            ranked[evidence_id] = (score, matched_text)

    if not ranked:
        for evidence_id, obj in evidence_by_id.items():
            haystack = "\n".join(
                piece
                for piece in (obj.title, obj.summary, obj.text, obj.quote)
                if piece
            )
            score = _score_text_match(args.query, haystack)
            if score > 0:
                ranked[evidence_id] = (score, haystack[:240])

    ordered = sorted(ranked.items(), key=lambda item: item[1][0], reverse=True)[: args.top_k]
    hits = [
        SearchEvidenceHit(
            evidence_id=evidence_id,
            kind=obj.kind,
            title=obj.title,
            summary=obj.summary,
            chapter_idx=obj.chapter_index,
            t_start=int(obj.ts) if obj.ts is not None else None,
            t_end=int(obj.t_end) if obj.t_end is not None else (int(obj.ts) if obj.ts is not None else None),
            note_node_ids=list(obj.note_node_ids),
            score=score,
            matched_text=matched_text,
            anchors=dict(obj.anchors),
            quote=obj.quote,
            path=obj.path,
        )
        for evidence_id, (score, matched_text) in ordered
        for obj in [evidence_by_id[evidence_id]]
    ]
    return SearchEvidenceOutput(hits=hits)


async def get_note_unit(
    ctx: ToolContext,
    *,
    bv: str,
    unit_id: str,
) -> GetNoteUnitOutput:
    """取讲义正文节点 / Fetch one Lecture Note IR teaching unit or focused block."""
    args = GetNoteUnitInput(bv=bv, unit_id=unit_id)
    lecture = await _load_lecture(ctx, args.bv)
    found = _find_note_node(lecture, args.unit_id)
    if found is None:
        raise ToolError("not_found", f"note node {args.unit_id!r} not found in {args.bv}")
    unit, block = found
    related_evidence_ids = _related_evidence_for_note_node(lecture, args.unit_id)
    return GetNoteUnitOutput(
        bv_id=lecture.bv_id,
        requested_node_id=args.unit_id,
        resolved_node_type="content_block" if block is not None else "teaching_unit",
        unit=unit,
        focused_block=block,
        related_evidence_ids=related_evidence_ids,
        primary_ref_kind="note",
        primary_ref_id=args.unit_id,
        primary_evidence_id=related_evidence_ids[0] if related_evidence_ids else "",
    )


async def get_evidence_object(
    ctx: ToolContext,
    *,
    bv: str,
    evidence_id: str,
) -> GetEvidenceObjectOutput:
    """取证据对象 / Fetch one Evidence Index object with relations and backlinks."""
    args = GetEvidenceObjectInput(bv=bv, evidence_id=evidence_id)
    lecture = await _load_lecture(ctx, args.bv)
    evidence_by_id = _evidence_index_map(lecture)
    obj = evidence_by_id.get(args.evidence_id)
    if obj is None:
        raise ToolError("not_found", f"evidence object {args.evidence_id!r} not found in {args.bv}")
    evidence = lecture.evidence_index
    relations = [
        rel
        for rel in (evidence.evidence_relations if evidence is not None else [])
        if rel.from_id == args.evidence_id or rel.to_id == args.evidence_id
    ]
    return GetEvidenceObjectOutput(
        bv_id=lecture.bv_id,
        evidence=obj,
        relations=relations,
        related_note_unit_ids=_note_unit_ids_for_evidence(lecture, obj),
        primary_ref_kind="evidence",
        primary_ref_id=obj.evidence_id,
        primary_note_node_id=(_note_unit_ids_for_evidence(lecture, obj) or [""])[0],
    )


async def get_chapter(
    ctx: ToolContext,
    *,
    bv: str,
    chapter_idx: int,
) -> GetChapterOutput:
    """取章节完整内容 / Fetch one chapter verbatim.

    Returns the full chapter JSON (summary, teaching_notes, points,
    pitfalls, frames, key_takeaways).  The agent calls this when the
    user explicitly anchors on a chapter card (``[Ch3]``) instead of
    asking a free-form question.

    :param bv: BV id of the lecture.
    :param chapter_idx: 1-based index into ``lecture.chapters``.
    :raises ToolError: ``not_found`` if the chapter does not exist.
    """
    args = GetChapterInput(bv=bv, chapter_idx=chapter_idx)
    lecture = await _load_lecture(ctx, args.bv)
    chapter = next((c for c in lecture.chapters if c.index == args.chapter_idx), None)
    if chapter is None:
        raise ToolError(
            "not_found",
            f"chapter_idx={args.chapter_idx} not found in {args.bv}; "
            f"valid range 1-{len(lecture.chapters)}",
        )
    flat = _flat_frames(lecture)
    flat_by_path = {f.path: (idx, ch_idx) for idx, (f, ch_idx) in enumerate(flat, start=1)}
    frame_views = [
        _frame_view(flat_by_path[f.path][0], f, args.chapter_idx)
        for f in chapter.frames
        if f.path in flat_by_path
    ]
    note_unit_id, note_unit_title = _find_note_unit_for_chapter(lecture, chapter.index)
    related_evidence_ids, related_note_node_ids = _chapter_backlinks(lecture, chapter.index)
    primary_ref_kind, primary_ref_id, primary_evidence_id = _explicit_chapter_primary_ref(
        lecture, chapter.index
    )
    primary_note_node_id, primary_note_unit_id = _primary_note_ids(
        lecture,
        related_note_node_ids=related_note_node_ids,
        primary_evidence_id=primary_evidence_id,
        primary_ref_kind=primary_ref_kind,
        primary_ref_id=primary_ref_id,
    )
    primary_note_node_type, primary_note_title = _primary_note_meta(
        lecture,
        primary_note_node_id=primary_note_node_id,
        primary_note_unit_id=primary_note_unit_id,
    )
    return GetChapterOutput(
        bv_id=lecture.bv_id,
        chapter_idx=chapter.index,
        title=chapter.title,
        start=int(chapter.start),
        end=int(chapter.end),
        summary=chapter.summary,
        learning_goal=chapter.learning_goal,
        teaching_notes=list(chapter.teaching_notes),
        points=[
            ChapterPointView(text=p.text, ts=int(p.ts), quote=p.quote)
            for p in chapter.points
        ],
        pitfalls=list(chapter.pitfalls),
        key_takeaways=list(chapter.key_takeaways),
        frames=frame_views,
        note_unit_id=note_unit_id,
        note_unit_title=note_unit_title,
        related_evidence_ids=related_evidence_ids,
        related_note_node_ids=related_note_node_ids,
        compatibility_anchor_kind="chapter",
        compatibility_anchor_id=f"Ch{chapter.index}",
        primary_ref_kind=primary_ref_kind,
        primary_ref_id=primary_ref_id,
        primary_evidence_id=primary_evidence_id,
        primary_note_node_id=primary_note_node_id,
        primary_note_unit_id=primary_note_unit_id,
        primary_note_node_type=primary_note_node_type,
        primary_note_title=primary_note_title,
        primary_evidence_kind=_primary_evidence_kind(lecture, primary_evidence_id),
    )


async def get_frame(
    ctx: ToolContext,
    *,
    bv: str,
    frame_id: int,
) -> GetFrameOutput:
    """取关键帧详情 / Fetch one keyframe by global ``[F7]`` index.

    ``frame_id`` is 1-based and matches the order in
    ``visual_evidence`` (preferred) followed by chapter-local frames
    not already seen by ``path``.  This is the same numbering the
    agent emits as ``[F12]`` anchors.

    :raises ToolError: ``not_found`` if the index is out of range.
    """
    args = GetFrameInput(bv=bv, frame_id=frame_id)
    lecture = await _load_lecture(ctx, args.bv)
    flat = _flat_frames(lecture)
    if args.frame_id > len(flat):
        raise ToolError(
            "not_found",
            f"frame_id={args.frame_id} not found in {args.bv}; valid range 1-{len(flat)}",
        )
    frame, chapter_idx = flat[args.frame_id - 1]
    related_evidence_ids, related_note_node_ids = _frame_backlinks(lecture, frame, chapter_idx)
    primary_ref_kind, primary_ref_id, primary_evidence_id = _explicit_frame_primary_ref(
        lecture, frame
    )
    primary_note_node_id, primary_note_unit_id = _primary_note_ids(
        lecture,
        related_note_node_ids=related_note_node_ids if primary_evidence_id else (),
        primary_evidence_id=primary_evidence_id,
        primary_ref_kind=primary_ref_kind,
        primary_ref_id=primary_ref_id,
    )
    primary_note_node_type, primary_note_title = _primary_note_meta(
        lecture,
        primary_note_node_id=primary_note_node_id,
        primary_note_unit_id=primary_note_unit_id,
    )
    return GetFrameOutput(
        bv_id=lecture.bv_id,
        frame_id=args.frame_id,
        ts=int(frame.ts),
        chapter_idx=chapter_idx,
        path=frame.path,
        caption=frame.caption,
        ocr_text=frame.ocr_text,
        insight=frame.insight,
        visual_type=frame.visual_type,
        importance_score=float(frame.importance_score),
        related_evidence_ids=related_evidence_ids,
        related_note_node_ids=related_note_node_ids,
        compatibility_anchor_kind="frame",
        compatibility_anchor_id=f"F{args.frame_id}",
        primary_ref_kind=primary_ref_kind,
        primary_ref_id=primary_ref_id,
        primary_evidence_id=primary_evidence_id,
        primary_note_node_id=primary_note_node_id,
        primary_note_unit_id=primary_note_unit_id,
        primary_note_node_type=primary_note_node_type,
        primary_note_title=primary_note_title,
        primary_evidence_kind=_primary_evidence_kind(lecture, primary_evidence_id),
    )


async def get_quote_context(
    ctx: ToolContext,
    *,
    bv: str,
    quote: str,
) -> GetQuoteContextOutput:
    """取字幕引文及上下文 / Locate a subtitle quote and its neighbours.

    Performs a substring match across all ``chapter.points[].quote``
    entries and returns the first hit plus up to one preceding and one
    following point in the same chapter, ordered by timestamp.  Useful
    when the user copy-pastes a sentence from the lecture and asks
    "what was being discussed here?".

    :raises ToolError: ``not_found`` if no point quote contains the substring.
    """
    args = GetQuoteContextInput(bv=bv, quote=quote)
    lecture = await _load_lecture(ctx, args.bv)
    needle = args.quote.strip()
    if not needle:
        raise ToolError("invalid_arg", "quote must be non-empty after trim")

    for ch in lecture.chapters:
        sorted_points = sorted(ch.points, key=lambda p: p.ts)
        for i, p in enumerate(sorted_points):
            if needle in p.quote:
                neighbours: list[NeighborQuoteView] = []
                if i > 0:
                    prev = sorted_points[i - 1]
                    neighbours.append(
                        NeighborQuoteView(text=prev.text, ts=int(prev.ts), quote=prev.quote)
                    )
                if i + 1 < len(sorted_points):
                    nxt = sorted_points[i + 1]
                    neighbours.append(
                        NeighborQuoteView(text=nxt.text, ts=int(nxt.ts), quote=nxt.quote)
                    )
                related_evidence_ids, related_note_node_ids = _quote_backlinks(
                    lecture, ch.index, needle
                )
                primary_ref_kind, primary_ref_id, primary_evidence_id = _explicit_quote_primary_ref(
                    lecture, ch.index, needle
                )
                primary_note_node_id, primary_note_unit_id = _primary_note_ids(
                    lecture,
                    related_note_node_ids=related_note_node_ids if primary_evidence_id else (),
                    primary_evidence_id=primary_evidence_id,
                    primary_ref_kind=primary_ref_kind,
                    primary_ref_id=primary_ref_id,
                )
                primary_note_node_type, primary_note_title = _primary_note_meta(
                    lecture,
                    primary_note_node_id=primary_note_node_id,
                    primary_note_unit_id=primary_note_unit_id,
                )
                return GetQuoteContextOutput(
                    bv_id=lecture.bv_id,
                    quote=args.quote,
                    matched_quote=p.quote,
                    t_start=int(p.ts),
                    t_end=int(p.ts),
                    chapter_idx=ch.index,
                    neighbor_quotes=neighbours,
                    related_evidence_ids=related_evidence_ids,
                    related_note_node_ids=related_note_node_ids,
                    compatibility_anchor_kind="quote",
                    compatibility_anchor_id=args.quote,
                    primary_ref_kind=primary_ref_kind,
                    primary_ref_id=primary_ref_id,
                    primary_evidence_id=primary_evidence_id,
                    primary_note_node_id=primary_note_node_id,
                    primary_note_unit_id=primary_note_unit_id,
                    primary_note_node_type=primary_note_node_type,
                    primary_note_title=primary_note_title,
                    primary_evidence_kind=_primary_evidence_kind(lecture, primary_evidence_id),
                )
    raise ToolError(
        "not_found",
        f"no point quote in {args.bv} contains the substring {args.quote!r}",
    )


async def explain_frame(
    ctx: ToolContext,
    *,
    bv: str,
    frame_id: int,
    context_radius_seconds: int = 60,
) -> ExplainFrameOutput:
    """解读关键帧 / Why-this-frame-matters using only cached metadata.

    Cheap counterpart to ``get_frame``: returns the same anchors plus a
    ``why_useful`` blurb (``frame.insight`` or ``selected_reason``) and
    the chapter points that fall within ``±context_radius_seconds`` of
    the frame's timestamp.  Does **not** call any VLM — strictly reads
    cached data so the agent can ask it without budget anxiety.
    """
    args = ExplainFrameInput(bv=bv, frame_id=frame_id)
    lecture = await _load_lecture(ctx, args.bv)
    flat = _flat_frames(lecture)
    if args.frame_id > len(flat):
        raise ToolError(
            "not_found",
            f"frame_id={args.frame_id} not found in {args.bv}; valid range 1-{len(flat)}",
        )
    frame, chapter_idx = flat[args.frame_id - 1]
    why = frame.insight.strip() or frame.selected_reason.strip() or frame.caption.strip()
    related_evidence_ids, related_note_node_ids = _frame_backlinks(lecture, frame, chapter_idx)
    primary_ref_kind, primary_ref_id, primary_evidence_id = _explicit_frame_primary_ref(
        lecture, frame
    )
    primary_note_node_id, primary_note_unit_id = _primary_note_ids(
        lecture,
        related_note_node_ids=related_note_node_ids if primary_evidence_id else (),
        primary_evidence_id=primary_evidence_id,
        primary_ref_kind=primary_ref_kind,
        primary_ref_id=primary_ref_id,
    )
    primary_note_node_type, primary_note_title = _primary_note_meta(
        lecture,
        primary_note_node_id=primary_note_node_id,
        primary_note_unit_id=primary_note_unit_id,
    )

    context_quotes: list[NeighborQuoteView] = []
    if chapter_idx is not None:
        chapter = next((c for c in lecture.chapters if c.index == chapter_idx), None)
        if chapter is not None:
            ts = int(frame.ts)
            lo, hi = ts - max(0, context_radius_seconds), ts + max(0, context_radius_seconds)
            nearby = [p for p in chapter.points if lo <= p.ts <= hi]
            nearby.sort(key=lambda p: abs(p.ts - ts))
            for p in nearby[:3]:
                context_quotes.append(
                    NeighborQuoteView(text=p.text, ts=int(p.ts), quote=p.quote)
                )

    return ExplainFrameOutput(
        bv_id=lecture.bv_id,
        frame_id=args.frame_id,
        ts=int(frame.ts),
        chapter_idx=chapter_idx,
        path=frame.path,
        caption=frame.caption,
        ocr_text=frame.ocr_text,
        why_useful=why,
        context_quotes=context_quotes,
        related_evidence_ids=related_evidence_ids,
        related_note_node_ids=related_note_node_ids,
        compatibility_anchor_kind="frame",
        compatibility_anchor_id=f"F{args.frame_id}",
        primary_ref_kind=primary_ref_kind,
        primary_ref_id=primary_ref_id,
        primary_evidence_id=primary_evidence_id,
        primary_note_node_id=primary_note_node_id,
        primary_note_unit_id=primary_note_unit_id,
        primary_note_node_type=primary_note_node_type,
        primary_note_title=primary_note_title,
        primary_evidence_kind=_primary_evidence_kind(lecture, primary_evidence_id),
    )


# ---------------------------------------------------------------------------
# MCP-only tools
# ---------------------------------------------------------------------------


async def summarize_video(
    ctx: ToolContext,
    *,
    url: str,
    force_refresh: bool = False,
) -> SummarizeVideoOutput:
    """摘要 Bilibili 视频 / Run the LectureMind pipeline on a URL.

    Forwards to the existing :class:`Pipeline.run`.  When the BV is
    already cached and ``force_refresh`` is False, returns the cached
    report metadata in O(1).  Disabled by default in the MCP server
    (``MCP_EXPOSE_SUMMARIZE=false``); guarded by ``ctx.pipeline``.

    :raises ToolError: ``unsupported`` when no pipeline is wired into the context.
    """
    args = SummarizeVideoInput(url=url, force_refresh=force_refresh)
    if ctx.pipeline is None:
        raise ToolError(
            "unsupported",
            "summarize_video requires a Pipeline-bound ToolContext "
            "(MCP server is running without the FastAPI process).",
        )
    try:
        report_path = await ctx.pipeline.run(args.url, force_refresh=args.force_refresh)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("internal", f"pipeline failed: {exc}") from exc

    # Resolve BV from path or URL — pipeline.run always names the report
    # ``<bv>.html`` so the stem is authoritative.
    bv = report_path.stem
    row = await ctx.db.get_summary(bv)
    taxonomy: dict[str, Any] | None = None
    if row is not None and row.summary_json:
        tax = row.summary_json.get("taxonomy") if isinstance(row.summary_json, dict) else None
        if isinstance(tax, dict):
            taxonomy = tax
    return SummarizeVideoOutput(
        bv=bv,
        report_url=f"/reports/{bv}.html",
        status="done",
        taxonomy=taxonomy,
    )


async def search_lectures(
    ctx: ToolContext,
    *,
    query: str,
    top_k: int = 5,
    domain: str | None = None,
    direction: str | None = None,
) -> SearchLecturesOutput:
    """跨视频 RAG 检索 / Hybrid retrieval across the whole library.

    Same retrieval engine as ``search_lecture`` but unscoped, with
    optional taxonomy filters applied **post-hoc** against
    ``summaries.domain`` / ``summaries.direction``.  When both filters
    match no rows, returns an empty list (not an error) so the calling
    Agent can decide how to communicate it.
    """
    args = SearchLecturesInput(
        query=query, top_k=top_k, domain=domain, direction=direction
    )
    try:
        rows = await ctx.rag.search(args.query, bv_id=None, top_k=args.top_k * 3 if (args.domain or args.direction) else args.top_k)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("internal", f"RAG search failed: {exc}") from exc

    if args.domain or args.direction:
        # Build a (bv → taxonomy) lookup once instead of N round-trips.
        bv_ids = {r["bv_id"] for r in rows}
        tax_map: dict[str, tuple[str | None, str | None]] = {}
        for bv in bv_ids:
            row = await ctx.db.get_summary(bv)
            if row is not None:
                tax_map[bv] = (row.domain, row.direction)
        kept: list[dict[str, Any]] = []
        for r in rows:
            d, di = tax_map.get(r["bv_id"], (None, None))
            if args.domain and d != args.domain:
                continue
            if args.direction and di != args.direction:
                continue
            kept.append(r)
        rows = kept[: args.top_k]

    return SearchLecturesOutput(chunks=[SearchHit(**r) for r in rows])


async def web_search(
    ctx: ToolContext,
    *,
    query: str,
    count: int = 5,
) -> WebSearchOutput:
    """联网搜索 / Search the public web when the user explicitly enables web access.

    This tool is intentionally graceful: missing API key, network errors,
    unexpected response shapes, and upstream 5xx all return an ``error``
    object inside the tool result instead of raising.  The Agent can then
    continue with lecture evidence and non-web learning supplements.
    """
    _ = ctx
    args = WebSearchInput(query=query, count=count)
    settings = get_settings()
    if not settings.bocha_api_key:
        return WebSearchOutput(
            query=args.query,
            error={"code": "missing_api_key", "message": "BOCHA_API_KEY is not configured."},
        )
    try:
        import httpx

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                settings.bocha_base_url,
                headers={
                    "Authorization": f"Bearer {settings.bocha_api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": args.query, "count": args.count},
            )
        if resp.status_code >= 400:
            return WebSearchOutput(
                query=args.query,
                error={"code": "upstream_error", "message": f"Bocha returned HTTP {resp.status_code}."},
            )
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return WebSearchOutput(
            query=args.query,
            error={"code": "request_failed", "message": str(exc)[:200]},
        )

    raw_results = _extract_web_results(data)
    results: list[WebSearchResult] = []
    for item in raw_results[: args.count]:
        url = str(item.get("url") or item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("summary") or item.get("content") or "").strip()
        hostname = urlparse(url).hostname or ""
        results.append(WebSearchResult(title=title, url=url, hostname=hostname, snippet=snippet))
    return WebSearchOutput(query=args.query, results=results)


def _extract_web_results(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("results", "web_results"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        inner = data.get("data")
        if isinstance(inner, dict):
            return _extract_web_results(inner)
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


async def get_knowledge_units(
    ctx: ToolContext,
    *,
    bv: str,
    kind: str | None = None,
) -> GetKnowledgeUnitsOutput:
    """取知识单元 / Pull structured knowledge units for fact extraction.

    External agents (Cursor / Claude / ChatGPT Apps) use this to mine
    discrete facts (concept / procedure / formula / pitfall …) without
    re-parsing HTML.  The optional ``kind`` filter is matched
    case-insensitively against ``KnowledgeUnitView.type``.
    """
    args = GetKnowledgeUnitsInput(bv=bv, kind=kind)
    lecture = await _load_lecture(ctx, args.bv)
    units: list[KnowledgeUnitOutItem] = []
    kind_lc = args.kind.lower() if args.kind else None
    for ku in lecture.knowledge_units:
        if kind_lc and ku.type.lower() != kind_lc:
            continue
        ts = int(ku.ts)
        units.append(
            KnowledgeUnitOutItem(
                id=ku.id,
                kind=ku.type,
                title=ku.title,
                explanation=ku.explanation,
                t_start=ts,
                t_end=ts,
                chapter_idx=ku.chapter_index,
                quote=ku.quote,
            )
        )
    return GetKnowledgeUnitsOutput(bv_id=lecture.bv_id, units=units)


async def get_frame_description(
    ctx: ToolContext,
    *,
    bv: str,
    frame_id: int,
) -> GetFrameOutput:
    """关键帧描述（MCP 别名）/ Public alias for ``get_frame``.

    Kept as a separate symbol so the MCP-facing schema can document
    external usage independently from the in-process Copilot path; the
    behaviour is byte-identical.
    """
    return await get_frame(ctx, bv=bv, frame_id=frame_id)


async def list_lectures(
    ctx: ToolContext,
    *,
    limit: int = 50,
    domain: str | None = None,
    direction: str | None = None,
) -> ListLecturesOutput:
    """枚举讲义元信息 / List lecture metadata, optionally filtered by taxonomy.

    The MCP equivalent of the FastAPI homepage.  Runs against the live
    ``summaries`` table; filtering is in-memory because the row count
    is expected to stay small (<1000) and it lets us drop the dependency
    on a new SQL DAO for Stage 3.
    """
    args = ListLecturesInput(limit=limit, domain=domain, direction=direction)
    rows = await ctx.db.list_summaries(limit=10000)
    out: list[LectureMetaItem] = []
    for row in rows:
        if args.domain and (row.domain or "") != args.domain:
            continue
        if args.direction and (row.direction or "") != args.direction:
            continue
        out.append(_summary_to_meta(row))
        if len(out) >= args.limit:
            break
    return ListLecturesOutput(lectures=out)


# ---------------------------------------------------------------------------
# Public registry — exposed for Stage 4 / Stage 7
# ---------------------------------------------------------------------------


COPILOT_TOOLS: tuple[str, ...] = (
    "search_lecture",
    "search_evidence",
    "search_lectures",
    "get_note_unit",
    "get_evidence_object",
    "get_chapter",
    "get_frame",
    "get_quote_context",
    "explain_frame",
)

MCP_TOOLS: tuple[str, ...] = (
    "summarize_video",
    "search_evidence",
    "search_lectures",
    "get_note_unit",
    "get_evidence_object",
    "get_chapter",
    "get_frame",
    "get_quote_context",
    "explain_frame",
    "get_knowledge_units",
    "list_lectures",
)
