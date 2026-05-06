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

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.storage.db import Database, SummaryRow
from app.understand.schema import (
    Chapter,
    Frame,
    KnowledgeUnitView,
    LectureJSON,
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


class SearchLectureOutput(BaseModel):
    chunks: list[SearchHit]


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
    return SearchLectureOutput(chunks=[SearchHit(**r) for r in rows])


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
                return GetQuoteContextOutput(
                    bv_id=lecture.bv_id,
                    quote=args.quote,
                    matched_quote=p.quote,
                    t_start=int(p.ts),
                    t_end=int(p.ts),
                    chapter_idx=ch.index,
                    neighbor_quotes=neighbours,
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
    "search_lectures",
    "get_chapter",
    "get_frame",
    "get_quote_context",
    "explain_frame",
)

MCP_TOOLS: tuple[str, ...] = (
    "summarize_video",
    "search_lectures",
    "get_chapter",
    "get_knowledge_units",
    "get_frame_description",
    "list_lectures",
)
