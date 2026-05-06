"""FastAPI routes for the Copilot (``/api/copilot/*``).

Four endpoints, all under HTTP Basic auth:

* ``POST /ask`` — streams the LangGraph agent as ``text/event-stream``.
  Event types match the Stage 5 spec: ``token`` / ``tool_call`` /
  ``tool_result`` / ``done`` / ``error``.  The SSE generator also runs
  :func:`app.copilot.agent.validate_anchors` on the full answer and
  emits warnings + a ``patched_answer`` in the ``done`` event so the
  front-end can swap invalid ``[t=…]`` / ``[F…]`` / ``[Ch…]`` anchors.
* ``GET /lectures`` — compact lecture metadata for the Stage 8
  folding tree; includes taxonomy and frame counts.
* ``POST /reindex`` — admin trigger for the Stage 2 indexer
  (``bv`` / ``all`` / ``taxonomy_only``); long-running modes run in a
  ``BackgroundTask``.
* ``POST /taxonomy`` — manual re-grouping; patches
  ``summaries.domain/direction/domain_tags`` only.

Concurrency controls (see D38 in the handoff):

* Global ``asyncio.Semaphore(settings.copilot_max_concurrent)`` shared
  across the process.
* Per-BV ``asyncio.Lock`` — at most one in-flight ``/ask`` per BV to
  prevent the same lecture from blowing embedding quota.
* Both live on ``app.state.copilot_*``; they are created lazily on
  first use so ``create_app`` stays lean.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app.auth import require_auth
from app.config import get_settings
from app.storage.db import Database

from . import agent as agent_mod
from . import tools as tool_registry
from .rag import RAGStore
from .taxonomy import normalize as normalize_taxonomy
from .tools import ToolContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/copilot", dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class Reference(BaseModel):
    kind: str
    id: int | None = None
    text: str | None = None


class HistoryTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    bv: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    references: list[Reference] = Field(default_factory=list)
    use_web: bool = Field(default=False)
    messages: list[HistoryTurn] | None = Field(
        default=None,
        description="Optional multi-turn history; capped at 10 turns by the front-end.",
    )


class LectureMeta(BaseModel):
    bv: str
    title: str = ""
    author: str = ""
    duration: int = 0
    cover_url: str = ""
    domain: str | None = None
    direction: str | None = None
    tags: list[str] = Field(default_factory=list)
    updated_at: str = ""


class ReindexRequest(BaseModel):
    bv: str | None = None
    all: bool = False
    taxonomy_only: bool = False


class ReindexResponse(BaseModel):
    ok: bool = True
    mode: str
    bv: str | None = None
    detail: str = ""


class TaxonomyPatch(BaseModel):
    bv: str = Field(..., min_length=1)
    domain: str | None = None
    direction: str | None = None
    tags: list[str] | None = None


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_db(request: Request) -> Database:
    return request.app.state.db


def _get_rag(request: Request) -> RAGStore | None:
    return getattr(request.app.state, "rag", None)


def _get_pipeline(request: Request) -> Any | None:
    return getattr(request.app.state, "pipeline", None)


def _get_tool_context(request: Request) -> ToolContext:
    return ToolContext(
        db=_get_db(request),
        rag=_get_rag(request),  # type: ignore[arg-type]
        pipeline=_get_pipeline(request),
    )


def _get_global_semaphore(request: Request) -> asyncio.Semaphore:
    state = request.app.state
    sem = getattr(state, "copilot_semaphore", None)
    if sem is None:
        settings = get_settings()
        sem = asyncio.Semaphore(settings.copilot_max_concurrent)
        state.copilot_semaphore = sem
    return sem


def _get_bv_lock(request: Request, bv: str) -> asyncio.Lock:
    state = request.app.state
    locks: dict[str, asyncio.Lock] = getattr(state, "copilot_bv_locks", None)  # type: ignore[assignment]
    if locks is None:
        locks = {}
        state.copilot_bv_locks = locks
    lock = locks.get(bv)
    if lock is None:
        lock = asyncio.Lock()
        locks[bv] = lock
    return lock


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    """Serialise a single SSE event to bytes.

    Kept tiny on purpose so tests can call it directly without touching
    a FastAPI client.  ``json.dumps`` uses ``ensure_ascii=False`` to
    keep CJK readable on the wire; the MIME type (``text/event-stream``)
    does not mandate ASCII.
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _extract_chunk_text(chunk: Any) -> str:
    """Pull plaintext from an ``AIMessageChunk`` (or dict-shaped equivalent).

    LangChain can return ``.content`` as either ``str`` or a list of
    ``{"type": "text"|"tool_use", ...}`` dicts; the latter shows up when
    models stream tool calls alongside narration.  We drop non-text
    parts so tool-call JSON never leaks into the ``token`` stream.
    """
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if content is None and isinstance(chunk, dict):
        content = chunk.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if isinstance(p, dict) and p.get("type") == "text":
                text = p.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    message = getattr(chunk, "message", None)
    if message is not None:
        return _extract_chunk_text(message)
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    generations = getattr(chunk, "generations", None)
    if generations is None and isinstance(chunk, dict):
        generations = chunk.get("generations")
    if generations:
        parts: list[str] = []
        for group in generations:
            items = group if isinstance(group, list) else [group]
            for item in items:
                text = _extract_chunk_text(item)
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)
    if isinstance(chunk, dict):
        for key in ("message", "output", "chunk"):
            if key in chunk:
                text = _extract_chunk_text(chunk[key])
                if text:
                    return text
    return ""


def _strip_thinking_tags(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped in {"<think>", "</think>", "<think></think>"}:
        return ""
    return text.replace("<think>", "").replace("</think>", "")


def _classify_agent_error(exc: Exception) -> tuple[str, str]:
    text = str(exc)
    lowered = text.lower()
    if (
        "allocationquota.freetieronly" in lowered
        or "free tier" in lowered
        or "quota" in lowered
        or "额度" in text
    ):
        return (
            "quota_exhausted",
            "DashScope/Qwen model quota is exhausted. Disable free-tier-only mode or switch QWEN_COPILOT_MODEL to a model with available quota.",
        )
    if "error code: 403" in lowered or "http 403" in lowered:
        return (
            "upstream_forbidden",
            "DashScope/Qwen returned 403. Check DASHSCOPE_API_KEY permissions, billing mode, and QWEN_COPILOT_MODEL quota.",
        )
    return "internal", text[:500]


def _summarise_tool_output(name: str, output: Any) -> dict[str, Any]:
    """Produce a compact ``tool_result`` payload for the client.

    We never stream the full tool response over SSE — it can be large
    (e.g. a full chapter) and the front-end does not need it for the
    progress UI.  The payload below is just enough to say "search_lecture
    found 3 chunks" or "get_chapter found chapter 2".
    """
    base: dict[str, Any] = {"name": name}
    if isinstance(output, dict):
        if "error" in output:
            base.update({"ok": False, "error": output["error"]})
            return base
        if "chunks" in output and isinstance(output["chunks"], list):
            base["chunks_count"] = len(output["chunks"])
        if "chapter_idx" in output:
            base["chapter_idx"] = output["chapter_idx"]
        if "frame_id" in output:
            base["frame_id"] = output["frame_id"]
        if "units" in output and isinstance(output["units"], list):
            base["units_count"] = len(output["units"])
    base.setdefault("ok", True)
    return base


async def run_agent_sse(
    graph: Any,
    state: dict[str, Any],
    ctx: ToolContext,
    bv: str,
) -> AsyncIterator[bytes]:
    """Async generator yielding SSE frames for one ``/ask`` turn.

    Exposed (not leading-underscore) so unit tests can exercise it with
    a stub ``graph`` directly and so Stage 7's MCP server can reuse the
    same stream-to-event transformation if we ever need an HTTP bridge.
    """
    full_text: list[str] = []
    try:
        async for ev in graph.astream_events(state, version="v2"):
            kind = ev.get("event")
            data = ev.get("data") or {}
            if kind == "on_chat_model_stream":
                text = _strip_thinking_tags(_extract_chunk_text(data.get("chunk")))
                if text:
                    full_text.append(text)
                    yield _sse_frame("token", {"delta": text})
            elif kind == "on_tool_start":
                yield _sse_frame(
                    "tool_call",
                    {"name": ev.get("name", ""), "args": data.get("input") or {}},
                )
            elif kind == "on_tool_end":
                yield _sse_frame(
                    "tool_result",
                    _summarise_tool_output(ev.get("name", ""), data.get("output")),
                )
            elif kind == "on_chat_model_end":
                # Non-streaming test stubs land here with the complete
                # AIMessage.  Only synthesise a token event if we saw no
                # ``on_chat_model_stream`` — otherwise we'd double-emit.
                if not full_text:
                    text = _strip_thinking_tags(_extract_chunk_text(data.get("output")))
                    if text:
                        full_text.append(text)
                        yield _sse_frame("token", {"delta": text})
    except asyncio.CancelledError:
        logger.info("copilot SSE cancelled for bv=%s", bv)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("copilot agent failed for bv=%s", bv)
        code, message = _classify_agent_error(exc)
        yield _sse_frame("error", {"code": code, "message": message})
        return

    answer = "".join(full_text)
    try:
        patched, warnings = await agent_mod.validate_anchors(
            answer, bv, ctx.db, ctx.rag
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("validate_anchors failed for bv=%s: %s", bv, exc)
        patched, warnings = answer, []

    yield _sse_frame(
        "done",
        {
            "anchors_validated": True,
            "warnings": warnings,
            "patched_answer": patched if patched != answer else None,
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/ask")
async def copilot_ask(
    body: AskRequest,
    request: Request,
    ctx: Annotated[ToolContext, Depends(_get_tool_context)],
    sem: Annotated[asyncio.Semaphore, Depends(_get_global_semaphore)],
) -> StreamingResponse:
    """Stream the Copilot agent as ``text/event-stream``.

    Flow:

    1. Build initial state (raises 404 if the BV is unknown).
    2. Acquire global semaphore + per-BV lock.
    3. Compile a fresh graph (see D32) and iterate its events.
    4. After the agent returns, run anchor validation and emit ``done``.

    We intentionally do the ``build_initial_state`` call *before*
    grabbing the locks — it is cheap, and surfacing a 404 for an
    unknown BV should not block other BVs from running.
    """
    try:
        state = await agent_mod.build_initial_state(
            ctx,
            body.bv,
            body.question,
            [r.model_dump() for r in body.references],
            history=[m.model_dump() for m in (body.messages or [])][-10:],
        )
    except tool_registry.ToolError as exc:
        if exc.code == "not_found":
            raise HTTPException(status_code=404, detail=exc.message) from exc
        raise HTTPException(status_code=500, detail=exc.message) from exc

    bv_lock = _get_bv_lock(request, body.bv)

    async def _gen() -> AsyncIterator[bytes]:
        async with sem:
            async with bv_lock:
                graph = agent_mod.build_graph(ctx, enable_web=body.use_web)
                async for frame in run_agent_sse(graph, state, ctx, body.bv):
                    yield frame

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # hint to nginx: do not buffer SSE
        },
    )


@router.get("/lectures", response_model=list[LectureMeta])
async def copilot_lectures(
    db: Annotated[Database, Depends(_get_db)],
) -> list[LectureMeta]:
    rows = await db.list_summaries(limit=500, offset=0)
    out: list[LectureMeta] = []
    for r in rows:
        if r.status != "done":
            continue
        out.append(
            LectureMeta(
                bv=r.bv_id,
                title=r.title or "",
                author=r.author or "",
                duration=int(r.duration or 0),
                cover_url=r.cover_url or "",
                domain=r.domain,
                direction=r.direction,
                tags=list(r.domain_tags or []),
                updated_at=str(r.updated_at or ""),
            )
        )
    return out


@router.post("/reindex", response_model=ReindexResponse)
async def copilot_reindex(
    body: ReindexRequest,
    bg: BackgroundTasks,
    ctx: Annotated[ToolContext, Depends(_get_tool_context)],
) -> ReindexResponse:
    """Trigger Stage 2 indexer without going through the CLI.

    For ``taxonomy_only`` the work is O(N) column updates, so we do it
    inline.  For full/partial reindex we offload to ``BackgroundTasks``
    because each BV is ~80 embedding API calls.
    """
    if ctx.rag is None:
        raise HTTPException(status_code=503, detail="RAGStore not initialised")

    if body.taxonomy_only:
        # Walk existing summaries, re-normalise and patch taxonomy columns.
        rows = await ctx.db.list_summaries(limit=10000, offset=0)
        from app.understand.schema import LectureJSON, Taxonomy

        touched = 0
        for r in rows:
            if not r.summary_json:
                continue
            try:
                lec = LectureJSON.model_validate(r.summary_json)
            except Exception:  # noqa: BLE001
                continue
            if lec.taxonomy is None:
                continue
            new_tax = normalize_taxonomy(lec.taxonomy)
            if new_tax is None:
                continue
            await ctx.db.update_taxonomy(
                r.bv_id,
                domain=new_tax.domain or None,
                direction=new_tax.direction or None,
                domain_tags=list(new_tax.tags or []),
            )
            touched += 1
        return ReindexResponse(
            ok=True, mode="taxonomy_only", detail=f"patched {touched} rows"
        )

    # Full / single-BV reindex — offload to background.
    from .indexer import index_lecture as _index_lecture

    async def _run_all() -> None:
        from app.understand.schema import LectureJSON

        rows = await ctx.db.list_summaries(limit=10000, offset=0)
        for r in rows:
            if not r.summary_json:
                continue
            try:
                lec = LectureJSON.model_validate(r.summary_json)
            except Exception:  # noqa: BLE001
                continue
            try:
                await _index_lecture(ctx.rag, r.bv_id, lec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reindex %s failed: %s", r.bv_id, exc)

    async def _run_single(bv: str) -> None:
        from app.understand.schema import LectureJSON

        row = await ctx.db.get_summary(bv)
        if row is None or not row.summary_json:
            logger.warning("reindex: bv %s not found", bv)
            return
        try:
            lec = LectureJSON.model_validate(row.summary_json)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reindex: lecture %s corrupt: %s", bv, exc)
            return
        try:
            await _index_lecture(ctx.rag, bv, lec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reindex %s failed: %s", bv, exc)

    if body.all:
        bg.add_task(_run_all)
        return ReindexResponse(ok=True, mode="all", detail="queued")
    if body.bv:
        bg.add_task(_run_single, body.bv)
        return ReindexResponse(ok=True, mode="single", bv=body.bv, detail="queued")
    raise HTTPException(
        status_code=400,
        detail="must specify one of: bv, all, taxonomy_only",
    )


@router.post("/taxonomy")
async def copilot_taxonomy(
    body: TaxonomyPatch,
    db: Annotated[Database, Depends(_get_db)],
) -> dict[str, Any]:
    row = await db.get_summary(body.bv)
    if row is None:
        raise HTTPException(status_code=404, detail=f"bv {body.bv} not found")
    try:
        ok = await db.update_taxonomy(
            body.bv,
            domain=body.domain,
            direction=body.direction,
            domain_tags=list(body.tags) if body.tags is not None else None,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": bool(ok), "bv": body.bv}
