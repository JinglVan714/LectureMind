"""FastMCP server exposing LectureMind tools to external Agents.

The server is a thin wrapper around Stage 3's ``app.copilot.tools``
registry — no new business logic lives here.  Cursor / Claude Desktop /
ChatGPT Apps discover tools via MCP and call them through stdio (local)
or HTTP+SSE (remote).

Key decisions (see handoff_copilot.md D48-D51):

* **Independent ToolContext.**  The server builds its own
  ``Database`` + ``RAGStore`` instance rather than reaching into
  ``app.state``; that way you can launch it as ``python
  scripts/run_mcp_server.py`` without bringing up the FastAPI process.
* **``summarize_video`` hidden by default.**  Registered only when
  ``settings.mcp_expose_summarize`` (``MCP_EXPOSE_SUMMARIZE=true``) —
  it otherwise stays invisible in ``list_tools``.  When enabled, it
  requires a live ``Pipeline``, which the MCP CLI can opt in to.
* **Errors surface as ``ValueError``** (fastmcp serialises the message
  into a structured MCP error).  Internal ``ToolError`` is formatted
  ``[code] message`` so Agents can still parse the code prefix.  This
  is the flipside of Stage 4's ``{"error": {...}}`` dict fallback —
  external protocols get explicit error envelopes, in-process LangGraph
  gets tolerant dicts.
* **Lifecycle bound to server instance.**  `build_mcp` returns an
  ``FastMCP`` with a ``lifespan`` hook that initialises the DB+RAG
  eagerly and closes them on shutdown.  Both stdio and HTTP servers
  share the same lifespan.

The public surface is just :func:`build_mcp` and :func:`build_default_context`.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from app.config import get_settings
from app.runtime.observe import create_run_record, emit_artifact, issue_verdict, write_run_record
from app.storage.db import Database

from . import tools as tool_registry
from .rag import RAGStore
from .tools import ToolContext, ToolError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default context (independent of FastAPI)
# ---------------------------------------------------------------------------


async def build_default_context(
    *, with_pipeline: bool = False
) -> tuple[ToolContext, Database, RAGStore]:
    """Open a fresh DB + RAG pair rooted at ``settings.db_path``.

    Returns a tuple so the caller (lifespan or tests) can call
    ``await db.close()`` / ``await rag.close()`` when done.  Pipeline
    creation is gated by ``with_pipeline`` because it imports the full
    ingest/render/understand stack — we do not want to pay that cost
    when the MCP client is only doing read-only queries.
    """
    settings = get_settings()
    db = Database(settings.db_path)
    await db.init()
    rag = RAGStore(settings.db_path)
    try:
        await rag.init()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAGStore init failed in MCP server (%s); search tools will return empty.",
            exc,
        )

    pipeline: Any | None = None
    if with_pipeline:
        # Lazy import to avoid pulling in Whisper / yt-dlp at module load.
        from app.pipeline import Pipeline as _Pipeline

        pipeline = _Pipeline(db)

    return ToolContext(db=db, rag=rag, pipeline=pipeline), db, rag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe(fn: Any) -> str:
    doc = (fn.__doc__ or "").strip().splitlines()
    return doc[0] if doc else fn.__name__


def _handle_tool_error(exc: ToolError) -> None:
    """Translate ``ToolError`` to a fastmcp-friendly exception.

    fastmcp v3 serialises raised exceptions into a structured error
    envelope; prefixing ``[code]`` lets the calling Agent extract the
    machine-readable code from the surface text without us having to
    register a custom error type.
    """
    raise ValueError(f"[{exc.code}] {exc.message}") from exc


def _dump(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a plain JSON-friendly dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"cannot serialise {type(obj).__name__} for MCP")


def _target_ref(tool_name: str, kwargs: dict[str, Any]) -> str:
    for key in ("bv", "url", "query", "unit_id", "evidence_id", "frame_id", "chapter_idx"):
        value = kwargs.get(key)
        if value not in (None, ""):
            return str(value)
    return tool_name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mcp(
    ctx: ToolContext | None = None,
    *,
    expose_summarize: bool | None = None,
    name: str = "lecturemind",
    instructions: str | None = None,
) -> FastMCP:
    """Assemble the FastMCP server with ``ctx`` captured in closures.

    Passing ``ctx=None`` (production path) wires the server to a fresh
    DB+RAG pair via :func:`build_default_context` on ``lifespan`` start
    and tears them down on stop.  Tests pass in a pre-built ``ctx``
    backed by ``tmp_path`` so we never touch the real DB.
    """
    settings = get_settings()
    if expose_summarize is None:
        expose_summarize = settings.mcp_expose_summarize

    # When the caller supplied ctx we do not own the DB/RAG — no
    # lifespan cleanup required.  Otherwise we manage them.
    owned_ctx = ctx is None

    state: dict[str, Any] = {"ctx": ctx, "db": None, "rag": None}

    @asynccontextmanager
    async def _lifespan(_server: FastMCP):  # noqa: ARG001
        if owned_ctx:
            built_ctx, db, rag = await build_default_context(
                with_pipeline=expose_summarize
            )
            state["ctx"] = built_ctx
            state["db"] = db
            state["rag"] = rag
        try:
            yield
        finally:
            if owned_ctx:
                try:
                    if state["rag"] is not None:
                        await state["rag"].close()
                except Exception:  # noqa: BLE001
                    pass

    mcp = FastMCP(
        name,
        instructions=(
            instructions
            or "LectureMind MCP server. 对已处理的 Bilibili 讲义做结构化检索、"
            "章节与关键帧定位、知识单元抽取。所有工具只读缓存，不会触发昂贵的 VLM "
            "或 embedding 调用（summarize_video 除外，且默认不暴露）。"
        ),
        lifespan=_lifespan,
    )

    def _ctx() -> ToolContext:
        c = state["ctx"]
        if c is None:
            raise RuntimeError(
                "MCP ToolContext not initialised; did you call run()/run_async()?"
            )
        return c

    async def _run_tool(tool_name: str, fn: Any, **kwargs: Any) -> dict[str, Any]:
        record = create_run_record(
            run_type="mcp_tool_request",
            entrypoint="mcp",
            target_ref=_target_ref(tool_name, kwargs),
            expected_artifacts=["mcp_result"],
            expected_profile=None,
            writeback_allowed=False,
            minimum_acceptance=["tool_schema_returned"],
        )
        try:
            out = await fn(_ctx(), **kwargs)
            payload = _dump(out)
            emit_artifact(
                record,
                artifact_type="mcp_result",
                path_or_ref=f"mcp://{tool_name}/response",
                producer=tool_name,
            )
            record.verdict = issue_verdict(
                status="accept",
                reasons=["tool_schema_returned"],
                warnings=[],
                next_actions=[],
                evidence_refs=[f"mcp://{tool_name}/response"],
            )
            record.status = "finished"
            return payload
        except ToolError as exc:
            record.verdict = issue_verdict(
                status="block",
                reasons=[f"{exc.code}: {exc.message}"],
                warnings=[],
                next_actions=[],
                evidence_refs=[],
            )
            record.status = "blocked"
            _handle_tool_error(exc)
            raise
        except Exception as exc:  # noqa: BLE001
            record.verdict = issue_verdict(
                status="block",
                reasons=[type(exc).__name__],
                warnings=[],
                next_actions=[],
                evidence_refs=[],
            )
            record.status = "blocked"
            raise
        finally:
            try:
                write_run_record(record, get_settings().data_dir / "debug")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "write_run_record failed for mcp tool=%s: %s", tool_name, exc
                )

    # ---- Tool wrappers ---------------------------------------------------

    @mcp.tool(
        name="search_lectures",
        description=_describe(tool_registry.search_lectures),
    )
    async def search_lectures(
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        direction: str | None = None,
    ) -> dict[str, Any]:
        return await _run_tool(
            "search_lectures",
            tool_registry.search_lectures,
            query=query,
            top_k=top_k,
            domain=domain,
            direction=direction,
        )

    @mcp.tool(
        name="search_evidence",
        description=_describe(tool_registry.search_evidence),
    )
    async def search_evidence(
        bv: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return await _run_tool(
            "search_evidence",
            tool_registry.search_evidence,
            bv=bv,
            query=query,
            top_k=top_k,
        )

    @mcp.tool(
        name="get_note_unit",
        description=_describe(tool_registry.get_note_unit),
    )
    async def get_note_unit(bv: str, unit_id: str) -> dict[str, Any]:
        return await _run_tool(
            "get_note_unit", tool_registry.get_note_unit, bv=bv, unit_id=unit_id
        )

    @mcp.tool(
        name="get_evidence_object",
        description=_describe(tool_registry.get_evidence_object),
    )
    async def get_evidence_object(bv: str, evidence_id: str) -> dict[str, Any]:
        return await _run_tool(
            "get_evidence_object",
            tool_registry.get_evidence_object,
            bv=bv,
            evidence_id=evidence_id,
        )

    @mcp.tool(
        name="get_chapter",
        description=_describe(tool_registry.get_chapter),
    )
    async def get_chapter(bv: str, chapter_idx: int) -> dict[str, Any]:
        return await _run_tool(
            "get_chapter", tool_registry.get_chapter, bv=bv, chapter_idx=chapter_idx
        )

    @mcp.tool(
        name="get_frame",
        description=_describe(tool_registry.get_frame),
    )
    async def get_frame(bv: str, frame_id: int) -> dict[str, Any]:
        return await _run_tool("get_frame", tool_registry.get_frame, bv=bv, frame_id=frame_id)

    @mcp.tool(
        name="get_quote_context",
        description=_describe(tool_registry.get_quote_context),
    )
    async def get_quote_context(bv: str, quote: str) -> dict[str, Any]:
        return await _run_tool(
            "get_quote_context", tool_registry.get_quote_context, bv=bv, quote=quote
        )

    @mcp.tool(
        name="explain_frame",
        description=_describe(tool_registry.explain_frame),
    )
    async def explain_frame(
        bv: str, frame_id: int, context_radius_seconds: int = 60
    ) -> dict[str, Any]:
        return await _run_tool(
            "explain_frame",
            tool_registry.explain_frame,
            bv=bv,
            frame_id=frame_id,
            context_radius_seconds=context_radius_seconds,
        )

    @mcp.tool(
        name="get_knowledge_units",
        description=_describe(tool_registry.get_knowledge_units),
    )
    async def get_knowledge_units(
        bv: str, kind: str | None = None
    ) -> dict[str, Any]:
        return await _run_tool(
            "get_knowledge_units", tool_registry.get_knowledge_units, bv=bv, kind=kind
        )

    @mcp.tool(
        name="get_frame_description",
        description=_describe(tool_registry.get_frame_description),
    )
    async def get_frame_description(bv: str, frame_id: int) -> dict[str, Any]:
        return await _run_tool(
            "get_frame_description",
            tool_registry.get_frame_description,
            bv=bv,
            frame_id=frame_id,
        )

    @mcp.tool(
        name="list_lectures",
        description=_describe(tool_registry.list_lectures),
    )
    async def list_lectures(
        limit: int = 50,
        domain: str | None = None,
        direction: str | None = None,
    ) -> dict[str, Any]:
        return await _run_tool(
            "list_lectures",
            tool_registry.list_lectures,
            limit=limit,
            domain=domain,
            direction=direction,
        )

    if expose_summarize:

        @mcp.tool(
            name="summarize_video",
            description=_describe(tool_registry.summarize_video),
        )
        async def summarize_video(
            url: str, force_refresh: bool = False
        ) -> dict[str, Any]:
            return await _run_tool(
                "summarize_video",
                tool_registry.summarize_video,
                url=url,
                force_refresh=force_refresh,
            )

    return mcp
