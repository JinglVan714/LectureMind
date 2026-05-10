"""LangGraph ReAct agent for the in-process Copilot.

Design choices (see ``handoff_copilot.md`` decisions D32-D37):

* Single-node ReAct loop: ``call_model → tools → call_model → …``.
  Everything else (streaming, SSE wrapping, auth) happens in Stage 5's
  API layer; this file only exposes the compiled graph plus two
  convenience helpers.
* ``build_graph(ctx)`` returns a fresh ``CompiledGraph`` per call.  The
  graph itself is stateless; the tools close over ``ctx`` via the
  ``make_copilot_tools`` factory so Stage 5 can spin up one graph per
  request without sharing Python globals.
* ``validate_anchors`` runs once on the final answer.  It does not
  mutate the tool-call history — only the content string — and returns
  a list of structured warnings for the API layer to forward to the
  client.
* ``quick_ask(ctx, bv, question)`` is an end-to-end convenience entry
  used by tests and the spec's manual REPL check
  (``quick_ask('BV1o2421A7Dr', '这一章的核心问题？')``).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from app.config import get_settings
from app.storage.db import Database

from . import tools as tool_registry
from .prompts import ANCHOR_PATTERNS, COPILOT_SYSTEM, format_references_block
from .rag import RAGStore
from .sections import SECTION_LINE_RE, SECTION_TYPES, normalise_section_type
from .tools import (
    ExplainFrameInput,
    GetChapterInput,
    GetFrameInput,
    GetQuoteContextInput,
    SearchLectureInput,
    SearchLecturesInput,
    ToolContext,
    ToolError,
    WebSearchInput,
    _flat_frames,
    _load_lecture,
)

logger = logging.getLogger(__name__)

FINAL_RESPONSE_PROMPT = (
    "请基于上面的工具结果直接输出最终答案。不要再请求工具，不要输出 <think>、</think> "
    "或任何思考过程。最终答案第一行必须单独写段标签；除明显跑题时写 [[offtopic]] 外，"
    "第一行必须是 [[evidence]]，即使只有一段也不能省略。最终答案按需分层，不要模板化凑段："
    "默认只输出 [[evidence]] 1 段；"
    "只有问题确实需要前置知识、原理、应用、跨视频、联网或边界说明时，才增加最必要的补充段。"
    "一般答案 1-2 段，复杂答案 3 段，4 段是硬上限；严禁固定输出 4 段，严禁同时列出 "
    "[[background]]、[[deep_dive]]、[[application]]。每段首行使用 [[evidence]]、[[background]]、"
    "[[extension]]、[[deep_dive]]、[[application]]、[[boundary]] 或 [[offtopic]]。"
    "[[evidence]] 只写当前讲义直接支持的内容，并尽量为关键事实附上 [t=05:46]、[F7] 或 [Ch3]；"
    "若讲义未直接展开但问题与讲义主题相关，请在 [[background]] 或 [[deep_dive]] 中二选一做学习向补全，"
    "并明确它不是讲义原话。时间必须写成分钟:秒，章节和关键帧编号不要加尖括号。"
)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    bv: str
    question: str
    references: list[dict] | None
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


# ---------------------------------------------------------------------------
# Tool wrapping — LangChain BaseTool objects with ctx captured in closures.
# ---------------------------------------------------------------------------


def _wrap_async(
    name: str,
    description: str,
    args_schema: type,
    coroutine: Callable[..., Awaitable[Any]],
) -> BaseTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def _describe(fn: Callable[..., Any]) -> str:
    doc = (fn.__doc__ or "").strip().splitlines()
    return doc[0] if doc else fn.__name__


def make_copilot_tools(ctx: ToolContext, *, enable_web: bool = False) -> list[BaseTool]:
    """Five Copilot tools bound to a captured ``ToolContext``.

    The returned ``BaseTool`` list is consumable by
    ``model.bind_tools(...)`` and ``ToolNode(...)``.  Each wrapper
    converts the underlying Pydantic output model to a plain JSON dict
    so the LLM sees ``{"chunks": [...]}`` rather than an opaque
    ``SearchLectureOutput`` repr; ``ToolError`` bubbles up as a
    ``{"error": {...}}`` dict instead of raising, because LangChain's
    ``ToolNode`` treats exceptions as fatal to the run.
    """

    async def _search_lecture(bv: str, query: str, top_k: int = 5) -> dict:
        try:
            out = await tool_registry.search_lecture(
                ctx, bv=bv, query=query, top_k=top_k
            )
            return out.model_dump(mode="json")
        except ToolError as exc:
            return {"error": exc.to_dict()}

    async def _search_lectures(
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        direction: str | None = None,
    ) -> dict:
        try:
            out = await tool_registry.search_lectures(
                ctx, query=query, top_k=top_k, domain=domain, direction=direction
            )
            return out.model_dump(mode="json")
        except ToolError as exc:
            return {"error": exc.to_dict()}

    async def _web_search(query: str, count: int = 5) -> dict:
        out = await tool_registry.web_search(ctx, query=query, count=count)
        return out.model_dump(mode="json")

    async def _get_chapter(bv: str, chapter_idx: int) -> dict:
        try:
            out = await tool_registry.get_chapter(
                ctx, bv=bv, chapter_idx=chapter_idx
            )
            return out.model_dump(mode="json")
        except ToolError as exc:
            return {"error": exc.to_dict()}

    async def _get_frame(bv: str, frame_id: int) -> dict:
        try:
            out = await tool_registry.get_frame(ctx, bv=bv, frame_id=frame_id)
            return out.model_dump(mode="json")
        except ToolError as exc:
            return {"error": exc.to_dict()}

    async def _get_quote_context(bv: str, quote: str) -> dict:
        try:
            out = await tool_registry.get_quote_context(ctx, bv=bv, quote=quote)
            return out.model_dump(mode="json")
        except ToolError as exc:
            return {"error": exc.to_dict()}

    async def _explain_frame(bv: str, frame_id: int) -> dict:
        try:
            out = await tool_registry.explain_frame(ctx, bv=bv, frame_id=frame_id)
            return out.model_dump(mode="json")
        except ToolError as exc:
            return {"error": exc.to_dict()}

    tools = [
        _wrap_async(
            "search_lecture",
            _describe(tool_registry.search_lecture),
            SearchLectureInput,
            _search_lecture,
        ),
        _wrap_async(
            "search_lectures",
            _describe(tool_registry.search_lectures),
            SearchLecturesInput,
            _search_lectures,
        ),
        _wrap_async(
            "get_chapter",
            _describe(tool_registry.get_chapter),
            GetChapterInput,
            _get_chapter,
        ),
        _wrap_async(
            "get_frame",
            _describe(tool_registry.get_frame),
            GetFrameInput,
            _get_frame,
        ),
        _wrap_async(
            "get_quote_context",
            _describe(tool_registry.get_quote_context),
            GetQuoteContextInput,
            _get_quote_context,
        ),
        _wrap_async(
            "explain_frame",
            _describe(tool_registry.explain_frame),
            ExplainFrameInput,
            _explain_frame,
        ),
    ]
    if enable_web:
        tools.append(
            _wrap_async(
                "web_search",
                _describe(tool_registry.web_search),
                WebSearchInput,
                _web_search,
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Model factory (test-mockable)
# ---------------------------------------------------------------------------


def _build_model() -> Any:
    """Construct a ``ChatOpenAI`` pointing at the DeepSeek endpoint.

    Lives behind a factory so tests can monkey-patch it to return a
    lightweight stub (see ``tests/test_copilot.py::TestAgent``).  Any
    object with an async ``ainvoke(messages)`` method and a
    ``bind_tools(tools)`` chainable is acceptable.

    The Copilot shares the text pipeline's DeepSeek credentials
    (``DEEPSEEK_API_KEY`` + ``DEEPSEEK_BASE_URL``). ``extra_body`` is
    only forwarded when the configured base URL still points at a
    DashScope-compatible Qwen endpoint — DeepSeek's ChatCompletions
    rejects the Qwen-specific ``enable_thinking`` key.
    """
    # Lazy import so pytest collection does not pay the LangChain import
    # cost for tests that do not touch the agent.
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.qwen_copilot_model,
        "base_url": settings.deepseek_base_url,
        "api_key": settings.deepseek_api_key,
        "streaming": True,
        "temperature": 0.2,
    }
    if "dashscope" in (settings.deepseek_base_url or "").lower():
        kwargs["extra_body"] = {"enable_thinking": False}
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(ctx: ToolContext, *, model: Any | None = None, enable_web: bool = False):
    """Compile a fresh ReAct graph bound to ``ctx``.

    ``model`` is an escape hatch for tests — passing a stub avoids the
    ``ChatOpenAI`` import and any outbound network config.  When
    ``None`` (production path), we lazily build one via
    :func:`_build_model`.
    """
    settings = get_settings()
    max_tool_calls = settings.copilot_max_tool_calls
    tools = make_copilot_tools(ctx, enable_web=enable_web)
    llm = model if model is not None else _build_model()
    llm_with_tools = llm.bind_tools(tools)

    def _has_text(msg: BaseMessage | None) -> bool:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        return True
        return False

    async def _call_model(state: AgentState) -> dict:
        messages = state["messages"]
        ai: BaseMessage = await llm_with_tools.ainvoke(messages)
        new_step = state.get("step_count", 0) + 1
        return {"messages": [ai], "step_count": new_step}

    async def _call_final_model(state: AgentState) -> dict:
        messages = [*state["messages"], HumanMessage(content=FINAL_RESPONSE_PROMPT)]
        ai: BaseMessage = await llm.ainvoke(messages)
        new_step = state.get("step_count", 0) + 1
        return {"messages": [ai], "step_count": new_step}

    def _should_continue(state: AgentState) -> str:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if last is None:
            return END
        if getattr(last, "tool_calls", None):
            return "tools"
        if not _has_text(last) and len(messages) >= 2 and isinstance(messages[-2], ToolMessage):
            return "final_model"
        return END

    def _after_tools(state: AgentState) -> str:
        if state.get("step_count", 0) >= max_tool_calls:
            return "final_model"
        return "call_model"

    graph = StateGraph(AgentState)
    graph.add_node("call_model", _call_model)
    graph.add_node("final_model", _call_final_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model", _should_continue, {"tools": "tools", "final_model": "final_model", END: END}
    )
    graph.add_conditional_edges(
        "tools", _after_tools, {"call_model": "call_model", "final_model": "final_model"}
    )
    graph.add_edge("final_model", END)
    return graph.compile()


async def build_initial_state(
    ctx: ToolContext,
    bv: str,
    question: str,
    references: list[dict] | None = None,
    history: list[dict] | None = None,
) -> AgentState:
    """Render the system prompt with live lecture metadata + the user turn.

    Pulled out of ``quick_ask`` so Stage 5's SSE endpoint can reuse it
    and inject per-request correlation data.  Raises ``ToolError`` when
    the BV is missing — letting the API layer return 404 cleanly
    instead of letting the LLM hallucinate a non-existent lecture.

    ``history`` is an optional list of ``{"role": "user"|"assistant",
    "content": str}`` dicts representing the previous multi-turn
    conversation (front-end caps at 10 per D43).  It is spliced between
    the system prompt and the new user turn.  Unknown roles are
    skipped; empty content is skipped; the caller is responsible for
    keeping the list short enough to fit the model context.
    """
    settings = get_settings()
    lecture = await _load_lecture(ctx, bv)
    tax = lecture.taxonomy
    system_text = COPILOT_SYSTEM.format(
        bv=bv,
        title=lecture.title or "(无标题)",
        domain=(tax.domain if tax and tax.domain else "（未分类）"),
        direction=(tax.direction if tax and tax.direction else "（未分类）"),
        references_block=format_references_block(references),
        max_tool_calls=settings.copilot_max_tool_calls,
    )
    msgs: list[BaseMessage] = [SystemMessage(content=system_text)]
    for turn in history or []:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            msgs.append(AIMessage(content=content))
        # Silently drop tool / system / unknown roles — the agent loop
        # does not need them for context and they complicate validation.
    msgs.append(HumanMessage(content=question))
    return {
        "bv": bv,
        "question": question,
        "references": references or [],
        "messages": msgs,
        "step_count": 0,
    }


# ---------------------------------------------------------------------------
# Anchor validation
# ---------------------------------------------------------------------------


def _ts_to_seconds(mm: str, ss: str) -> int:
    return int(mm) * 60 + int(ss)


_MALFORMED_TS_SECONDS = re.compile(r"\[t=(\d{3,})\]")
_MALFORMED_FRAME = re.compile(r"\[F<(\d+)>\]")
_MALFORMED_CHAPTER = re.compile(r"\[Ch<(\d+)>\]")
_WEB_ANCHOR = re.compile(r"\[web\s*·\s*([^\]\s]+)\]")
_MALFORMED_WEB_ANCHOR = re.compile(r"\[web\s*[\-:：]\s*([^\]\s]+)\]")
_CROSS_BV_ANCHOR = re.compile(r"\[(BV[0-9A-Za-z]+)\s*·\s*Ch(\d+)\]")


async def validate_anchors(
    answer: str,
    bv: str,
    db: Database,
    rag: RAGStore | None = None,
) -> tuple[str, list[dict]]:
    """Scan ``answer`` for ``[t=…]/[F…]/[Ch…]`` and flag invalid ones.

    Returns ``(patched_answer, warnings)`` where ``warnings`` is a list
    of ``{"kind": "t|F|Ch", "value": "...", "reason": "..."}`` dicts.

    Behaviour:

    * Invalid ``[t=MM:SS]`` (out of ``[0, duration]``) → replaced by
      ``[⚠ t=MM:SS]``.  We never *delete* an anchor, matching the
      "don't touch LLM output more than necessary" principle used by
      ``mark_unverified_points`` elsewhere in the codebase.
    * Invalid ``[Fid]`` / ``[Chid]`` → same wrap treatment.
    * ``rag`` is accepted for signature compatibility with the Stage 4
      plan but not used today; reserved for future cross-bv validation.
    """
    _ = rag  # reserved
    warnings: list[dict] = []
    raw_section_matches = list(SECTION_LINE_RE.finditer(answer))
    section_matches = [
        m for m in raw_section_matches if normalise_section_type(m.group(1)) in SECTION_TYPES
    ]
    section_types = [normalise_section_type(m.group(1)) for m in section_matches]
    if not section_matches:
        warnings.append(
            {"kind": "section", "value": "", "reason": "missing_section_label"}
        )
    if len(section_types) > 4:
        warnings.append(
            {
                "kind": "section",
                "value": ",".join(section_types),
                "reason": "section_count_exceeded",
            }
        )
    duplicated_section_types = sorted(
        {kind for kind in section_types if section_types.count(kind) > 1}
    )
    for kind in duplicated_section_types:
        warnings.append(
            {"kind": "section", "value": f"[[{kind}]]", "reason": "duplicated_section_type"}
        )
    if "offtopic" in section_types and len(section_types) > 1:
        warnings.append(
            {
                "kind": "section",
                "value": ",".join(section_types),
                "reason": "offtopic_mixed_with_answer",
            }
        )
    for match in raw_section_matches:
        kind = normalise_section_type(match.group(1))
        if kind not in SECTION_TYPES:
            warnings.append(
                {"kind": "section", "value": match.group(0), "reason": "unknown_section_type"}
            )

    def _section_type_at(pos: int) -> str | None:
        current: str | None = None
        for match in section_matches:
            if match.start() > pos:
                break
            current = normalise_section_type(match.group(1))
        return current

    row = await db.get_summary(bv)
    duration = int(row.duration) if row and row.duration else None
    chapter_count = 0
    frame_count = 0
    if row and row.summary_json:
        try:
            from app.understand.schema import LectureJSON

            lec = LectureJSON.model_validate(row.summary_json)
            chapter_count = len(lec.chapters)
            frame_count = len(_flat_frames(lec))
        except Exception as exc:  # noqa: BLE001
            logger.warning("validate_anchors: failed to load lecture %s: %s", bv, exc)

    def _validate_ts(raw: str, secs: int, mm: int, ss: int) -> str:
        if duration is not None and (secs < 0 or secs > duration):
            warnings.append(
                {"kind": "t", "value": raw, "reason": f"out of [0, {duration}]"}
            )
            return f"[⚠ t={mm:02d}:{ss:02d}]"
        return f"[t={mm:02d}:{ss:02d}]"

    def _sub_ts(match: re.Match[str]) -> str:
        mm, ss = match.group(1), match.group(2)
        return _validate_ts(match.group(0), _ts_to_seconds(mm, ss), int(mm), int(ss))

    def _sub_malformed_ts_seconds(match: re.Match[str]) -> str:
        secs = int(match.group(1))
        mm, ss = divmod(secs, 60)
        warnings.append(
            {"kind": "t", "value": match.group(0), "reason": "malformed_seconds"}
        )
        return _validate_ts(match.group(0), secs, mm, ss)

    def _sub_frame(match: re.Match[str]) -> str:
        fid = int(match.group(1))
        if frame_count == 0:
            # No lecture metadata available — don't wrap, but record a soft warning.
            warnings.append(
                {"kind": "F", "value": match.group(0), "reason": "lecture not found"}
            )
            return match.group(0)
        if fid < 1 or fid > frame_count:
            warnings.append(
                {"kind": "F", "value": match.group(0), "reason": f"out of 1..{frame_count}"}
            )
            return f"[⚠ F{fid}]"
        return match.group(0)

    def _sub_malformed_frame(match: re.Match[str]) -> str:
        fid = int(match.group(1))
        warnings.append(
            {"kind": "F", "value": match.group(0), "reason": "malformed_angle_brackets"}
        )
        if frame_count and (fid < 1 or fid > frame_count):
            return f"[⚠ F{fid}]"
        return f"[F{fid}]"

    def _sub_chapter(match: re.Match[str]) -> str:
        cid = int(match.group(1))
        if chapter_count == 0:
            warnings.append(
                {"kind": "Ch", "value": match.group(0), "reason": "lecture not found"}
            )
            return match.group(0)
        if cid < 1 or cid > chapter_count:
            warnings.append(
                {"kind": "Ch", "value": match.group(0), "reason": f"out of 1..{chapter_count}"}
            )
            return f"[⚠ Ch{cid}]"
        return match.group(0)

    def _sub_malformed_chapter(match: re.Match[str]) -> str:
        cid = int(match.group(1))
        warnings.append(
            {"kind": "Ch", "value": match.group(0), "reason": "malformed_angle_brackets"}
        )
        if chapter_count and (cid < 1 or cid > chapter_count):
            return f"[⚠ Ch{cid}]"
        return f"[Ch{cid}]"

    patched = _MALFORMED_TS_SECONDS.sub(_sub_malformed_ts_seconds, answer)
    patched = _MALFORMED_FRAME.sub(_sub_malformed_frame, patched)
    patched = _MALFORMED_CHAPTER.sub(_sub_malformed_chapter, patched)
    for match in _MALFORMED_WEB_ANCHOR.finditer(patched):
        warnings.append(
            {"kind": "web", "value": match.group(0), "reason": "malformed_web_anchor"}
        )
    for match in _WEB_ANCHOR.finditer(patched):
        if _section_type_at(match.start()) == "evidence":
            warnings.append(
                {"kind": "web", "value": match.group(0), "reason": "web_anchor_in_evidence"}
            )
    for match in _CROSS_BV_ANCHOR.finditer(patched):
        if _section_type_at(match.start()) != "extension":
            warnings.append(
                {
                    "kind": "cross_bv",
                    "value": match.group(0),
                    "reason": "cross_bv_anchor_outside_extension",
                }
            )
    patched = ANCHOR_PATTERNS["t"].sub(_sub_ts, patched)
    patched = ANCHOR_PATTERNS["F"].sub(_sub_frame, patched)
    patched = ANCHOR_PATTERNS["Ch"].sub(_sub_chapter, patched)
    for i, match in enumerate(section_matches):
        kind = normalise_section_type(match.group(1))
        if kind != "evidence":
            continue
        start = match.end()
        end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(patched)
        body = patched[start:end]
        if not (
            ANCHOR_PATTERNS["t"].search(body)
            or ANCHOR_PATTERNS["F"].search(body)
            or ANCHOR_PATTERNS["Ch"].search(body)
        ):
            warnings.append(
                {"kind": "section", "value": "[[evidence]]", "reason": "evidence_section_no_anchor"}
            )
    return patched, warnings


# ---------------------------------------------------------------------------
# Convenience entry
# ---------------------------------------------------------------------------


async def quick_ask(
    ctx: ToolContext,
    bv: str,
    question: str,
    references: list[dict] | None = None,
    *,
    model: Any | None = None,
) -> tuple[str, list[dict]]:
    """Single-shot end-to-end invocation returning the validated answer.

    The ``model`` kwarg exists for tests; production callers leave it
    ``None``.  Returns ``(answer, warnings)`` so integration tests can
    assert on both the content and the anchor-validation signal.
    """
    graph = build_graph(ctx, model=model)
    state = await build_initial_state(ctx, bv, question, references)
    final: AgentState = await graph.ainvoke(state)
    last = final["messages"][-1] if final.get("messages") else None
    content = getattr(last, "content", "") if last is not None else ""
    if not isinstance(content, str):
        # LangChain can return a list[ContentBlock] when the model emits
        # tool_calls but no text; join the textual parts for callers.
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return await validate_anchors(content, bv, ctx.db, ctx.rag)
