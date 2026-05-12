"""M2.2 regression tests for map-chapter truncation handling.

These tests pin the down-stream safety net that fires when the
``ChapterPlanner._enforce_max_duration`` upstream prevention misses
an outlier (e.g. a 10min chapter with extreme code density). The
contract:

1. ``LECTURE_MAP_CHAPTER_MAX_TOKENS`` is forwarded to ``_call_llm`` so
   every map-chapter call shares the same 8192-token ceiling as the
   rest of the M2 pipeline (was unset → DeepSeek default ~4096).
2. ``finish_reason='length'`` plus a JSON parse error short-circuits
   the retry loop. Re-issuing the same prompt under the same budget
   would only burn another LLM call for the same truncated output.
3. The actionable error message tagged by ``MAP_CHAPTER_TRUNCATION_TAG``
   names BOTH knobs the user can turn (``LECTURE_MAP_CHAPTER_MAX_TOKENS``
   and ``LECTURE_CHAPTER_MAX_DURATION_SEC``) so the rendered HTML
   chapter card is self-debugging.
4. Non-truncation errors (timeout / transient API err) still burn the
   full retry budget — we only short-circuit truncation specifically.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_tmp = tempfile.mkdtemp(prefix="lecturemind-test-mctrunc-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.understand.chapter_cache import ChapterCache  # noqa: E402
from app.understand.chapter_planner import ChapterAnchor  # noqa: E402
from app.understand.ir_map_reduce import (  # noqa: E402
    MAP_CHAPTER_TRUNCATION_TAG,
    MapReduceIRBuilder,
)
from app.understand.profile import _PROFILE_TABLE  # noqa: E402


# ---------- duck-typed fixtures (same shape as test_ir_map_reduce.py) ------


@dataclass
class _Seg:
    start: float
    end: float
    text: str


@dataclass
class _Frm:
    timestamp: float
    ocr_text: str = ""
    caption: str = ""
    visual_type: str = "slide_text"
    path: str = ""


@dataclass
class _Meta:
    bv_id: str = "BVTRUNC"
    title: str = "trunc fixture"
    author: str = "u"
    duration: float = 1800.0
    url: str = "https://www.bilibili.com/video/BVTRUNC"
    cover_url: str = ""


# ---------- async OpenAI stub with finish_reason support -------------------


class _StubMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _StubMsg(content)
        self.finish_reason = finish_reason


class _StubResp:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_StubChoice(content, finish_reason=finish_reason)]
        self.usage = None


@dataclass
class _StubResult:
    """What a responder returns: either the JSON content + finish_reason,
    a raw string + finish_reason (so we can simulate a truncated tail),
    or an exception to raise."""

    content: Any = None  # dict (-> json.dumps) | str | None
    finish_reason: str = "stop"
    raise_exc: BaseException | None = None


class _StubCompletions:
    def __init__(self, parent: "_StubClient") -> None:
        self._p = parent

    async def create(self, **kwargs: Any) -> _StubResp:
        msgs = kwargs.get("messages", [])
        sys = ""
        user = ""
        for m in msgs:
            if m.get("role") == "system":
                sys = m.get("content", "")
            elif m.get("role") == "user":
                user = m.get("content", "")
        kind = "global" if "REDUCE_GLOBAL" in sys or "讲义全局编辑" in sys else "map"
        self._p.calls.append({"kind": kind, "user": user, "kwargs": dict(kwargs)})
        if kind == "map":
            result = self._p.map_responder(user, kwargs, self._p)
        else:
            result = self._p.global_responder(user, kwargs, self._p)
        if isinstance(result, _StubResult):
            if result.raise_exc is not None:
                raise result.raise_exc
            content = result.content
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            return _StubResp(content or "", finish_reason=result.finish_reason)
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, dict):
            return _StubResp(json.dumps(result, ensure_ascii=False))
        return _StubResp(str(result or ""))


class _StubChat:
    def __init__(self, parent: "_StubClient") -> None:
        self.completions = _StubCompletions(parent)


class _StubClient:
    def __init__(
        self,
        *,
        map_responder: Callable[..., Any] | None = None,
        global_responder: Callable[..., Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.map_responder = map_responder or _ok_map
        self.global_responder = global_responder or _ok_global
        self.chat = _StubChat(self)

    @property
    def map_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["kind"] == "map"]


def _chapter_index(user: str) -> int:
    m = re.search(r"第\s*(\d+)\s*章", user)
    return int(m.group(1)) if m else 1


def _ok_map(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
    idx = _chapter_index(user)
    return {
        "title": f"章节 {idx}",
        "summary": f"summary {idx}",
        "learning_goal": "g",
        "teaching_notes": ["n"],
        "process_steps": [],
        "points": [{"text": "p", "ts": 0.0, "quote": "q"}],
        "code_blocks": [],
        "formula_blocks": [],
        "pitfalls": [],
        "key_takeaways": [],
        "knowledge_units": [],
    }


def _ok_global(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
    return {
        "lecture_summary": "lecture s",
        "mainline": [
            {"step": 1, "title": "t1", "ts": 0, "chapter_index": 1},
        ],
        "glossary_resolved": [],
        "cross_references": [],
    }


# ---------- minimal Settings stub -----------------------------------------


@dataclass
class _StubSettings:
    qwen_text_model: str = "stub-model"
    deepseek_request_timeout: float = 30.0
    lecture_map_prompt_version: str = "m2-map-v1"
    lecture_map_chapter_max_retries: int = 1
    lecture_map_chapter_timeout: float = 30.0
    # M2.2 — the new knob under test. Default 8192 mirrors the prod
    # default. Tests override this when they want to verify forwarding.
    lecture_map_chapter_max_tokens: int = 8192
    lecture_reduce_global_timeout: float = 30.0
    lecture_reduce_global_max_retries: int = 1
    lecture_reduce_global_max_tokens: int = 4000
    keyframe_min: int = 8
    keyframe_max: int = 20

    def text_extra_body(self) -> dict[str, Any]:
        return {}


def _anchor(start: float, end: float, *, text: str = "anchor") -> ChapterAnchor:
    return ChapterAnchor(
        start_sec=start,
        end_sec=end,
        confidence=1.0,
        anchor_text=text,
        source="equal_split",
        mode="structural",
    )


def _segments_for_chapter1() -> list[_Seg]:
    # Just chapter 1's window. Keep the second chapter empty so the
    # builder still issues an LLM call (a totally empty chapter is a
    # corner case unrelated to the truncation contract).
    return [
        _Seg(0.0, 5.0, "intro one"),
        _Seg(5.0, 10.0, "intro two"),
    ]


def _builder_with(
    *,
    client: _StubClient,
    cache: ChapterCache,
    settings: _StubSettings | None = None,
) -> MapReduceIRBuilder:
    profile = _PROFILE_TABLE["long"].with_duration(1800.0)
    return MapReduceIRBuilder(
        client=client,
        settings=settings or _StubSettings(),
        profile=profile,
        chapter_cache=cache,
    )


# ===========================================================================
# 1. max_tokens forwarded to the LLM call
# ===========================================================================


def test_max_tokens_forwarded_to_call_llm(tmp_path: Path) -> None:
    """LECTURE_MAP_CHAPTER_MAX_TOKENS must reach
    ``client.chat.completions.create``'s kwargs as ``max_tokens``.

    Pre-M2.2 the call passed nothing → DeepSeek defaulted to ~4096
    and BV1ypdgBCE9B chapter 5 truncated. This is the contract that
    keeps the fix wired up.
    """
    client = _StubClient()
    settings = _StubSettings(lecture_map_chapter_max_tokens=8192)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    builder = _builder_with(client=client, cache=cache, settings=settings)
    asyncio.run(
        builder.build(
            chapter_plan=[_anchor(0.0, 1800.0)],
            segments=_segments_for_chapter1(),
            frames=[],
            meta=_Meta(),
            study_questions=[],
        )
    )
    # First call is the map-chapter; check max_tokens reached the wire.
    map_kwargs = client.map_calls[0]["kwargs"]
    assert map_kwargs.get("max_tokens") == 8192


def test_zero_max_tokens_omits_kwarg(tmp_path: Path) -> None:
    """Setting LECTURE_MAP_CHAPTER_MAX_TOKENS=0 must fall back to the
    backend default (i.e. don't pass ``max_tokens`` at all). This is
    the documented escape hatch for users running on a backend whose
    default exceeds 8192."""
    client = _StubClient()
    settings = _StubSettings(lecture_map_chapter_max_tokens=0)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    builder = _builder_with(client=client, cache=cache, settings=settings)
    asyncio.run(
        builder.build(
            chapter_plan=[_anchor(0.0, 1800.0)],
            segments=_segments_for_chapter1(),
            frames=[],
            meta=_Meta(),
            study_questions=[],
        )
    )
    map_kwargs = client.map_calls[0]["kwargs"]
    assert "max_tokens" not in map_kwargs


# ===========================================================================
# 2. finish_reason='length' short-circuits the retry loop
# ===========================================================================


def test_truncated_response_short_circuits_retry_loop(tmp_path: Path) -> None:
    """When ``finish_reason='length'`` AND the JSON tail is unterminated,
    the map-chapter loop must give up after **1** attempt rather than
    burning the full ``max_retries+1`` budget on the same prompt.

    The same prompt at the same budget produces the same truncated
    output, so retrying is pure cost. Pin attempts == 1.
    """
    truncated_tail = (
        '{"title": "ch1", "summary": "s", "points": [{"text": "p", "ts": 0,'
    )  # cut mid-array; valid JSON parse will fail

    def _trunc_responder(
        user: str, kwargs: dict[str, Any], client: _StubClient
    ) -> _StubResult:
        return _StubResult(content=truncated_tail, finish_reason="length")

    client = _StubClient(map_responder=_trunc_responder)
    # max_retries=2 so we'd burn 3 calls without the short-circuit.
    settings = _StubSettings(lecture_map_chapter_max_retries=2)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    builder = _builder_with(client=client, cache=cache, settings=settings)
    _, stats = asyncio.run(
        builder.build(
            chapter_plan=[_anchor(0.0, 1800.0)],
            segments=_segments_for_chapter1(),
            frames=[],
            meta=_Meta(),
            study_questions=[],
        )
    )
    # Exactly 1 map call for the truncated chapter — short-circuit
    # worked.
    assert len(client.map_calls) == 1
    # And exactly one MapFailure with attempts == 1, finish_reason
    # carrying the truncation signal.
    assert len(stats.map_failures) == 1
    f = stats.map_failures[0]
    assert f.attempts == 1
    assert f.finish_reason == "length"
    # The truncation tag must appear in the recorded error excerpt so
    # downstream verifier can dispatch on it.
    assert MAP_CHAPTER_TRUNCATION_TAG in f.error_excerpt


# ===========================================================================
# 3. Placeholder summary surfaces actionable hint
# ===========================================================================


def test_truncated_chapter_placeholder_summary_is_actionable(tmp_path: Path) -> None:
    """The ``"本章生成失败"`` placeholder card on the rendered HTML must
    name BOTH knobs the user can turn so the page is self-debugging.

    Pre-M2.2: ``"本章内容生成失败，请手动重跑或检查日志。"`` (zero
    actionable guidance).
    Post-M2.2: explicit mention of ``LECTURE_MAP_CHAPTER_MAX_TOKENS``
    and ``LECTURE_CHAPTER_MAX_DURATION_SEC``.
    """
    truncated_tail = '{"title": "ch1", "summary": "s",'  # parse fails

    def _trunc(user: str, kwargs: dict[str, Any], client: _StubClient) -> _StubResult:
        return _StubResult(content=truncated_tail, finish_reason="length")

    client = _StubClient(map_responder=_trunc)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    builder = _builder_with(client=client, cache=cache)
    ir, _ = asyncio.run(
        builder.build(
            chapter_plan=[_anchor(0.0, 1800.0)],
            segments=_segments_for_chapter1(),
            frames=[],
            meta=_Meta(),
            study_questions=[],
        )
    )
    assert len(ir.chapters) == 1
    summary = ir.chapters[0].summary
    # Title still degrades gracefully so the HTML renders.
    assert "生成失败" in ir.chapters[0].title
    # The summary now carries actionable guidance for both layers of
    # the M2.2 fix.
    assert "LECTURE_MAP_CHAPTER_MAX_TOKENS" in summary
    assert "LECTURE_CHAPTER_MAX_DURATION_SEC" in summary
    assert "finish_reason=length" in summary


# ===========================================================================
# 4. Non-truncation errors still burn the full retry budget
# ===========================================================================


def test_non_truncation_error_uses_full_retry_budget(tmp_path: Path) -> None:
    """A timeout / transient API err is recoverable on retry, unlike a
    truncation. The retry loop must NOT short-circuit on these — only
    truncation gets the special treatment.
    """
    state = {"calls": 0}

    def _flaky(user: str, kwargs: dict[str, Any], client: _StubClient) -> _StubResult:
        state["calls"] += 1
        # Always fail with a non-truncation error.
        return _StubResult(raise_exc=TimeoutError("network glitch"))

    client = _StubClient(map_responder=_flaky)
    # max_retries=2 → up to 3 attempts.
    settings = _StubSettings(lecture_map_chapter_max_retries=2)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    builder = _builder_with(client=client, cache=cache, settings=settings)
    _, stats = asyncio.run(
        builder.build(
            chapter_plan=[_anchor(0.0, 1800.0)],
            segments=_segments_for_chapter1(),
            frames=[],
            meta=_Meta(),
            study_questions=[],
        )
    )
    # All 3 attempts burned (max_retries + 1 = 3) — no short-circuit.
    assert state["calls"] == 3
    assert stats.map_failures[0].attempts == 3
    assert stats.map_failures[0].error_class == "TimeoutError"
    # finish_reason carried "" because TimeoutError fires before the
    # response object is built.
    assert stats.map_failures[0].finish_reason == ""
