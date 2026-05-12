"""Regression tests for the M2.1 truncation symmetry fix.

Companion to ``tests/test_ir_max_tokens.py``. That file pins the
single-call IR builder (``LectureIRBuilder._call`` /
``_parse_to_dict``); this one extends the same contract to the two
remaining places that emit large JSON payloads and so share the
4096-default truncation risk:

1. ``CriticReviserAgent.revise`` — full-rewrite Reviser (hit when
   ``LECTURE_REVISER_MODE=full`` or via the M1 legacy fallback).
2. ``MapReduceIRBuilder._call_llm`` /
   ``MapReduceIRBuilder.reduce_global_pass`` — the long / epic profile
   global pass that synthesises mainline + knowledge_units.

Three behaviours are pinned for each call site:

* ``max_tokens`` is forwarded from the appropriate Settings field.
* The returned object surfaces ``finish_reason`` (tuple for
  ``_call_llm``, ``usage`` dict for ``revise``).
* When ``finish_reason='length'`` and the JSON tail is therefore
  unterminated, the error path emits an actionable message that names
  the right knob (``LECTURE_REVISER_MAX_TOKENS`` /
  ``LECTURE_REDUCE_GLOBAL_MAX_TOKENS``) and falls back to the strict-
  vs-loose contract the rest of the agents already follow.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

_tmp = tempfile.mkdtemp(prefix="lecturemind-test-truncation-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.config import get_settings  # noqa: E402
from app.ingest.subtitle import SubtitleSegment  # noqa: E402
from app.understand.agents import (  # noqa: E402
    CriticReviserAgent,
    CritiqueIssue,
)
from app.understand.chapter_cache import ChapterCache  # noqa: E402
from app.understand.ir_map_reduce import (  # noqa: E402
    MapReduceIRBuilder,
    ReduceGlobalError,
)
from app.understand.lecturize import LecturizeContext  # noqa: E402
from app.understand.profile import select_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Shared stub plumbing — mirrors tests/test_ir_max_tokens.py
# ---------------------------------------------------------------------------


class _StubChoice:
    def __init__(self, content: str, finish_reason: str | None) -> None:
        self.message = type("_Msg", (), {"content": content})()
        # When finish_reason is None we deliberately skip the attribute so
        # the legacy-stub path (no finish_reason surface) stays exercised.
        if finish_reason is not None:
            self.finish_reason = finish_reason


class _StubUsage:
    prompt_tokens = 50
    completion_tokens = 120
    total_tokens = 170


class _StubResp:
    def __init__(self, content: str, finish_reason: str | None) -> None:
        self.choices = [_StubChoice(content, finish_reason)]
        self.usage = _StubUsage()


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        # default: clean JSON, normal stop
        self.next_response: _StubResp = _StubResp("{}", "stop")

    async def create(self, **kwargs: Any) -> _StubResp:
        self.calls.append(dict(kwargs))
        return self.next_response


class _StubChat:
    def __init__(self, completions: _Recorder) -> None:
        self.completions = completions


class _StubClient:
    def __init__(self) -> None:
        self.completions = _Recorder()
        self.chat = _StubChat(self.completions)


def _run(coro):
    return asyncio.run(coro)


def _make_ctx() -> LecturizeContext:
    return LecturizeContext(
        bv_id="BVtesttrunc",
        url="https://www.bilibili.com/video/BVtesttrunc",
        title="truncation regression",
        author="t",
        duration=1380.0,
        cover_url="",
        segments=[SubtitleSegment(start=0.0, end=4.0, text="开场介绍")],
        frame_descs=[],
    )


def _make_issues() -> list[CritiqueIssue]:
    return [
        CritiqueIssue(
            kind="missing_glossary",
            severity="medium",
            location="chapters[0]",
            evidence="术语未定义",
            suggestion="补 glossary",
        )
    ]


# ===========================================================================
# A. CriticReviserAgent.revise
# ===========================================================================


def _agent_with_stub() -> tuple[CriticReviserAgent, _StubClient]:
    agent = CriticReviserAgent()
    stub = _StubClient()
    agent._client = stub  # type: ignore[assignment]
    return agent, stub


def test_revise_forwards_lecture_reviser_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LECTURE_REVISER_MAX_TOKENS", "8192")
    get_settings.cache_clear()
    try:
        agent, stub = _agent_with_stub()
        # Re-read settings so the constructor's cached _settings is fresh.
        agent._settings = get_settings()  # type: ignore[assignment]
        stub.completions.next_response = _StubResp('{"chapters": []}', "stop")
        obj, usage = _run(agent.revise(_make_ctx(), {"chapters": []}, _make_issues()))
        assert isinstance(obj, dict)
        assert stub.completions.calls[0].get("max_tokens") == 8192
        assert isinstance(usage, dict)
    finally:
        get_settings.cache_clear()


def test_revise_omits_max_tokens_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LECTURE_REVISER_MAX_TOKENS", "0")
    get_settings.cache_clear()
    try:
        agent, stub = _agent_with_stub()
        agent._settings = get_settings()  # type: ignore[assignment]
        stub.completions.next_response = _StubResp('{"chapters": []}', "stop")
        _run(agent.revise(_make_ctx(), {"chapters": []}, _make_issues()))
        assert "max_tokens" not in stub.completions.calls[0]
    finally:
        get_settings.cache_clear()


def test_revise_truncation_strict_raises_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode + finish_reason='length' + bad JSON → RuntimeError
    whose message mentions both LECTURE_REVISER_MAX_TOKENS and the
    LECTURE_REVISER_MODE=patch escape hatch."""
    monkeypatch.setenv("LECTURE_STRICT_AGENTS", "true")
    monkeypatch.setenv("LECTURE_REVISER_MAX_TOKENS", "8192")
    get_settings.cache_clear()
    try:
        agent, stub = _agent_with_stub()
        agent._settings = get_settings()  # type: ignore[assignment]
        agent._strict = True
        truncated = '{"chapters": [{"title": "ch1", "summary": "未完'  # mid-string
        stub.completions.next_response = _StubResp(truncated, "length")
        with pytest.raises(RuntimeError) as exc:
            _run(agent.revise(_make_ctx(), {"chapters": []}, _make_issues()))
        msg = str(exc.value)
        assert "truncated" in msg.lower()
        assert "LECTURE_REVISER_MAX_TOKENS" in msg
        assert "LECTURE_REVISER_MODE=patch" in msg
    finally:
        get_settings.cache_clear()


def test_revise_truncation_loose_logs_warning_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Loose mode (default) keeps the no-op contract — returns
    ``(None, usage)`` so the pipeline degrades to "no revisions" — but
    the warning log must still carry the truncation diagnostic."""
    monkeypatch.setenv("LECTURE_STRICT_AGENTS", "false")
    monkeypatch.setenv("LECTURE_REVISER_MAX_TOKENS", "8192")
    get_settings.cache_clear()
    try:
        agent, stub = _agent_with_stub()
        agent._settings = get_settings()  # type: ignore[assignment]
        agent._strict = False
        truncated = '{"chapters": [{"title": "ch1", "summary": "未完'
        stub.completions.next_response = _StubResp(truncated, "length")
        with caplog.at_level(logging.WARNING, logger="app.understand.agents"):
            obj, usage = _run(
                agent.revise(_make_ctx(), {"chapters": []}, _make_issues())
            )
        assert obj is None
        assert usage  # _usage(resp) was returned, not the empty dict
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "truncated" in joined.lower()
        assert "LECTURE_REVISER_MAX_TOKENS" in joined
    finally:
        get_settings.cache_clear()


def test_revise_non_length_finish_reason_keeps_legacy_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When finish_reason is 'stop' the JSON failure is a real prompt /
    hallucination bug, not truncation — keep the legacy wording so
    matrix / verifier scripts that parse logs do not break."""
    monkeypatch.setenv("LECTURE_STRICT_AGENTS", "false")
    get_settings.cache_clear()
    try:
        agent, stub = _agent_with_stub()
        agent._settings = get_settings()  # type: ignore[assignment]
        agent._strict = False
        stub.completions.next_response = _StubResp("not json at all", "stop")
        with caplog.at_level(logging.WARNING, logger="app.understand.agents"):
            obj, _usage_dict = _run(
                agent.revise(_make_ctx(), {"chapters": []}, _make_issues())
            )
        assert obj is None
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "Reviser returned non-JSON" in joined
        assert "truncated" not in joined.lower()
    finally:
        get_settings.cache_clear()


# ===========================================================================
# B. MapReduceIRBuilder._call_llm + reduce_global_pass
# ===========================================================================


class _SettingsLite:
    """Minimal stand-in for app.config.Settings used by MapReduceIRBuilder.

    Only the attributes the builder actually touches are defined; the
    rest fall through ``getattr(..., default)`` lookups in the impl.
    """

    qwen_text_model = "deepseek-v4-flash"

    def __init__(self, **overrides: Any) -> None:
        self.lecture_map_chapter_max_retries = 0
        self.lecture_map_chapter_timeout = 30.0
        self.lecture_reduce_global_max_retries = 1
        self.lecture_reduce_global_timeout = 30.0
        self.lecture_reduce_global_max_tokens = 4000
        self.lecture_map_prompt_version = "test-v1"
        for k, v in overrides.items():
            setattr(self, k, v)

    def text_extra_body(self) -> dict[str, Any]:
        return {}


def _make_mr_builder(stub_client: _StubClient, tmp_path: Path) -> MapReduceIRBuilder:
    """Build a MapReduceIRBuilder against the in-memory stub.

    ``ChapterCache`` is required (its ``.enabled`` attribute is touched
    on the map path) but the global-pass tests don't go anywhere near
    it, so a fresh on-disk one is fine.
    """
    real_settings = get_settings()
    profile = select_profile(2400.0, real_settings)  # long
    cache = ChapterCache(tmp_path / "cc.sqlite")
    return MapReduceIRBuilder(
        client=stub_client,
        settings=_SettingsLite(),
        profile=profile,
        chapter_cache=cache,
    )


def test_call_llm_returns_tuple_with_finish_reason(tmp_path: Path) -> None:
    """Smoke: the new ``tuple[str, str]`` return shape is honoured even
    when the stub omits ``finish_reason`` entirely (legacy stub compat)."""
    stub = _StubClient()
    stub.completions.next_response = _StubResp('{"x": 1}', None)
    builder = _make_mr_builder(stub, tmp_path)
    content, finish_reason = _run(
        builder._call_llm(system="sys", user="usr", timeout=5.0)
    )
    assert content == '{"x": 1}'
    assert finish_reason == ""


def test_call_llm_surfaces_length_finish_reason(tmp_path: Path) -> None:
    stub = _StubClient()
    stub.completions.next_response = _StubResp('{"x":', "length")
    builder = _make_mr_builder(stub, tmp_path)
    _content, finish_reason = _run(
        builder._call_llm(system="sys", user="usr", timeout=5.0)
    )
    assert finish_reason == "length"


def test_reduce_global_truncation_raises_reduce_global_error_with_hint(
    tmp_path: Path,
) -> None:
    """Two consecutive truncation responses (retries+1=2) exhaust the
    budget and surface ReduceGlobalError; its chained cause must be the
    actionable RuntimeError we raise on finish_reason='length'."""
    stub = _StubClient()
    truncated = '{"lecture_summary": "未完'  # unterminated string
    stub.completions.next_response = _StubResp(truncated, "length")
    builder = _make_mr_builder(stub, tmp_path)
    meta = type(
        "_Meta",
        (),
        {"bv_id": "BVtest", "title": "t", "duration": 2400},
    )()
    local_ir = {
        "chapters": [
            {
                "title": "ch1",
                "start": 0,
                "end": 1200,
                "points": [],
                "knowledge_units": [],
            }
        ],
        "knowledge_units": [],
        "mainline": [],
        "cross_references": [],
        "final_synthesis": "",
    }
    with pytest.raises(ReduceGlobalError) as exc:
        _run(
            builder.reduce_global_pass(
                local_ir=local_ir,
                glossary_conflicts=[],
                meta=meta,
                study_questions=[],
                stats={
                    "reduce_global_attempts": 0,
                    "reduce_global_sec": 0.0,
                },
            )
        )
    msg = str(exc.value)
    # The truncation-specific hint should ride along inside the chain
    # (ReduceGlobalError carries the last_exc as __cause__).
    cause_chain = []
    cur: BaseException | None = exc.value
    while cur is not None:
        cause_chain.append(str(cur))
        cur = cur.__cause__
    joined = "\n".join(cause_chain)
    assert "truncated" in joined.lower()
    assert "LECTURE_REDUCE_GLOBAL_MAX_TOKENS" in joined
    # And the outer ReduceGlobalError message itself still names the
    # final attempt count so log readers don't have to walk the chain.
    assert "after" in msg.lower() and "attempts" in msg.lower()


def test_reduce_global_non_length_keeps_legacy_failure_path(tmp_path: Path) -> None:
    """When finish_reason is empty / 'stop', the JSON error must NOT be
    wrapped in our truncation message — the legacy ``except Exception``
    branch keeps logging "Reduce-global attempt N/N failed: …" with the
    raw ValueError text so existing log parsers stay happy."""
    stub = _StubClient()
    stub.completions.next_response = _StubResp("not json", "stop")
    builder = _make_mr_builder(stub, tmp_path)
    meta = type("_Meta", (), {"bv_id": "BV", "title": "t", "duration": 2400})()
    local_ir = {
        "chapters": [
            {
                "title": "ch1",
                "start": 0,
                "end": 1200,
                "points": [],
                "knowledge_units": [],
            }
        ],
        "knowledge_units": [],
        "mainline": [],
        "cross_references": [],
        "final_synthesis": "",
    }
    with pytest.raises(ReduceGlobalError) as exc:
        _run(
            builder.reduce_global_pass(
                local_ir=local_ir,
                glossary_conflicts=[],
                meta=meta,
                study_questions=[],
                stats={"reduce_global_attempts": 0, "reduce_global_sec": 0.0},
            )
        )
    cause_chain = []
    cur: BaseException | None = exc.value
    while cur is not None:
        cause_chain.append(str(cur))
        cur = cur.__cause__
    joined = "\n".join(cause_chain)
    assert "truncated" not in joined.lower()
