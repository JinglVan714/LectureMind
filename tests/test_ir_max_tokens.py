"""Regression tests for the LectureIR max_tokens / truncation fix.

Bug observed on 2026-05-12 with BV1ypdgBCE9B (23-minute code-heavy
video, standard profile):

    RuntimeError: LLM returned non-JSON LectureIR: Expecting ',' delimiter:
    line 482 column 108 (char 19949)

Root cause: the single-call ``LectureIRBuilder._call`` did not pass
``max_tokens`` to ``chat.completions.create``, so DeepSeek's silent
per-call default (4096) clipped the JSON output mid-string. The retry
decorator could not recover because the same prompt deterministically
re-truncated.

Three behaviours are pinned here:

1. ``_call`` forwards ``settings.lecture_ir_max_tokens`` as
   ``max_tokens`` to the OpenAI-compatible client.
2. The returned usage dict carries ``finish_reason`` from
   ``choices[0]``.
3. ``_parse_to_dict`` raises a *distinct*, actionable RuntimeError
   when the backend reports ``finish_reason='length'`` and the JSON
   tail is therefore unterminated.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import pytest

_tmp = tempfile.mkdtemp(prefix="lecturemind-test-ir-maxtok-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.config import get_settings  # noqa: E402
from app.ingest.subtitle import SubtitleSegment  # noqa: E402
from app.understand.ir_builder import LectureIRBuilder  # noqa: E402
from app.understand.lecturize import LecturizeContext  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = type("_Msg", (), {"content": content})()
        self.finish_reason = finish_reason


class _StubUsage:
    prompt_tokens = 100
    completion_tokens = 200
    total_tokens = 300


class _StubResp:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.choices = [_StubChoice(content, finish_reason)]
        self.usage = _StubUsage()


class _RecordingCompletions:
    """Captures kwargs passed to ``chat.completions.create``.

    Returns whatever ``next_response`` is configured to so individual
    tests can simulate ``finish_reason='stop'`` vs ``='length'``
    without touching real network.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next_response: _StubResp = _StubResp("{}", "stop")

    async def create(self, **kwargs: Any) -> _StubResp:
        self.calls.append(dict(kwargs))
        return self.next_response


class _StubChat:
    def __init__(self, completions: _RecordingCompletions) -> None:
        self.completions = completions


class _StubClient:
    def __init__(self) -> None:
        self.completions = _RecordingCompletions()
        self.chat = _StubChat(self.completions)


def _make_ctx() -> LecturizeContext:
    return LecturizeContext(
        bv_id="BVtestmaxtoken",
        url="https://www.bilibili.com/video/BVtestmaxtoken",
        title="max_tokens regression",
        author="test",
        duration=1380.0,
        cover_url="",
        segments=[SubtitleSegment(start=0.0, end=4.0, text="开场介绍")],
        frame_descs=[],
    )


def _builder_with_stub() -> tuple[LectureIRBuilder, _StubClient]:
    builder = LectureIRBuilder()
    stub = _StubClient()
    builder._client = stub  # type: ignore[assignment]
    return builder, stub


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. max_tokens is forwarded
# ---------------------------------------------------------------------------


def test_call_forwards_lecture_ir_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LECTURE_IR_MAX_TOKENS", "8192")
    get_settings.cache_clear()
    try:
        builder, stub = _builder_with_stub()
        stub.completions.next_response = _StubResp('{"chapters": []}', "stop")
        content, usage = _run(builder._call("user msg"))
        assert content == '{"chapters": []}'
        assert usage.get("finish_reason") == "stop"
        assert len(stub.completions.calls) == 1
        assert stub.completions.calls[0].get("max_tokens") == 8192
    finally:
        get_settings.cache_clear()


def test_call_omits_max_tokens_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LECTURE_IR_MAX_TOKENS=0`` is the documented escape hatch and
    must leave ``max_tokens`` unset so the backend default applies."""
    monkeypatch.setenv("LECTURE_IR_MAX_TOKENS", "0")
    get_settings.cache_clear()
    try:
        builder, stub = _builder_with_stub()
        stub.completions.next_response = _StubResp('{"chapters": []}', "stop")
        _run(builder._call("user msg"))
        assert "max_tokens" not in stub.completions.calls[0]
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 2. finish_reason='length' raises an actionable RuntimeError
# ---------------------------------------------------------------------------


def test_parse_to_dict_truncation_emits_actionable_error() -> None:
    """The exact failure mode reported on BV1ypdgBCE9B: the JSON tail
    is unterminated AND the backend told us finish_reason='length'."""
    builder = LectureIRBuilder()
    ctx = _make_ctx()
    truncated = (
        '{"core_question": "x", "chapters": [{"title": "ch1", "code_blocks": ['
        '{"language": "bash", "code": "curl -fsSL https://x.sh | bash",'
        ' "explanation": "安装 Claude'  # <-- clipped mid-string
    )
    with pytest.raises(RuntimeError) as exc:
        builder._parse_to_dict(truncated, ctx, finish_reason="length")
    msg = str(exc.value)
    # Must NOT be the generic message — it has to mention truncation
    # AND point users at the two real knobs (max_tokens / profile
    # threshold) so they stop debugging a phantom prompt bug.
    assert "truncated" in msg.lower()
    assert "LECTURE_IR_MAX_TOKENS" in msg
    assert "LECTURE_PROFILE_THRESHOLDS_SEC" in msg


def test_parse_to_dict_other_json_error_keeps_legacy_message() -> None:
    """When finish_reason is not 'length' the error message stays at
    its M1 wording, because the bug is then probably a real prompt
    leak / hallucination, not a truncation."""
    builder = LectureIRBuilder()
    ctx = _make_ctx()
    bad_json = "not json at all"
    with pytest.raises(RuntimeError) as exc:
        builder._parse_to_dict(bad_json, ctx, finish_reason="stop")
    msg = str(exc.value)
    assert "LLM returned non-JSON LectureIR" in msg
    assert "truncated" not in msg.lower()


# ---------------------------------------------------------------------------
# 3. finish_reason flows through usage dict
# ---------------------------------------------------------------------------


def test_call_includes_finish_reason_in_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LECTURE_IR_MAX_TOKENS", "8192")
    get_settings.cache_clear()
    try:
        builder, stub = _builder_with_stub()
        stub.completions.next_response = _StubResp('{"x": 1}', "length")
        _content, usage = _run(builder._call("user msg"))
        assert usage["finish_reason"] == "length"
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 200
    finally:
        get_settings.cache_clear()
