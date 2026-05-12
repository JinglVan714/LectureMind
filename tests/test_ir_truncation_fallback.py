"""Regression tests for the M2.1 single-call → map-reduce auto-fallback.

These tests pin the architectural safety net introduced in
``app/understand/ir_builder.py::build_with_agents``. When the
single-call IR builder raises the truncation ``RuntimeError`` (tagged
via ``_TRUNCATION_ERROR_TAG``), ``build_with_agents`` must transparently
rerun the build via :class:`MapReduceIRBuilder`, surfacing only a
``stats['map_reduce']['auto_fallback']=True`` flag instead of a hard
failure.

Five behaviours are pinned, in order of how badly each one would break
the user's experience if it regressed:

1. **Happy path / no fallback**: a clean standard-profile build does
   NOT touch the map-reduce code path and ``map_reduce`` is absent
   from stats (we don't want to confuse downstream telemetry).
2. **Truncation → auto-fallback success**: the single-call path raises
   the truncation RuntimeError; ``MapReduceIRBuilder`` is constructed
   with the existing chapter_plan; final IR comes from map-reduce;
   ``stats['map_reduce']['auto_fallback']`` is True.
3. **Truncation + empty chapter_plan → re-raise**: with no anchors
   map-reduce cannot run, so the original RuntimeError bubbles up
   (pipeline.py's outer try/except then falls back to v1 lecturizer).
4. **Truncation + ``LECTURE_IR_AUTO_FALLBACK_TO_MAP_REDUCE=false`` →
   re-raise**: the user can disable the safety net to debug a real
   root cause; the original RuntimeError surfaces unmodified.
5. **Non-truncation RuntimeError → propagate untouched**: only the
   exact ``_TRUNCATION_ERROR_TAG`` shape triggers fallback; any other
   RuntimeError (validation failure, network error, etc.) propagates.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

_tmp = tempfile.mkdtemp(prefix="lecturemind-test-trunc-fb-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.config import get_settings  # noqa: E402
from app.ingest.subtitle import SubtitleSegment  # noqa: E402
from app.understand.agents import CritiqueResult  # noqa: E402
from app.understand.ir import LectureIR, hydrate_lecture_ir_data  # noqa: E402
from app.understand.ir_builder import (  # noqa: E402
    _TRUNCATION_ERROR_TAG,
    LectureIRBuilder,
)
from app.understand.ir_patches import RevisePatchResult  # noqa: E402
from app.understand.lecturize import LecturizeContext  # noqa: E402
from app.understand.profile import LectureProfile  # noqa: E402
from app.understand.vlm import FrameDescription  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures — re-uses the shape from tests/test_p7_routing.py
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _make_ctx(duration: float = 1380.0, bv: str = "BVtrunc23min") -> LecturizeContext:
    return LecturizeContext(
        bv_id=bv,
        url=f"https://www.bilibili.com/video/{bv}",
        title="truncation fallback fixture (23min code-heavy)",
        author="t",
        duration=duration,
        cover_url="",
        segments=[
            SubtitleSegment(start=0.0, end=4.0, text="开场介绍主题"),
            SubtitleSegment(start=600.0, end=604.0, text="中段补充细节"),
            SubtitleSegment(start=1200.0, end=1204.0, text="结尾收束总结"),
        ],
        frame_descs=[
            FrameDescription(
                timestamp=600.0,
                path=Path(os.environ["DATA_DIR"]) / "keyframes" / "trunc" / "0600.jpg",
                caption="trunc kf",
                ocr_text="",
                visual_type="diagram",
                importance_score=0.5,
            )
        ],
    )


def _make_ir(ctx: LecturizeContext, *, marker: str = "from-fallback") -> LectureIR:
    """Build a minimal valid LectureIR. ``marker`` lets tests tell
    which path produced the final IR (single-call vs map-reduce)."""
    data = {
        "core_question": f"trunc fixture / {marker}",
        "mainline": ["进入主题", "讨论结构", "收束总结"],
        "chapters": [
            {
                "title": f"导论 [{marker}]",
                "start": 0,
                "end": 600,
                "summary": "导论。",
                "points": [{"text": "p1", "ts": 0, "quote": "开场介绍主题"}],
            },
            {
                "title": "正文",
                "start": 600,
                "end": int(ctx.duration),
                "summary": "正文。",
                "points": [{"text": "p2", "ts": 600, "quote": "中段补充细节"}],
            },
        ],
    }
    hydrate_lecture_ir_data(data, ctx)
    return LectureIR.model_validate(data)


def _standard_profile(duration: float) -> LectureProfile:
    """Standard profile (single-call path) — matches profile.py exactly."""
    return LectureProfile(
        name="standard",
        duration_sec=duration,
        use_chapter_planner=True,
        use_map_reduce=False,
        use_chapter_cache=False,
        critic_mode="full",
        reviser_mode="patch",
        study_question_mode="multi_window",
        chapter_planner_mode="hint",
    )


class _StubCritic:
    """Critic that always says 'ok' so the loop exits without LLM calls."""

    async def audit(self, context):
        return CritiqueResult(
            verdict="ok",
            summary="stub",
            issues=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def revise_patch(self, *, context, issues):  # pragma: no cover
        return RevisePatchResult()

    async def revise(self, ctx, ir_json, issues):  # pragma: no cover
        return None, {}


class _StubStudyAgent:
    async def generate(self, ctx):
        from app.understand.agents import StudyQuestionsResult

        return StudyQuestionsResult(
            questions=["why?"],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            warnings=[],
            metrics={},
        )


def _make_builder() -> LectureIRBuilder:
    b = LectureIRBuilder()
    b._study_agent = _StubStudyAgent()  # type: ignore[assignment]
    b._critic_agent = _StubCritic()
    return b


# ---------------------------------------------------------------------------
# 1. Happy path — clean standard run does NOT touch map-reduce
# ---------------------------------------------------------------------------


def test_standard_clean_run_no_fallback_no_map_reduce_stats() -> None:
    """When the single-call path returns a valid IR, ``stats['map_reduce']``
    must NOT be present. A stray ``auto_fallback`` flag would confuse the
    matrix verifier into reporting an unrelated map-reduce telemetry block.
    """
    ctx = _make_ctx()
    profile = _standard_profile(ctx.duration)
    expected_ir = _make_ir(ctx, marker="single-call-clean")

    builder = _make_builder()

    async def _fake_single(**kwargs: Any):
        return expected_ir, 0.001  # (ir, validate_sec)

    builder._build_via_single_call = _fake_single  # type: ignore[assignment]

    # Sentinel: map-reduce path MUST NOT be called.
    async def _forbidden_mr(**kwargs: Any):  # pragma: no cover - defensive
        raise AssertionError("map-reduce path triggered on clean standard run")

    builder._build_via_map_reduce = _forbidden_mr  # type: ignore[assignment]

    out_ir, stats = _run(
        builder.build_with_agents(
            ctx,
            profile=profile,
            chapter_plan=["a", "b"],
            chapter_cache=None,
        )
    )
    assert out_ir is expected_ir
    assert "map_reduce" not in stats


# ---------------------------------------------------------------------------
# 2. Truncation → auto-fallback succeeds via map-reduce
# ---------------------------------------------------------------------------


def test_standard_truncation_triggers_map_reduce_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BV1ypdgBCE9B real-machine failure shape — single-call raises
    the truncation RuntimeError, build_with_agents must transparently
    rerun via MapReduceIRBuilder using the EXISTING chapter_plan."""
    ctx = _make_ctx()
    profile = _standard_profile(ctx.duration)
    expected_fallback_ir = _make_ir(ctx, marker="from-map-reduce-fallback")
    captured: dict[str, Any] = {}
    builder = _make_builder()

    async def _truncated_single(**kwargs: Any):
        # Verbatim shape of the error _parse_to_dict raises on
        # finish_reason='length'. Real production prefixes carry the
        # raw char count + a position hint; we keep both so the
        # _TRUNCATION_ERROR_TAG substring match exercises the real
        # detection path.
        raise RuntimeError(
            "LLM truncated LectureIR JSON (finish_reason=length, raw "
            "19601 chars): Expecting ',' delimiter: line 416 column 6 "
            "(char 17710). Raise LECTURE_IR_MAX_TOKENS or lower "
            "LECTURE_PROFILE_THRESHOLDS_SEC so this video routes to the "
            "map-reduce builder."
        )

    builder._build_via_single_call = _truncated_single  # type: ignore[assignment]

    class _StubMapReduceStats:
        map_calls = 3
        cache_hits = 0
        cache_writes = 3
        map_failures = ()
        map_total_sec = 1.2
        reduce_local_sec = 0.1
        reduce_global_sec = 0.05
        reduce_global_attempts = 1

    async def _fake_mr(*, ctx, profile, chapter_plan, chapter_cache, study_questions, settings):
        captured["mr_call"] = {
            "anchors": len(list(chapter_plan or [])),
            "study_q": list(study_questions or []),
            "profile_name": profile.name,
            "cache_is_none": chapter_cache is None,
            "ctx_bv": ctx.bv_id,
        }
        return expected_fallback_ir, _StubMapReduceStats(), 1.5

    builder._build_via_map_reduce = _fake_mr  # type: ignore[assignment]

    # Stub make_chapter_cache_from_settings so we don't touch sqlite.
    sentinel_cache = object()
    monkeypatch.setattr(
        "app.understand.ir_builder.make_chapter_cache_from_settings",
        lambda s: sentinel_cache,
    )

    out_ir, stats = _run(
        builder.build_with_agents(
            ctx,
            profile=profile,
            chapter_plan=["anchor-a", "anchor-b", "anchor-c"],
            chapter_cache=None,
        )
    )

    assert out_ir is expected_fallback_ir, "Final IR must come from map-reduce, not the failed single call"
    assert captured["mr_call"]["anchors"] == 3
    assert captured["mr_call"]["profile_name"] == "standard"
    # The chapter_cache was None when build_with_agents was called, so
    # the fallback path had to construct one on demand.
    assert captured["mr_call"]["cache_is_none"] is False
    assert captured["mr_call"]["study_q"] == ["why?"]  # forwarded from StudyAgent

    # Telemetry contract: caller can detect "this run was rescued" via
    # a single dict lookup without parsing logs.
    assert "map_reduce" in stats
    assert stats["map_reduce"]["auto_fallback"] is True
    assert stats["map_reduce"]["map_calls"] == 3


# ---------------------------------------------------------------------------
# 3. Truncation + empty chapter_plan → re-raise (no fallback possible)
# ---------------------------------------------------------------------------


def test_standard_truncation_with_empty_chapter_plan_reraises() -> None:
    """Map-reduce requires non-empty anchors. With chapter_plan=[] the
    original RuntimeError must bubble up so pipeline.py's outer
    try/except can take the v1 lecturizer fallback instead."""
    ctx = _make_ctx()
    profile = _standard_profile(ctx.duration)
    builder = _make_builder()

    async def _truncated_single(**kwargs: Any):
        raise RuntimeError(
            f"{_TRUNCATION_ERROR_TAG} (finish_reason=length, raw 19601 chars): mock"
        )

    builder._build_via_single_call = _truncated_single  # type: ignore[assignment]

    async def _forbidden_mr(**kwargs: Any):  # pragma: no cover - defensive
        raise AssertionError("map-reduce path triggered with empty chapter_plan")

    builder._build_via_map_reduce = _forbidden_mr  # type: ignore[assignment]

    with pytest.raises(RuntimeError) as exc:
        _run(
            builder.build_with_agents(
                ctx,
                profile=profile,
                chapter_plan=[],  # the critical condition
                chapter_cache=None,
            )
        )
    # The original truncation message must be preserved verbatim — we
    # don't want to mask the root cause when no recovery is possible.
    assert _TRUNCATION_ERROR_TAG in str(exc.value)


# ---------------------------------------------------------------------------
# 4. Truncation + safety-net disabled → re-raise
# ---------------------------------------------------------------------------


def test_standard_truncation_respects_disabled_safety_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Users who set ``LECTURE_IR_AUTO_FALLBACK_TO_MAP_REDUCE=false``
    explicitly want the raw error to surface (e.g. while debugging a
    prompt that's producing huge but legitimate output)."""
    ctx = _make_ctx()
    profile = _standard_profile(ctx.duration)
    builder = _make_builder()

    async def _truncated_single(**kwargs: Any):
        raise RuntimeError(
            f"{_TRUNCATION_ERROR_TAG} (finish_reason=length, raw 19601 chars): mock"
        )

    builder._build_via_single_call = _truncated_single  # type: ignore[assignment]

    async def _forbidden_mr(**kwargs: Any):  # pragma: no cover - defensive
        raise AssertionError("map-reduce path triggered with safety net disabled")

    builder._build_via_map_reduce = _forbidden_mr  # type: ignore[assignment]

    # Flip the kill-switch via env so a fresh Settings instance picks it
    # up. get_settings is cached so we have to invalidate it too.
    monkeypatch.setenv("LECTURE_IR_AUTO_FALLBACK_TO_MAP_REDUCE", "false")
    get_settings.cache_clear()
    try:
        builder._settings = get_settings()
        with pytest.raises(RuntimeError) as exc:
            _run(
                builder.build_with_agents(
                    ctx,
                    profile=profile,
                    chapter_plan=["a", "b", "c"],
                    chapter_cache=None,
                )
            )
        assert _TRUNCATION_ERROR_TAG in str(exc.value)
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 5. Non-truncation RuntimeError must propagate untouched
# ---------------------------------------------------------------------------


def test_standard_non_truncation_runtime_error_propagates() -> None:
    """A validation failure, network error, or any other RuntimeError
    whose message does NOT contain _TRUNCATION_ERROR_TAG must propagate
    without triggering the map-reduce fallback. Otherwise unrelated
    bugs would silently turn into map-reduce calls + extra LLM spend."""
    ctx = _make_ctx()
    profile = _standard_profile(ctx.duration)
    builder = _make_builder()

    sentinel_error_message = "some completely unrelated validation failure"

    async def _other_failure(**kwargs: Any):
        raise RuntimeError(sentinel_error_message)

    builder._build_via_single_call = _other_failure  # type: ignore[assignment]

    async def _forbidden_mr(**kwargs: Any):  # pragma: no cover - defensive
        raise AssertionError("map-reduce path triggered on non-truncation RuntimeError")

    builder._build_via_map_reduce = _forbidden_mr  # type: ignore[assignment]

    with pytest.raises(RuntimeError) as exc:
        _run(
            builder.build_with_agents(
                ctx,
                profile=profile,
                chapter_plan=["a", "b"],
                chapter_cache=None,
            )
        )
    # The original message must come through unchanged.
    assert sentinel_error_message in str(exc.value)
    assert _TRUNCATION_ERROR_TAG not in str(exc.value)
