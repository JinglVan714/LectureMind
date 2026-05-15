"""M2 P7 — Pipeline routing + audit() + reviser_mode dispatch + telemetry tests.

These are mock-only integration tests that drive
``LectureIRBuilder.build_with_agents`` and ``_run_critic_loop_v2`` through
their per-profile branches without touching any LLM or network. They are
the regression net P7 leans on before the 5-video real-machine sweep.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Mirror tests/test_smoke.py environment bootstrapping so the module can
# be imported standalone via ``pytest tests/test_p7_routing.py``.
_tmp = tempfile.mkdtemp(prefix="lecturemind-test-p7-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.config import get_settings  # noqa: E402
from app.ingest.subtitle import SubtitleSegment  # noqa: E402
from app.understand.agents import (  # noqa: E402
    CritiqueIssue,
    CritiqueResult,
)
from app.understand.critic_context import CriticContext  # noqa: E402
from app.understand.ir import LectureIR, hydrate_lecture_ir_data  # noqa: E402
from app.understand.ir_builder import LectureIRBuilder  # noqa: E402
from app.understand.ir_patches import IRPatch, RejectedPatch, RevisePatchResult  # noqa: E402
from app.understand.lecturize import LecturizeContext  # noqa: E402
from app.understand.profile import LectureProfile, select_profile  # noqa: E402
from app.understand.vlm import FrameDescription  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (small, deterministic IR + ctx that we can hand to the builder)
# ---------------------------------------------------------------------------


def _make_ctx(*, duration: float, bv: str = "BV1p7test000") -> LecturizeContext:
    return LecturizeContext(
        bv_id=bv,
        url=f"https://www.bilibili.com/video/{bv}",
        title="P7 routing fixture",
        author="P7",
        duration=duration,
        cover_url="",
        segments=[
            SubtitleSegment(start=0.0, end=4.0, text="开场介绍主题"),
            SubtitleSegment(start=120.0, end=124.0, text="中段补充细节"),
            SubtitleSegment(start=240.0, end=244.0, text="结尾收束总结"),
        ],
        frame_descs=[
            FrameDescription(
                timestamp=120.0,
                path=Path(os.environ["DATA_DIR"]) / "keyframes" / "p7" / "00120.jpg",
                caption="P7 keyframe",
                ocr_text="",
                visual_type="diagram",
                importance_score=0.5,
            )
        ],
    )


def _make_ir(ctx: LecturizeContext) -> LectureIR:
    data = {
        "core_question": "P7 routing 是否正确？",
        "mainline": ["进入主题", "讨论结构", "收束总结"],
        "chapters": [
            {
                "title": "导论",
                "start": 0,
                "end": 120,
                "summary": "本章导论。",
                "points": [
                    {"text": "开场点 1", "ts": 0, "quote": "开场介绍主题"}
                ],
            },
            {
                "title": "正文",
                "start": 120,
                "end": int(ctx.duration),
                "summary": "本章正文。",
                "points": [
                    {"text": "正文点 1", "ts": 120, "quote": "中段补充细节"}
                ],
            },
        ],
    }
    hydrate_lecture_ir_data(data, ctx)
    return LectureIR.model_validate(data)


def _make_critique(*, with_issues: bool, usage: dict | None = None) -> CritiqueResult:
    issues = (
        [
            CritiqueIssue(
                kind="missing_glossary",
                severity="medium",
                location="chapters[0]",
                evidence="术语未在 glossary 中",
                suggestion="补 glossary",
            )
        ]
        if with_issues
        else []
    )
    return CritiqueResult(
        verdict="needs_revision" if with_issues else "ok",
        summary="stub",
        issues=issues,
        usage=usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _builder_with_stub(agent: Any) -> LectureIRBuilder:
    builder = LectureIRBuilder()
    builder._critic_agent = agent
    return builder


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Profile router is wired into pipeline imports + select_profile contract
# ---------------------------------------------------------------------------


class TestProfileRouter:
    def test_select_profile_returns_expected_branches(self):
        """The four profile names round-trip through Settings thresholds.

        Locks in the contract pipeline.py uses to populate
        ``timing['profile']`` so a future profile rename is caught here
        instead of at real-machine regression time.
        """
        s = get_settings()
        assert select_profile(60.0, s).name == "tiny"
        # 600s is comfortably inside the new standard band (<900s).
        # The old threshold default (180, 1500, 3600) put 900s in
        # standard; the 2026-05-12 fix lowered the upper bound to 900s
        # so that BV1ypdgBCE9B-class 20+min code-heavy videos route
        # straight to the map-reduce builder. See app/config.py.
        assert select_profile(600.0, s).name == "standard"
        # 900s is now the boundary that flips into long (exclusive
        # lower edge belongs to the upper bucket).
        assert select_profile(900.0, s).name == "long"
        assert select_profile(2200.0, s).name == "long"
        assert select_profile(5000.0, s).name == "epic"


# ---------------------------------------------------------------------------
# 2-4. _run_critic_loop_v2 dispatch matrix (off / patch / full)
# ---------------------------------------------------------------------------


class TestCriticLoopV2Dispatch:
    """``_run_critic_loop_v2`` is the P7 entry point. Each ``reviser_mode``
    branch must do the right kind of LLM call (or none at all) and emit
    the right telemetry shape so pipeline_stats / matrix readers can
    rely on the fields without ``KeyError`` defensiveness.
    """

    def _profile(self, name: str, *, critic_mode: str, reviser_mode: str) -> LectureProfile:
        return LectureProfile(
            name=name,  # type: ignore[arg-type]
            duration_sec=900.0,
            use_chapter_planner=True,
            use_map_reduce=False,
            use_chapter_cache=False,
            critic_mode=critic_mode,  # type: ignore[arg-type]
            reviser_mode=reviser_mode,  # type: ignore[arg-type]
            study_question_mode="multi_window",
            chapter_planner_mode="hint",
        )

    def test_off_mode_skips_audit_and_returns_ir_unchanged(self):
        """tiny profile: critic_mode='off' → audit() must not be called."""
        calls = {"audit": 0, "revise_patch": 0, "revise": 0}

        class _StubAgent:
            async def audit(self, context):
                calls["audit"] += 1
                return _make_critique(with_issues=True)

            async def revise_patch(self, *, context, issues):
                calls["revise_patch"] += 1
                return RevisePatchResult()

            async def revise(self, ctx, ir_json, issues):
                calls["revise"] += 1
                return None, {}

        builder = _builder_with_stub(_StubAgent())
        ctx = _make_ctx(duration=120.0)
        ir = _make_ir(ctx)
        profile = self._profile("tiny", critic_mode="off", reviser_mode="off")

        critique, latest_ir, rounds, usages, telemetry = _run(
            builder._run_critic_loop_v2(ctx, ir, ["q?"], profile=profile)
        )

        assert calls == {"audit": 0, "revise_patch": 0, "revise": 0}
        assert latest_ir is ir
        assert rounds == 0
        assert telemetry["critic_sec"] == 0.0
        assert telemetry["reviser_sec"] == 0.0
        # Off mode must NOT inject a 'reviser' telemetry block — pipeline
        # readers use its presence as the "reviser actually ran" signal.
        assert "reviser" not in telemetry
        assert critique.verdict == "ok"

    def test_patch_mode_applies_patches_and_records_telemetry(self):
        """standard / long / epic / code profile: audit → revise_patch → apply_patches."""
        calls = {"audit": 0, "revise_patch": 0, "revise": 0}
        # Patch the chapter[0] summary; stays inside the apply_patches whitelist.
        proposed_patch = IRPatch(
            op="set_text",
            path="/chapters/0/points/0/text",
            text="P7 patched 正文点 1",
        )

        class _StubAgent:
            def __init__(self):
                self.audits = 0

            async def audit(self, context):
                calls["audit"] += 1
                self.audits += 1
                # Verify the dispatcher built a CriticContext for us.
                assert isinstance(context, CriticContext)
                assert context.profile_name == "standard"
                # First audit raises an issue; the next round (after the
                # patch was applied) must report ``ok`` so the loop exits.
                return _make_critique(with_issues=(self.audits == 1))

            async def revise_patch(self, *, context, issues):
                calls["revise_patch"] += 1
                return RevisePatchResult(
                    patches=[proposed_patch],
                    rejected=[],
                    unfixable_issues=[],
                    usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
                )

            async def revise(self, ctx, ir_json, issues):
                calls["revise"] += 1
                return None, {}

        builder = _builder_with_stub(_StubAgent())
        ctx = _make_ctx(duration=900.0)
        ir = _make_ir(ctx)
        profile = self._profile("standard", critic_mode="full", reviser_mode="patch")

        critique, latest_ir, rounds, usages, telemetry = _run(
            builder._run_critic_loop_v2(ctx, ir, ["q?"], profile=profile)
        )

        # Two audits: the first raised an issue, the second confirmed
        # the patch resolved it. revise_patch only ran once.
        assert calls["audit"] == 2
        assert calls["revise_patch"] == 1
        assert calls["revise"] == 0, "patch mode must not call legacy revise()"
        # The patch *was* applied: chapters[0].points[0].text changed.
        assert latest_ir.chapters[0].points[0].text == "P7 patched 正文点 1"
        assert rounds == 1
        # Telemetry: pipeline.py copies this verbatim into stats['reviser'].
        rev = telemetry["reviser"]
        assert rev["mode"] == "patch"
        assert rev["proposed"] == 1
        assert rev["applied"] == 1
        assert rev["rejected"] == 0
        assert rev["unfixable"] == 0
        assert rev["schema_rollbacks"] == 0
        # Final critique reflects the second (clean) audit.
        assert critique.verdict == "ok"

    def test_full_mode_falls_back_to_legacy_revise(self):
        """``LECTURE_REVISER_MODE=full`` opt-in: audit → revise (not revise_patch)."""
        calls = {"audit": 0, "revise_patch": 0, "revise": 0}

        class _StubAgent:
            async def audit(self, context):
                calls["audit"] += 1
                return _make_critique(with_issues=True)

            async def revise_patch(self, *, context, issues):
                calls["revise_patch"] += 1
                return RevisePatchResult()

            async def revise(self, ctx, ir_json, issues):
                calls["revise"] += 1
                # Mimic a Reviser that returns None (no productive change)
                # so the loop exits after one round without a re-validation.
                return None, {"prompt_tokens": 9, "completion_tokens": 11, "total_tokens": 20}

        builder = _builder_with_stub(_StubAgent())
        ctx = _make_ctx(duration=900.0)
        ir = _make_ir(ctx)
        profile = self._profile("standard", critic_mode="full", reviser_mode="full")

        critique, latest_ir, rounds, usages, telemetry = _run(
            builder._run_critic_loop_v2(ctx, ir, ["q?"], profile=profile)
        )

        assert calls["audit"] == 1
        assert calls["revise"] == 1
        assert calls["revise_patch"] == 0
        # revise() returned None → no IR mutation, but audit ran.
        assert latest_ir is ir
        assert telemetry["reviser"]["mode"] == "full"
        assert critique.verdict == "needs_revision"

    def test_audit_failure_is_treated_as_skip_not_pipeline_fatal(self):
        """A malformed Critic response must not kill the lecture pipeline.

        Real runs can reach this branch when the audit call returns a clipped
        or otherwise malformed JSON object. The builder should keep the
        current IR and degrade to a no-op critique instead of propagating the
        parser failure through strict mode.
        """
        calls = {"audit": 0}

        class _StubAgent:
            async def audit(self, context):
                calls["audit"] += 1
                raise json.JSONDecodeError("Expecting ',' delimiter", '{"x": 1', 7)

        builder = _builder_with_stub(_StubAgent())
        ctx = _make_ctx(duration=900.0)
        ir = _make_ir(ctx)
        profile = self._profile("standard", critic_mode="full", reviser_mode="patch")

        critique, latest_ir, rounds, usages, telemetry = _run(
            builder._run_critic_loop_v2(ctx, ir, ["q?"], profile=profile)
        )

        assert calls["audit"] == 1
        assert latest_ir is ir
        assert rounds == 0
        assert usages == []
        assert telemetry["critic_sec"] == 0.0
        assert telemetry["reviser_sec"] == 0.0
        assert critique.verdict == "ok"
        assert critique.summary == "(critic skipped after failure)"


# ---------------------------------------------------------------------------
# 5. build_with_agents long profile dispatches to MapReduceIRBuilder
# ---------------------------------------------------------------------------


class TestBuildWithAgentsRouting:
    def test_long_profile_dispatches_to_map_reduce_builder(self, monkeypatch):
        """``profile.use_map_reduce=True`` → MapReduceIRBuilder is constructed
        with the chapter_plan + cache and its ``build`` is awaited instead
        of the single-pass ``_call`` round-trip.
        """
        ctx = _make_ctx(duration=2200.0, bv="BV1p7long000")
        ir = _make_ir(ctx)

        captured: dict[str, Any] = {}

        class _StubMapReduceBuilder:
            def __init__(self, *, client, settings, profile, chapter_cache):
                captured["init"] = {
                    "profile_name": profile.name,
                    "chapter_cache": chapter_cache,
                }

            async def build(self, *, chapter_plan, segments, frames, meta, study_questions=None):
                captured["build"] = {
                    "anchors": len(chapter_plan),
                    "study_q": list(study_questions or []),
                    "meta_bv": meta.bv_id,
                }
                # Lightweight stand-in for MapReduceStats from ir_map_reduce.
                class _Stats:
                    map_calls = 4
                    cache_hits = 2
                    cache_writes = 2
                    map_failures = ()
                    map_total_sec = 0.5
                    reduce_local_sec = 0.1
                    reduce_global_sec = 0.05
                    reduce_global_attempts = 1

                return ir, _Stats()

        # Patch the lazy import target inside ir_builder.build_with_agents.
        import app.understand.ir_map_reduce as mr_mod

        monkeypatch.setattr(mr_mod, "MapReduceIRBuilder", _StubMapReduceBuilder)

        # Stub the optional study-question agent so we don't hit the LLM.
        class _StubStudy:
            async def generate(self, ctx):
                from app.understand.agents import StudyQuestionsResult

                return StudyQuestionsResult(
                    questions=["P7 study?"],
                    usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    warnings=[],
                    metrics={},
                )

        # Stub the critic so the loop exits without any LLM round-trip.
        class _StubCritic:
            async def audit(self, context):
                return _make_critique(with_issues=False)

            async def revise_patch(self, *, context, issues):  # pragma: no cover
                return RevisePatchResult()

            async def revise(self, ctx, ir_json, issues):  # pragma: no cover
                return None, {}

        builder = _builder_with_stub(_StubCritic())
        builder._study_agent = _StubStudy()  # type: ignore[assignment]

        s = get_settings()
        long_profile = LectureProfile(
            name="long",
            duration_sec=2200.0,
            use_chapter_planner=True,
            use_map_reduce=True,
            use_chapter_cache=True,
            critic_mode="projected",
            reviser_mode="patch",
            study_question_mode="multi_window",
            chapter_planner_mode="structural",
        )

        # A non-None chapter_cache marker so we can assert it was passed
        # through verbatim (the real cache is exercised in P3 tests).
        sentinel_cache = object()

        out_ir, stats = _run(
            builder.build_with_agents(
                ctx,
                profile=long_profile,
                chapter_plan=["anchor-a", "anchor-b", "anchor-c"],
                chapter_cache=sentinel_cache,
            )
        )

        assert out_ir is ir
        assert captured["init"]["profile_name"] == "long"
        assert captured["init"]["chapter_cache"] is sentinel_cache
        assert captured["build"]["anchors"] == 3
        assert captured["build"]["study_q"] == ["P7 study?"]
        assert captured["build"]["meta_bv"] == "BV1p7long000"

        # Stats must surface map_reduce telemetry block; pipeline.py
        # copies it into timing['map_reduce'] verbatim.
        assert stats["map_reduce"]["map_calls"] == 4
        assert stats["map_reduce"]["cache_hits"] == 2
        assert stats["map_reduce"]["reduce_global_attempts"] == 1


# ---------------------------------------------------------------------------
# 6. Banner format string includes the new P7 routing fields
# ---------------------------------------------------------------------------


class TestStartupBanner:
    def test_banner_contains_p7_routing_tokens(self):
        """``app/main.py`` must log ``profile_router=on`` /
        ``chapter_planner=on`` / ``reviser_mode=...`` so an operator
        can read the routing decision off a single startup line."""
        from app import main as app_main

        src = Path(app_main.__file__).read_text(encoding="utf-8")
        assert "profile_router=on" in src
        assert "chapter_planner=on" in src
        assert "reviser_mode=%s" in src
        # The legacy chapter_cache + vlm_tiering / vlm_cache flags must
        # still be on the same banner so the M3 telemetry segment reading
        # does not regress.
        assert "chapter_cache=%s" in src
        assert "vlm_tiering=%s" in src
        assert "vlm_cache=%s" in src
