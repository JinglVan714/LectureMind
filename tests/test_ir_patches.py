"""Unit tests for :mod:`app.understand.ir_patches` (M2 P6).

The 13 cases follow plan §6.2 step 1 verbatim. They cover the
``IRPatch`` whitelist, ``apply_patches`` rollback semantics, and the
new ``CriticReviserAgent.revise_patch`` method.

Why patch Reviser? The legacy ``revise()`` rewrites the **entire**
LectureIR JSON in one LLM call, which on long lectures regularly
times out (600s) and silently drops protected arrays even when a
single ``code_block.explanation`` was the only target. The patch
Reviser instead emits a small list of ``IRPatch`` operations against
a strict whitelist; ``apply_patches`` mutates ``ir.model_dump()`` and
re-validates — any whitelist breach (or unintended schema corruption)
rolls the IR back to its pre-patch state.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from app.understand.agents import CriticReviserAgent, CritiqueIssue
from app.understand.critic_context import build_critic_context
from app.understand.ir import IRChapter, IRCodeBlock, IRPoint, KnowledgeUnit, LectureIR
from app.understand.ir_patches import (
    IRPatch,
    PatchRejected,
    RejectedPatch,
    RevisePatchResult,
    apply_patches,
)
from app.understand.profile import _PROFILE_TABLE


# ---------- helpers --------------------------------------------------


def _ir() -> LectureIR:
    """A small but realistic LectureIR for patch experiments."""
    return LectureIR(
        bv_id="BVPATCHTEST",
        title="Sample patch lecture",
        author="U",
        duration=900.0,
        chapters=[
            IRChapter(
                index=1,
                title="Intro",
                start=0.0,
                end=300.0,
                summary="intro summary",
                points=[
                    IRPoint(text="P1", ts=10.0, quote="ORIGINAL_QUOTE"),
                ],
                pitfalls=["existing pitfall"],
                code_blocks=[
                    IRCodeBlock(
                        language="python",
                        code="x = 1",
                        ts=20.0,
                        chapter_index=1,
                        source="ocr",
                        explanation="ORIGINAL_EXPLANATION",
                    )
                ],
            ),
            IRChapter(
                index=2,
                title="Body",
                start=300.0,
                end=600.0,
                summary="body summary",
                points=[IRPoint(text="P2", ts=400.0, quote="qb")],
            ),
            IRChapter(
                index=3,
                title="DECOY_CHAPTER_FAR_AWAY",
                start=600.0,
                end=900.0,
                summary="decoy summary that the patch reviser must not see",
                points=[
                    IRPoint(
                        text="DECOY_POINT_TEXT",
                        ts=700.0,
                        quote="DECOY_QUOTE_should_not_appear",
                    )
                ],
            ),
        ],
        knowledge_units=[
            KnowledgeUnit(
                id="ku-1",
                type="concept",
                title="Attention",
                explanation="ORIGINAL_KU_EXPL",
                ts=15.0,
                quote="kuq",
                chapter_index=1,
            )
        ],
    )


# ---------- apply_patches: positive cases ----------------------------


def test_apply_patches_replaces_code_blocks_array() -> None:
    ir = _ir()
    new_blocks = [
        {
            "language": "python",
            "code": "y = 2",
            "ts": 25.0,
            "chapter_index": 1,
            "source": "ocr",
            "explanation": "new block",
        }
    ]
    patch = IRPatch(op="replace_array", path="/chapters/0/code_blocks", value=new_blocks)
    revised, rejected = apply_patches(ir, [patch])
    assert rejected == []
    assert len(revised.chapters[0].code_blocks) == 1
    assert revised.chapters[0].code_blocks[0].code == "y = 2"
    assert revised.chapters[0].code_blocks[0].explanation == "new block"


def test_apply_patches_appends_to_pitfalls() -> None:
    ir = _ir()
    patch = IRPatch(
        op="append",
        path="/chapters/0/pitfalls/-",
        value="new pitfall",
    )
    revised, rejected = apply_patches(ir, [patch])
    assert rejected == []
    assert revised.chapters[0].pitfalls == ["existing pitfall", "new pitfall"]


def test_apply_patches_set_quote_on_point() -> None:
    ir = _ir()
    patch = IRPatch(
        op="set_quote",
        path="/chapters/0/points/0/quote",
        quote="REPAIRED_QUOTE",
    )
    revised, rejected = apply_patches(ir, [patch])
    assert rejected == []
    assert revised.chapters[0].points[0].quote == "REPAIRED_QUOTE"


def test_apply_patches_set_text_on_knowledge_unit() -> None:
    """KnowledgeUnit's user-facing label is ``title``; ``set_text``
    rewrites it (set_text/path /knowledge_units/{j}/title)."""
    ir = _ir()
    patch = IRPatch(
        op="set_text",
        path="/knowledge_units/0/title",
        text="Multi-Head Attention",
    )
    revised, rejected = apply_patches(ir, [patch])
    assert rejected == []
    assert revised.knowledge_units[0].title == "Multi-Head Attention"


def test_apply_patches_set_explanation_on_code_block() -> None:
    ir = _ir()
    patch = IRPatch(
        op="set_explanation",
        path="/chapters/0/code_blocks/0/explanation",
        text="Improved explanation: x is a literal int.",
    )
    revised, rejected = apply_patches(ir, [patch])
    assert rejected == []
    assert "Improved explanation" in revised.chapters[0].code_blocks[0].explanation


# ---------- apply_patches: rejection cases --------------------------


def test_apply_patches_rejects_replace_full_chapter() -> None:
    """``/chapters/0`` is NOT in the replace_array whitelist — rejected."""
    ir = _ir()
    patch = IRPatch(
        op="replace_array",
        path="/chapters/0",
        value=[{"index": 1, "title": "hijacked"}],
    )
    revised, rejected = apply_patches(ir, [patch])
    assert revised.chapters[0].title == "Intro"  # untouched
    assert len(rejected) == 1
    assert rejected[0].patch == patch
    assert "whitelist" in rejected[0].reason.lower() or "path" in rejected[0].reason.lower()


def test_apply_patches_rejects_modify_chapter_start_sec() -> None:
    """Structural fields (start / start_sec / end / title) are off-limits."""
    ir = _ir()
    # Try every conceivable op spelling for the chapter start field.
    cases = [
        IRPatch(op="set_text", path="/chapters/0/start_sec", text="0"),
        IRPatch(op="set_text", path="/chapters/0/start", text="0"),
        IRPatch(op="replace_array", path="/chapters/0/title", value="hijacked"),
    ]
    revised, rejected = apply_patches(ir, cases)
    assert revised.chapters[0].start == 0.0
    assert revised.chapters[0].title == "Intro"
    assert len(rejected) == 3
    for r in rejected:
        assert r.patch is not None
        assert r.patch in cases


def test_apply_patches_rejects_modify_point_ts() -> None:
    """``points[*].ts`` is timestamp truth — no patch may rewrite it."""
    ir = _ir()
    patch = IRPatch(
        op="set_text",
        path="/chapters/0/points/0/ts",
        text="999",
    )
    revised, rejected = apply_patches(ir, [patch])
    assert revised.chapters[0].points[0].ts == 10.0
    assert len(rejected) == 1


def test_apply_patches_rejects_unknown_op() -> None:
    ir = _ir()
    patch = IRPatch(
        op="set_severity",  # not in whitelist
        path="/chapters/0/points/0/quote",
        text="whatever",
    )
    revised, rejected = apply_patches(ir, [patch])
    assert revised.chapters[0].points[0].quote == "ORIGINAL_QUOTE"
    assert len(rejected) == 1
    assert "op" in rejected[0].reason.lower()


def test_apply_patches_schema_violation_rolls_back() -> None:
    """A whitelisted patch that produces an invalid IR rolls back the
    entire batch — the cheap belt-and-braces against an LLM that
    proposes a path-legal but schema-illegal value (e.g. negative ts
    in a code block, breaking ``IRCodeBlock.ts >= 0``).
    """
    ir = _ir()
    bad_blocks = [
        {
            "language": "python",
            "code": "z = 3",
            "ts": -1.0,  # violates IRCodeBlock.ts >= 0
            "chapter_index": 1,
            "source": "ocr",
            "explanation": "bad",
        }
    ]
    bad = IRPatch(op="replace_array", path="/chapters/0/code_blocks", value=bad_blocks)
    good = IRPatch(op="set_quote", path="/chapters/0/points/0/quote", quote="GOOD_QUOTE")
    revised, rejected = apply_patches(ir, [good, bad])
    # Rollback: even ``good`` must not have applied because the batch
    # ended up failing schema validation.
    assert revised is ir
    assert revised.chapters[0].points[0].quote == "ORIGINAL_QUOTE"
    # Rejected list contains a synthetic ``patch=None`` entry tagged as
    # ``schema_violation:...``.
    assert any(r.patch is None and r.reason.startswith("schema_violation") for r in rejected)


# ---------- CriticReviserAgent.revise_patch -------------------------


# Minimal async OpenAI client stub.

class _StubMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMsg(content)


class _StubResp:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]
        self.usage = None


class _StubCompletions:
    def __init__(self, parent: "_StubClient") -> None:
        self._p = parent

    async def create(self, **kwargs: Any) -> _StubResp:
        self._p.calls.append(kwargs)
        result = self._p.responder(kwargs, self._p)
        if isinstance(result, BaseException):
            raise result
        return _StubResp(json.dumps(result, ensure_ascii=False))


class _StubChat:
    def __init__(self, parent: "_StubClient") -> None:
        self.completions = _StubCompletions(parent)


class _StubClient:
    def __init__(self, *, responder: Callable[..., Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responder = responder or (lambda kwargs, client: {"patches": [], "unfixable_issues": []})
        self.chat = _StubChat(self)


def _issue(*, kind: str = "missing_code", location: str = "chapters[0].code_blocks") -> CritiqueIssue:
    return CritiqueIssue(
        kind=kind,
        severity="high",
        location=location,
        evidence="evidence",
        suggestion="suggestion",
    )


def _agent_with(client: _StubClient) -> CriticReviserAgent:
    return CriticReviserAgent(client=client)


def test_revise_patch_uses_local_input_only() -> None:
    """The patch Reviser prompt must NOT contain unrelated chapters'
    contents. Issues cite chapter 0 only — chapter 2's decoy strings
    (title "DECOY_CHAPTER_FAR_AWAY", quote "DECOY_QUOTE_should_not_appear")
    must therefore be absent from the user prompt body. This is the
    cost-saving claim of the patch Reviser: even on epic videos the
    prompt stays small because issues anchor the relevant slice.
    """
    captured: dict[str, Any] = {}

    def _capture(kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
        captured["kwargs"] = kwargs
        return {"patches": [], "unfixable_issues": []}

    client = _StubClient(responder=_capture)
    agent = _agent_with(client)
    ir = _ir()
    profile = _PROFILE_TABLE["standard"].with_duration(900.0)
    context = build_critic_context(
        profile,
        ir=ir,
        segments=[],
        frames=[],
        study_questions=["Q?"],
    )
    issues = [_issue(location="chapters[0].code_blocks")]
    import asyncio

    result = asyncio.run(agent.revise_patch(context=context, issues=issues))
    assert isinstance(result, RevisePatchResult)

    # Inspect the captured user message — chapter 2 decoys must not appear.
    user_msg = ""
    for m in captured["kwargs"]["messages"]:
        if m.get("role") == "user":
            user_msg = m.get("content", "")
    assert "DECOY_CHAPTER_FAR_AWAY" not in user_msg
    assert "DECOY_QUOTE_should_not_appear" not in user_msg
    assert "DECOY_POINT_TEXT" not in user_msg
    # Cited chapter content does survive — sanity check.
    assert "ORIGINAL_QUOTE" in user_msg or "ORIGINAL_EXPLANATION" in user_msg


def test_revise_patch_handles_unfixable_issues_gracefully() -> None:
    """When the LLM declares ``unfixable_issues`` the result preserves
    them verbatim and still returns ``patches=[]`` with no exception.
    """

    def _responder(kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
        return {
            "patches": [],
            "unfixable_issues": [
                "Chapter 0 missing prerequisite: linear algebra basics",
                "Cross-reference between chapter 1 and 3 cannot be inferred",
            ],
        }

    client = _StubClient(responder=_responder)
    agent = _agent_with(client)
    ir = _ir()
    profile = _PROFILE_TABLE["long"].with_duration(2000.0)
    context = build_critic_context(
        profile,
        ir=ir,
        segments=[],
        frames=[],
        study_questions=[],
    )
    issues = [_issue(kind="missing_prerequisite", location="chapters[0]")]
    import asyncio

    result = asyncio.run(agent.revise_patch(context=context, issues=issues))
    assert result.patches == []
    assert len(result.unfixable_issues) == 2
    assert "linear algebra" in result.unfixable_issues[0]


def test_reviser_mode_full_preserves_old_behavior() -> None:
    """``revise_patch`` is purely additive — the existing ``revise()``
    API and the legacy ``LECTURE_REVISER_MODE=full`` settings path
    stay byte-for-byte compatible.
    """
    sig = inspect.signature(CriticReviserAgent.revise)
    # Original signature: (self, ctx, ir_json, issues). Kept verbatim
    # so the M1 _run_critic_loop dispatch remains untouched.
    assert list(sig.parameters) == ["self", "ctx", "ir_json", "issues"]

    # ``revise_patch`` exists as a separate coroutine — does not shadow.
    assert hasattr(CriticReviserAgent, "revise_patch")
    assert inspect.iscoroutinefunction(CriticReviserAgent.revise_patch)

    # Settings path: explicit MODE=full survives the legacy compat
    # ``_apply_reviser_mode_compat`` model_validator.
    from app.config import Settings

    s = Settings(LECTURE_REVISER_MODE="full", LECTURE_REVISER_ENABLED=False)
    assert s.lecture_reviser_mode == "full"
