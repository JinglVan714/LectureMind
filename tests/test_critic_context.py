"""Unit tests for :mod:`app.understand.critic_context` (M2 P5).

The 6 cases follow the impl plan §5.2 step 1 list verbatim. They
exercise :func:`build_critic_context` against the four M2
:class:`LectureProfile` templates returned by ``_PROFILE_TABLE``:

  * tiny      — ``critic_mode="off"``  (raises)
  * standard  — ``critic_mode="full"`` (full segments + frames)
  * long      — ``critic_mode="projected"`` (segments / frames None)
  * epic      — ``critic_mode="projected"`` (same as long, sanity check)

The projected-mode tests double-check the IR projection drops every
``quote`` / ``text`` / ``ts`` field underneath ``points`` so the
Critic prompt cannot leak quote-level detail into a context where
quote validation is disabled (``skip_quote_validation=True``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from app.understand.critic_context import (
    CriticContext,
    build_critic_context,
)
from app.understand.profile import LectureProfile, _PROFILE_TABLE


# ---------- helpers ---------------------------------------------------


@dataclass
class _Seg:
    start: float
    end: float
    text: str


@dataclass
class _Frm:
    timestamp: float
    ocr_text: str
    caption: str
    visual_type: str = "slide"


def _ir_dict() -> dict:
    """Realistic IR dict containing every field projected mode strips."""
    return {
        "bv_id": "BVTEST",
        "title": "Sample lecture",
        "duration": 1800.0,
        "final_synthesis": "Three-line summary of the video.",
        "mainline": ["先 A", "再 B", "最后 C"],
        "glossary": [{"term": "X", "definition": "x def"}],
        "cross_references": [
            {"from_chapter": 1, "to_chapter": 2, "relation": "depends_on"}
        ],
        "chapters": [
            {
                "index": 1,
                "title": "Intro",
                "start": 0.0,
                "end": 600.0,
                "summary": "Intro summary.",
                "points": [
                    {"text": "P1", "ts": 30.0, "quote": "VERBATIM-QUOTE-1"},
                    {"text": "P2", "ts": 90.0, "quote": "VERBATIM-QUOTE-2"},
                ],
                "code_blocks": [
                    {"language": "python", "code": "x=1", "ts": 60.0}
                ],
                "unverified": False,
            },
            {
                "index": 2,
                "title": "Body",
                "start": 600.0,
                "end": 1500.0,
                "summary": "Body summary.",
                "points": [
                    {"text": "P3", "ts": 700.0, "quote": "VERBATIM-QUOTE-3"}
                ],
                "code_blocks": [],
                "unverified": True,
            },
        ],
    }


def _segments() -> list[_Seg]:
    return [
        _Seg(start=0.0, end=5.0, text="Hello"),
        _Seg(start=5.0, end=10.0, text="World"),
    ]


def _frames() -> list[_Frm]:
    return [
        _Frm(timestamp=2.5, ocr_text="ocr-A", caption="cap-A"),
        _Frm(timestamp=7.5, ocr_text="ocr-B", caption="cap-B"),
    ]


def _study_questions() -> list[str]:
    return ["What is X?", "How does Y work?"]


def _profile(name: str) -> LectureProfile:
    return _PROFILE_TABLE[name].with_duration(1800.0)


# ---------- tests -----------------------------------------------------


def test_build_critic_context_full_includes_segments_and_frames() -> None:
    """standard profile (mode=full) forwards segments + frames + full IR."""
    ctx = build_critic_context(
        _profile("standard"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )

    assert isinstance(ctx, CriticContext)
    assert ctx.profile_name == "standard"
    assert ctx.mode == "full"
    assert ctx.segments_payload is not None
    assert ctx.frames_payload is not None
    assert len(ctx.segments_payload) == 2
    assert len(ctx.frames_payload) == 2
    # The IR payload in full mode keeps point-level details (in particular
    # the quote field, which is exactly what the full-mode Critic uses
    # to validate quotes).
    serialised = json.dumps(ctx.ir_payload, ensure_ascii=False)
    assert "VERBATIM-QUOTE-1" in serialised
    assert ctx.skip_quote_validation is False
    assert ctx.chapter_issue_digest is None


def test_build_critic_context_projected_strips_segments_and_frames() -> None:
    """long profile (mode=projected) drops segments and frames entirely.

    The projected Critic is supposed to audit lecture-level structure
    only; trimming the heavy fields is what makes its prompt fit in
    one round-trip on a 90 min video. Forgetting to strip them here
    would silently re-introduce the long-video latency regression
    that motivated map-reduce in the first place.
    """
    ctx = build_critic_context(
        _profile("long"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )
    assert ctx.mode == "projected"
    assert ctx.segments_payload is None
    assert ctx.frames_payload is None


def test_build_critic_context_projected_omits_point_quote() -> None:
    """projected ir_payload contains no ``quote`` / ``text`` / ``ts`` from points.

    Spec §3.5 chapter projection keeps ``title / start_sec / end_sec /
    summary / points_count / code_blocks_count / unverified`` only.
    Anything below ``points`` is collapsed into the integer count, so
    the surface JSON cannot leak the original quote strings.
    """
    ctx = build_critic_context(
        _profile("long"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )
    serialised = json.dumps(ctx.ir_payload, ensure_ascii=False)
    assert "VERBATIM-QUOTE-1" not in serialised
    assert "VERBATIM-QUOTE-2" not in serialised
    assert "VERBATIM-QUOTE-3" not in serialised

    # Per-chapter shape contract.
    chapters = ctx.ir_payload["chapters"]
    assert len(chapters) == 2
    assert {c["title"] for c in chapters} == {"Intro", "Body"}
    for proj in chapters:
        # Required projected keys.
        for key in (
            "title",
            "start_sec",
            "end_sec",
            "summary",
            "points_count",
            "code_blocks_count",
            "unverified",
        ):
            assert key in proj, f"projected chapter missing {key}"
        # Stripped keys must NOT be present.
        for forbidden in ("points", "code_blocks"):
            assert forbidden not in proj, (
                f"projected chapter must not contain raw {forbidden!r}"
            )

    # points_count / code_blocks_count match the source dict.
    assert chapters[0]["points_count"] == 2
    assert chapters[0]["code_blocks_count"] == 1
    assert chapters[1]["unverified"] is True

    # The epic profile mirrors long's projection.
    ctx_epic = build_critic_context(
        _profile("epic"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )
    assert ctx_epic.mode == "projected"
    assert ctx_epic.ir_payload["chapters"][0] == chapters[0]


def test_build_critic_context_projected_keeps_study_questions() -> None:
    """study_questions survive projection — they drive coverage check #5."""
    ctx = build_critic_context(
        _profile("long"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )
    assert ctx.study_questions_payload
    # Either list[str] passthrough or list[dict] with ``question`` field —
    # whichever the implementation picks, the original strings must be
    # recoverable verbatim.
    serialised = json.dumps(ctx.study_questions_payload, ensure_ascii=False)
    assert "What is X?" in serialised
    assert "How does Y work?" in serialised


def test_build_critic_context_skip_quote_validation_set_correctly() -> None:
    """full → skip=False; projected → skip=True (spec §3.5)."""
    full = build_critic_context(
        _profile("standard"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )
    projected = build_critic_context(
        _profile("long"),
        ir=_ir_dict(),
        segments=_segments(),
        frames=_frames(),
        study_questions=_study_questions(),
    )
    assert full.skip_quote_validation is False
    assert projected.skip_quote_validation is True


def test_build_critic_context_critic_mode_off_raises_or_returns_marker() -> None:
    """tiny profile (critic_mode='off') has no defined CriticContext.

    Callers must check ``profile.critic_mode != 'off'`` before
    invoking ``build_critic_context``; if they don't, the function
    raises a ``ValueError`` so the bug surfaces loudly rather than
    silently shipping an incomplete context to the LLM.
    """
    with pytest.raises(ValueError) as excinfo:
        build_critic_context(
            _profile("tiny"),
            ir=_ir_dict(),
            segments=_segments(),
            frames=_frames(),
            study_questions=_study_questions(),
        )
    assert "off" in str(excinfo.value).lower()
