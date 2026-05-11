"""Build a per-profile :class:`CriticContext` for the Critic LLM (M2 P5).

The Critic stage has historically packed *everything* the Critic could
possibly want — full IR + full subtitle + every keyframe + study
questions — into one prompt. That works for ≤25 min videos but blows
up the token / latency budget on the 60+ min epic profile, where a
single Critic call would take ~2 minutes and reliably trip the LLM's
context limit.

P5 splits the input-packing concern out of :class:`CriticReviserAgent`
so the upcoming long / epic path can swap to a *projected* view:

* **Full mode** (standard profile) keeps the M1 behaviour: complete IR
  payload, full segments, all frames, ``skip_quote_validation=False``
  so the Critic still polices ``point.quote`` literal accuracy.

* **Projected mode** (long / epic profile) ships only what makes sense
  to audit at the lecture level: ``lecture_summary`` /
  ``mainline`` / ``glossary`` / ``cross_references`` and a per-chapter
  shell (``title / start_sec / end_sec / summary / points_count /
  code_blocks_count / unverified``). Segments and frames are dropped
  entirely — the projected Critic checks structural invariants only,
  per spec §3.5's 5-item checklist. ``skip_quote_validation`` is True
  because the source-of-truth quote evidence (subtitles) is no longer
  in the prompt.

This module is **side-effect free**: no LLM client, no SQLite, no
network. It exists to be unit-tested in isolation; the actual call
into ``audit()`` is wired by P7 once the patch Reviser (P6) is in
place.

Calling :func:`build_critic_context` with a profile whose
``critic_mode`` is ``"off"`` raises :class:`ValueError` instead of
silently returning a half-formed context — the gate belongs upstream
(at :meth:`LectureIRBuilder._run_critic_loop` or wherever the Critic
is dispatched), and a loud failure here is preferable to shipping an
empty prompt to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .profile import LectureProfile

CriticContextMode = Literal["full", "projected"]


@dataclass(frozen=True)
class CriticContext:
    """Bundle of inputs the Critic LLM needs for one audit call.

    Frozen because the dispatcher (``CriticReviserAgent.audit`` in P7)
    treats it as immutable — accidentally mutating ``ir_payload`` or
    ``segments_payload`` would let a downstream stage poison the
    prompt for sibling calls in the same builder run.
    """

    profile_name: str
    mode: CriticContextMode
    ir_payload: dict
    segments_payload: list[dict] | None
    frames_payload: list[dict] | None
    study_questions_payload: list[dict]
    chapter_issue_digest: list[dict] | None
    skip_quote_validation: bool


# ---- helpers --------------------------------------------------------


def _ir_to_dict(ir: Any) -> dict:
    """Best-effort conversion of an IR-like input to a plain dict.

    Accepts a Pydantic model (``model_dump`` if present), a dict
    (returned as-is, deep-copied via ``json.loads(json.dumps(...))``
    style would be paranoid here — we only project; the caller owns
    the original), or anything else (raises ``TypeError``).
    """
    if hasattr(ir, "model_dump"):
        return ir.model_dump()
    if isinstance(ir, dict):
        return ir
    raise TypeError(
        f"build_critic_context expected IR as Pydantic model or dict, "
        f"got {type(ir).__name__}"
    )


def _seg_payload(seg: Any) -> dict:
    """Normalise one subtitle segment to ``{ts, end, text}``.

    Mirrors :class:`app.ingest.subtitle.SubtitleSegment` field names
    (``start`` / ``end`` / ``text``); duck-typed so test fakes /
    future ``SubtitleSegmentLite`` adapters work without a code
    change here.
    """
    start = getattr(seg, "start", None)
    if start is None:
        start = getattr(seg, "start_sec", 0.0)
    end = getattr(seg, "end", None)
    if end is None:
        end = getattr(seg, "end_sec", start)
    text = getattr(seg, "text", "") or ""
    return {"ts": float(start), "end": float(end), "text": text}


def _frame_payload(frame: Any) -> dict:
    """Normalise one keyframe descriptor to a JSON-safe dict.

    Adapts both :class:`app.understand.vlm.FrameDescription`
    (``timestamp`` / ``ocr_text`` / ``caption``) and the newer
    duck-typed shape used by the chapter cache (``ts`` / ``ocr`` /
    ``caption``).
    """
    ts = getattr(frame, "ts", None)
    if ts is None:
        ts = getattr(frame, "timestamp", 0.0)
    ocr = getattr(frame, "ocr", None)
    if ocr is None:
        ocr = getattr(frame, "ocr_text", "")
    return {
        "ts": float(ts),
        "ocr": (ocr or "").strip(),
        "caption": (getattr(frame, "caption", "") or "").strip(),
        "visual_type": str(getattr(frame, "visual_type", "other") or "other"),
    }


def _study_question_payload(sq: Any) -> dict:
    """Wrap a study-question entry into a ``{question}`` dict.

    The pipeline currently produces ``list[str]`` (M1) but the spec
    schema is ``list[dict]`` so projected-mode prompts can attach
    extra metadata (chapter index, evidence hint) in future phases
    without breaking the Critic prompt template.
    """
    if isinstance(sq, dict):
        # Already structured — pass through, but coerce a missing
        # ``question`` field from ``q`` / string fallback.
        if "question" not in sq:
            for key in ("q", "text", "title"):
                if key in sq:
                    return {**sq, "question": str(sq[key])}
        return sq
    return {"question": str(sq)}


def _project_chapter(ch: dict) -> dict:
    """Strip a chapter dict down to the spec §3.5 projected shape.

    The 7 keys kept are exactly the lecture-level audit surface;
    everything else (points, code_blocks, formula_blocks, frames,
    teaching_notes, ...) is collapsed to a count or dropped.
    Critically, ``points`` is replaced with ``points_count`` so the
    Critic prompt cannot leak ``point.quote`` strings into a context
    that has ``skip_quote_validation=True``.
    """
    return {
        "title": str(ch.get("title", "")),
        "start_sec": float(ch.get("start", ch.get("start_sec", 0.0)) or 0.0),
        "end_sec": float(ch.get("end", ch.get("end_sec", 0.0)) or 0.0),
        "summary": str(ch.get("summary", "")),
        "points_count": len(ch.get("points") or []),
        "code_blocks_count": len(ch.get("code_blocks") or []),
        "unverified": bool(ch.get("unverified", False)),
    }


def _project_ir(ir_dict: dict) -> dict:
    """Build the projected IR payload for long / epic profiles.

    Falls back gracefully when M1-era IRs are missing the post-P4
    fields (``glossary`` / ``cross_references``); the empty list is
    not surprising to the Critic and keeps the prompt schema stable
    across pre-/post-map-reduce pipelines.
    """
    chapters = ir_dict.get("chapters") or []
    return {
        "lecture_summary": (
            ir_dict.get("lecture_summary")
            or ir_dict.get("final_synthesis")
            or ir_dict.get("summary")
            or ""
        ),
        "mainline": list(ir_dict.get("mainline") or []),
        "glossary": list(ir_dict.get("glossary") or []),
        "cross_references": list(ir_dict.get("cross_references") or []),
        "chapters": [_project_chapter(ch) for ch in chapters if isinstance(ch, dict)],
    }


# ---- public API -----------------------------------------------------


def build_critic_context(
    profile: LectureProfile,
    *,
    ir: Any,
    segments: Iterable[Any],
    frames: Iterable[Any],
    study_questions: Iterable[Any],
    chapter_issue_digest: list[dict] | None = None,
) -> CriticContext:
    """Pack Critic inputs according to ``profile.critic_mode``.

    Raises :class:`ValueError` when the profile says the Critic is
    off — see module docstring for the rationale (loud failure beats
    a silently malformed prompt).
    """
    mode = profile.critic_mode
    if mode == "off":
        raise ValueError(
            f"profile {profile.name!r} has critic_mode='off'; the caller "
            f"must short-circuit *before* invoking build_critic_context."
        )

    ir_dict = _ir_to_dict(ir)
    sq_payload = [_study_question_payload(q) for q in study_questions]

    if mode == "full":
        return CriticContext(
            profile_name=profile.name,
            mode="full",
            ir_payload=ir_dict,
            segments_payload=[_seg_payload(s) for s in segments],
            frames_payload=[_frame_payload(f) for f in frames],
            study_questions_payload=sq_payload,
            chapter_issue_digest=chapter_issue_digest,
            skip_quote_validation=False,
        )

    # mode == "projected"
    return CriticContext(
        profile_name=profile.name,
        mode="projected",
        ir_payload=_project_ir(ir_dict),
        segments_payload=None,
        frames_payload=None,
        study_questions_payload=sq_payload,
        chapter_issue_digest=chapter_issue_digest,
        skip_quote_validation=True,
    )


__all__ = [
    "CriticContext",
    "CriticContextMode",
    "build_critic_context",
]
