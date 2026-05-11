"""Deterministic chapter-anchor extractor (M2 P2).

The :func:`plan_chapters` entry point combines three deterministic
signals (transition-phrase regex on subtitle text, silent gaps, and
visual-type shifts in keyframes) into a list of ``ChapterAnchor``
objects covering the full video duration.

Two modes drive selection (decided by the caller's
:class:`~app.understand.profile.LectureProfile`):

* ``hint`` (tiny / standard) — keep up to ``chapters_max + 2`` strongest
  candidates and let the LLM finalise. Allowed to under-shoot
  ``chapters_min``.
* ``structural`` (long / epic) — enforce
  ``chapters_min ≤ N ≤ chapters_max`` strictly, with equal-split
  fallback for under-runs, greedy lowest-confidence merging for
  over-runs, segment-boundary snapping, and a per-chapter minimum
  width floor.

This module is dependency-free at runtime (no LLM, no SQLite) so it
can be unit-tested without any I/O. P2 only ships extraction; the
``pipeline_stats.chapter_plan_alignment`` telemetry is wired up in P7.

P2 deviation note:
    The spec §3.2 ``ChapterAnchor.source`` literal lists 4 values
    (``transition_phrase`` / ``silence`` / ``visual_shift`` /
    ``equal_split``). We strictly honour that union — for the
    *implicit first chapter* (which always starts at t=0) we tag the
    source as ``equal_split`` because that's the closest fit
    (structural placement, no detected signal). Tests treat this as
    expected behaviour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

from app.ingest.subtitle import SubtitleSegment

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Settings
    from app.understand.profile import LectureProfile


AnchorSource = Literal[
    "transition_phrase", "silence", "visual_shift", "equal_split"
]
AnchorMode = Literal["hint", "structural"]


@dataclass(frozen=True)
class ChapterAnchor:
    """A single chapter slice within a planned lecture.

    Attributes
    ----------
    start_sec, end_sec
        Half-open chapter window. ``end_sec`` of chapter *i* equals
        ``start_sec`` of chapter *i+1* (no gaps). The first chapter
        always starts at 0.0; the last ends at ``duration_sec``.
    confidence
        Confidence in **the start boundary**, in [0, 1]. The first
        chapter is always 1.0 (trivial structural anchor).
    anchor_text
        Short subtitle excerpt near ``start_sec`` for human-readable
        diagnostics (truncated to ~80 chars).
    source
        Which signal produced this anchor (see :data:`AnchorSource`).
    mode
        Whether ``plan_chapters`` ran in hint or structural mode.
    """

    start_sec: float
    end_sec: float
    confidence: float
    anchor_text: str
    source: AnchorSource
    mode: AnchorMode


# ---------------------------------------------------------------------------
# Internal candidate type — ``ChapterAnchor`` describes a *chapter*, but
# during collection we need to track candidate *boundaries* (i.e. potential
# start times mid-video) along with their metadata. We promote candidates to
# anchors only after selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    time_sec: float
    confidence: float
    anchor_text: str
    source: AnchorSource


# Transition-phrase patterns; each matches the BEGINNING of a subtitle
# segment text. Kept conservative to avoid false positives on filler
# speech ("那么" alone is too common).
# NOTE: alternation order matters — list longer prefixes first so e.g.
# ``来看`` is preferred over ``看`` for the input ``我们来看下一部分``.
_TRANSITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(那么|接下来|下面|现在)我们(来看|看|来讲|讲|开始|进入|学习|讨论|分析)"),
    re.compile(r"^(总结|小结|最后|综上|回顾)"),
    re.compile(r"^第[一二三四五六七八九十\d]+(部分|章|节|步|个|阶段)"),
    re.compile(r"^首先[，,]?"),
    re.compile(r"^其次[，,]?"),
)


# Visual-shift detection looks for *category* changes between adjacent
# frames (slide_text → code, person → diagram, etc.). Within-category
# variation (e.g. two "slide_text" frames with different OCR) is not a
# chapter boundary signal.
_VISUAL_SHIFT_TOLERANCE_SEC = 2.0


def plan_chapters(
    *,
    profile: "LectureProfile",
    duration_sec: float,
    segments: list[SubtitleSegment],
    frames: Sequence[Any],
    chapters_min: int,
    chapters_max: int,
    settings: "Settings",
) -> list[ChapterAnchor]:
    """Plan chapter boundaries for a single lecture.

    Parameters
    ----------
    profile
        :class:`~app.understand.profile.LectureProfile` carrying the
        ``chapter_planner_mode`` (``hint`` or ``structural``) and the
        ``use_chapter_planner`` master switch.
    duration_sec
        Total video duration (seconds). Used as the implicit
        ``end_sec`` of the last chapter.
    segments
        Ordered subtitle segments. Each must expose ``.start``,
        ``.end`` and ``.text`` (see :class:`SubtitleSegment`).
    frames
        Keyframe descriptions. Each must expose ``.timestamp`` and
        ``.visual_type``; we accept any duck-typed object so tests can
        avoid importing the full VLM stack.
    chapters_min, chapters_max
        Inclusive bounds on the number of chapters. Typically supplied
        by :class:`~app.understand.length_adapt.LengthBudget`.
    settings
        :class:`~app.config.Settings` instance carrying the planner
        config (master switch, min duration, signal confidences, etc).

    Returns
    -------
    list[ChapterAnchor]
        Empty list when planning is gated off (disabled, sub-min-sec,
        or profile opts out). Otherwise a non-empty contiguous
        sequence covering ``[0, duration_sec]``.
    """

    # -- Gating ------------------------------------------------------------
    if not settings.lecture_chapter_planner_enabled:
        return []
    if duration_sec < settings.lecture_chapter_planner_min_sec:
        return []
    if not profile.use_chapter_planner:
        return []
    if chapters_max < 1:
        return []

    mode: AnchorMode = profile.chapter_planner_mode  # "hint" | "structural"

    # -- Collect candidate boundaries (mid-video splits) -------------------
    candidates = _collect_candidates(segments, frames, settings)
    candidates = _snap_to_segment_boundaries(candidates, segments)
    # De-duplicate near-coincident candidates (same boundary detected
    # by multiple signals); keep the highest-confidence one.
    candidates = _dedupe_candidates(candidates)

    # -- Mode dispatch -----------------------------------------------------
    if mode == "structural":
        return _build_structural(
            candidates,
            duration_sec=duration_sec,
            segments=segments,
            chapters_min=chapters_min,
            chapters_max=chapters_max,
        )
    return _build_hint(
        candidates,
        duration_sec=duration_sec,
        segments=segments,
        chapters_max=chapters_max,
    )


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------


def _collect_candidates(
    segments: list[SubtitleSegment],
    frames: Sequence[Any],
    settings: "Settings",
) -> list[_Candidate]:
    out: list[_Candidate] = []

    # 1) Transition phrases (strongest signal).
    tp_conf = settings.lecture_planner_transition_phrase_confidence
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        for pat in _TRANSITION_PATTERNS:
            if pat.match(text):
                out.append(
                    _Candidate(
                        time_sec=float(seg.start),
                        confidence=tp_conf,
                        anchor_text=text[:80],
                        source="transition_phrase",
                    )
                )
                break  # one match per segment is enough

    # 2) Silent gaps between adjacent segments.
    silence_sec = settings.lecture_planner_silence_sec
    silence_conf = settings.lecture_planner_silence_confidence
    for prev, nxt in zip(segments, segments[1:]):
        gap = float(nxt.start) - float(prev.end)
        if gap >= silence_sec:
            out.append(
                _Candidate(
                    time_sec=float(nxt.start),
                    confidence=silence_conf,
                    anchor_text=(nxt.text or "")[:80],
                    source="silence",
                )
            )

    # 3) Visual-type shifts in adjacent frames, validated against a
    #    nearby segment boundary (within ±_VISUAL_SHIFT_TOLERANCE_SEC).
    visual_conf = settings.lecture_planner_visual_shift_confidence
    seg_boundaries = sorted({float(s.start) for s in segments})
    sorted_frames = sorted(frames, key=lambda f: float(getattr(f, "timestamp", 0.0)))
    for prev, nxt in zip(sorted_frames, sorted_frames[1:]):
        prev_type = str(getattr(prev, "visual_type", "") or "")
        nxt_type = str(getattr(nxt, "visual_type", "") or "")
        if not prev_type or not nxt_type or prev_type == nxt_type:
            continue
        shift_t = float(getattr(nxt, "timestamp", 0.0))
        # Snap to nearest segment boundary if within tolerance.
        nearest = _nearest_value(seg_boundaries, shift_t)
        if nearest is None:
            continue
        if abs(nearest - shift_t) > _VISUAL_SHIFT_TOLERANCE_SEC:
            continue
        anchor_text = _segment_text_at(segments, nearest)
        out.append(
            _Candidate(
                time_sec=nearest,
                confidence=visual_conf,
                anchor_text=anchor_text[:80],
                source="visual_shift",
            )
        )

    # Drop the trivial t=0 candidate; the implicit first chapter
    # always starts at 0 and we don't want it competing for slots.
    out = [c for c in out if c.time_sec > 0.0]
    out.sort(key=lambda c: c.time_sec)
    return out


def _snap_to_segment_boundaries(
    candidates: list[_Candidate], segments: list[SubtitleSegment]
) -> list[_Candidate]:
    """Snap each candidate's ``time_sec`` to the nearest subtitle
    boundary within ±_VISUAL_SHIFT_TOLERANCE_SEC. Subtitle-derived
    candidates are already at exact boundaries; only frame-derived
    ones can drift, so this is mostly a no-op for them.
    """
    if not segments:
        return candidates
    boundaries = sorted({float(s.start) for s in segments} | {float(s.end) for s in segments})
    snapped: list[_Candidate] = []
    for c in candidates:
        nearest = _nearest_value(boundaries, c.time_sec)
        if nearest is None:
            snapped.append(c)
            continue
        if abs(nearest - c.time_sec) <= _VISUAL_SHIFT_TOLERANCE_SEC:
            if nearest != c.time_sec:
                c = _Candidate(
                    time_sec=nearest,
                    confidence=c.confidence,
                    anchor_text=c.anchor_text,
                    source=c.source,
                )
        snapped.append(c)
    return snapped


def _dedupe_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Collapse candidates that landed on the same time after snapping.
    Keep the highest-confidence one; ties broken by source priority
    (transition_phrase > silence > visual_shift).
    """
    priority = {"transition_phrase": 3, "silence": 2, "visual_shift": 1, "equal_split": 0}
    by_time: dict[float, _Candidate] = {}
    for c in candidates:
        existing = by_time.get(c.time_sec)
        if existing is None:
            by_time[c.time_sec] = c
            continue
        if c.confidence > existing.confidence or (
            c.confidence == existing.confidence
            and priority.get(c.source, 0) > priority.get(existing.source, 0)
        ):
            by_time[c.time_sec] = c
    return sorted(by_time.values(), key=lambda c: c.time_sec)


# ---------------------------------------------------------------------------
# Hint mode
# ---------------------------------------------------------------------------


def _build_hint(
    candidates: list[_Candidate],
    *,
    duration_sec: float,
    segments: list[SubtitleSegment],
    chapters_max: int,
) -> list[ChapterAnchor]:
    """Hint mode: keep top-K=chapters_max+2 strongest candidates,
    sort by time, materialise chapters. May under-shoot
    ``chapters_min`` — that's the contract; the LLM finalises."""
    k = max(1, chapters_max + 2)
    # Pick top-K by confidence (descending), then re-sort by time.
    top = sorted(candidates, key=lambda c: -c.confidence)[: max(0, k - 1)]
    top.sort(key=lambda c: c.time_sec)
    return _materialise(
        top,
        duration_sec=duration_sec,
        segments=segments,
        mode="hint",
    )


# ---------------------------------------------------------------------------
# Structural mode
# ---------------------------------------------------------------------------


def _build_structural(
    candidates: list[_Candidate],
    *,
    duration_sec: float,
    segments: list[SubtitleSegment],
    chapters_min: int,
    chapters_max: int,
) -> list[ChapterAnchor]:
    """Structural mode: enforce strict ``chapters_min ≤ N ≤ chapters_max``.

    Process:
      1. Start with all snapped candidates.
      2. If we'd over-shoot chapters_max, greedily drop the
         lowest-confidence boundary until we fit (greedy merge).
      3. If we'd under-shoot chapters_min, insert ``equal_split``
         boundaries at the largest existing chapter gaps until we
         reach chapters_min.
      4. Apply a per-chapter minimum width floor: if any chapter is
         shorter than ``duration / chapters_max / 3``, merge it with
         its predecessor.
      5. Re-check chapters_min after the floor pass and re-insert
         equal-split anchors if needed.
    """
    chapters_min = max(1, chapters_min)
    chapters_max = max(chapters_min, chapters_max)
    n_target_max_boundaries = chapters_max - 1  # boundaries; first chapter implicit

    # Step 1: take all candidates.
    boundaries = list(candidates)

    # Step 2: greedy-merge over-runs.
    if len(boundaries) > n_target_max_boundaries:
        # Sort by confidence descending and keep the strongest N.
        boundaries = sorted(boundaries, key=lambda c: -c.confidence)[: n_target_max_boundaries]
        boundaries.sort(key=lambda c: c.time_sec)

    # Step 3: equal-split fill for under-runs.
    boundaries = _fill_equal_split(
        boundaries,
        duration_sec=duration_sec,
        segments=segments,
        target_count=chapters_min - 1,
    )

    # Step 4: floor pass.
    boundaries = _enforce_min_width(
        boundaries,
        duration_sec=duration_sec,
        chapters_max=chapters_max,
    )

    # Step 5: re-fill if floor pass dropped us below chapters_min.
    boundaries = _fill_equal_split(
        boundaries,
        duration_sec=duration_sec,
        segments=segments,
        target_count=chapters_min - 1,
    )

    return _materialise(
        boundaries,
        duration_sec=duration_sec,
        segments=segments,
        mode="structural",
    )


def _fill_equal_split(
    boundaries: list[_Candidate],
    *,
    duration_sec: float,
    segments: list[SubtitleSegment],
    target_count: int,
) -> list[_Candidate]:
    """Insert ``equal_split`` boundaries at the largest gaps until we
    have at least ``target_count`` boundaries (i.e. ``target_count+1``
    chapters)."""
    if target_count <= 0:
        return boundaries
    if len(boundaries) >= target_count:
        return boundaries

    seg_starts = sorted({float(s.start) for s in segments}) if segments else []

    while len(boundaries) < target_count:
        # Build the current set of gap intervals (start, end) and pick
        # the widest one to bisect.
        times = [0.0] + [b.time_sec for b in boundaries] + [duration_sec]
        widest_i = max(
            range(len(times) - 1),
            key=lambda i: times[i + 1] - times[i],
        )
        midpoint = (times[widest_i] + times[widest_i + 1]) / 2.0
        if seg_starts:
            nearest = _nearest_value(seg_starts, midpoint)
            if nearest is not None and 0.0 < nearest < duration_sec:
                midpoint = nearest
        # Avoid degenerate insertions on the boundary itself.
        if midpoint <= 0.0 or midpoint >= duration_sec:
            break
        # If the midpoint equals an existing boundary (numerical
        # coincidence), nudge slightly to keep boundaries unique.
        if any(abs(midpoint - b.time_sec) < 0.5 for b in boundaries):
            break
        anchor_text = _segment_text_at(segments, midpoint)[:80]
        boundaries.append(
            _Candidate(
                time_sec=midpoint,
                confidence=1.0,  # structural placement; not LLM-overridable
                anchor_text=anchor_text or "（结构化等距插入）",
                source="equal_split",
            )
        )
        boundaries.sort(key=lambda c: c.time_sec)
    return boundaries


def _enforce_min_width(
    boundaries: list[_Candidate],
    *,
    duration_sec: float,
    chapters_max: int,
) -> list[_Candidate]:
    """Drop boundaries that produce chapters shorter than
    ``duration / chapters_max / 3`` (per spec §3.2). Merging happens
    by simply removing the offending boundary so the chapter fuses
    with its predecessor.
    """
    if not boundaries:
        return boundaries
    min_width = max(1.0, duration_sec / max(1, chapters_max) / 3.0)
    keep: list[_Candidate] = []
    last_end = 0.0
    for b in boundaries:
        width = b.time_sec - last_end
        if width < min_width and keep:
            # Merge: skip this boundary (current chapter eats into next).
            continue
        keep.append(b)
        last_end = b.time_sec
    # Final chapter (last boundary -> duration) width check.
    if keep and (duration_sec - keep[-1].time_sec) < min_width:
        keep.pop()
    return keep


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def _materialise(
    boundaries: list[_Candidate],
    *,
    duration_sec: float,
    segments: list[SubtitleSegment],
    mode: AnchorMode,
) -> list[ChapterAnchor]:
    """Convert ordered mid-video boundaries into a contiguous list of
    :class:`ChapterAnchor` covering ``[0, duration_sec]``."""

    chapters: list[ChapterAnchor] = []
    # First chapter (always starts at 0) — synthetic ``equal_split`` source.
    first_text = (segments[0].text[:80] if segments else "（开场）") or "（开场）"
    cursor = 0.0
    if boundaries:
        first_end = boundaries[0].time_sec
    else:
        first_end = duration_sec
    chapters.append(
        ChapterAnchor(
            start_sec=0.0,
            end_sec=float(first_end),
            confidence=1.0,
            anchor_text=first_text,
            source="equal_split",
            mode=mode,
        )
    )
    cursor = first_end
    for i, b in enumerate(boundaries):
        nxt = boundaries[i + 1].time_sec if i + 1 < len(boundaries) else duration_sec
        chapters.append(
            ChapterAnchor(
                start_sec=float(b.time_sec),
                end_sec=float(nxt),
                confidence=float(b.confidence),
                anchor_text=str(b.anchor_text),
                source=b.source,
                mode=mode,
            )
        )
        cursor = nxt
    return chapters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_value(sorted_values: Iterable[float], target: float) -> float | None:
    """Return the entry in ``sorted_values`` nearest to ``target``,
    or ``None`` for an empty input. Linear scan because callers
    operate on small lists (<1000 items)."""
    nearest: float | None = None
    best = float("inf")
    for v in sorted_values:
        d = abs(v - target)
        if d < best:
            best = d
            nearest = v
    return nearest


def _segment_text_at(segments: list[SubtitleSegment], t: float) -> str:
    """Return the text of the subtitle segment containing ``t``, or
    the closest preceding one. Empty string if no segments exist."""
    if not segments:
        return ""
    candidate = segments[0].text or ""
    for s in segments:
        if float(s.start) <= t <= float(s.end):
            return s.text or candidate
        if float(s.start) <= t:
            candidate = s.text or candidate
    return candidate


__all__ = [
    "ChapterAnchor",
    "AnchorSource",
    "AnchorMode",
    "plan_chapters",
]
