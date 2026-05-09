"""Length-adaptive ceilings for the lecture pipeline.

The original pipeline hard-coded all caps (chapters: 3-8, mainline: 6-8,
points/chapter: 2-5, keyframes: 8-20, glossary: 5-15, …) regardless of
video duration. A 44-minute technical lecture therefore produced almost
the same amount of structured content as an 11-minute conceptual talk,
collapsing per-minute information density by ~4×.

This module centralises the duration-aware ceilings so every consumer
(prompt builders, keyframe extractor, IR hydration, renderer) reads the
same numbers. The functions take ``duration`` in seconds and return
sensible integers.

The curves are deliberately gentle (linear with both a floor and a
ceiling) so:

* Short videos (<5 min) still produce a usable, compact lecture and
  match existing test snapshots.
* Long videos (≥30 min) get markedly more chapters, points, glossary
  entries and visual evidence to keep per-minute density roughly stable.
* Pathological inputs (multi-hour streams) cap out before exploding the
  prompt or the HTML page size.
"""
from __future__ import annotations

from dataclasses import dataclass


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def _minutes(duration_sec: float | int) -> float:
    try:
        return max(0.0, float(duration_sec)) / 60.0
    except (TypeError, ValueError):
        return 0.0


def chapters_target(duration_sec: float | int) -> tuple[int, int]:
    """Recommended ``(min_chapters, max_chapters)`` for a video.

    * <10 min  → 3-4
    * 10-30 min → 4-6
    * 30-60 min → 6-9
    * 60-90 min → 7-11
    * 90+ min  → cap at 12 to keep TOC scannable.
    """
    m = _minutes(duration_sec)
    if m < 10:
        return 3, 4
    if m < 30:
        return 4, 6
    if m < 60:
        return 6, 9
    if m < 90:
        return 7, 11
    return 8, 12


def mainline_max(duration_sec: float | int) -> int:
    """Cap on global ``mainline`` (cognitive path) entries."""
    m = _minutes(duration_sec)
    return _clip(round(4 + m / 7.5), 4, 14)


def points_per_chapter_max(chapter_seconds: float | int) -> int:
    """Cap on ``points`` per chapter, scaled with chapter length."""
    minutes = _minutes(chapter_seconds)
    return _clip(round(3 + minutes * 0.6), 3, 12)


def keyframe_window(
    duration_sec: float | int,
    *,
    base_min: int,
    base_max: int,
) -> tuple[int, int]:
    """Adaptive ``(keyframe_min, keyframe_max)`` enveloping ``base_*``.

    The user-configured ``KEYFRAME_MIN`` / ``KEYFRAME_MAX`` (defaults
    8 / 20) become **lower bounds** instead of absolute bounds, so long
    videos get more visual evidence without requiring config changes.
    """
    m = _minutes(duration_sec)
    adapt_min = max(base_min, _clip(round(m * 0.4), 4, 24))
    # 1.5 frames/min plus a safety floor; hard-cap at 80 to keep HTML
    # base64-inline size manageable (~10MB cover).
    adapt_max = max(base_max, _clip(round(m * 1.5 + 4), 8, 80))
    if adapt_max < adapt_min + 2:
        adapt_max = adapt_min + 2
    return adapt_min, adapt_max


def chapter_frame_max(chapter_seconds: float | int) -> int:
    """How many keyframes may attach to a single chapter."""
    minutes = _minutes(chapter_seconds)
    return _clip(round(2 + minutes / 4.0), 2, 8)


def global_visual_limit(duration_sec: float | int) -> int:
    """Cap on the lecture-level "全局关键图解" gallery.

    ≤ 10 min keeps the legacy ceiling of 5 (preserving short-video
    HTML density and matching golden tests). Longer videos scale up
    linearly, capped at 24 to avoid overwhelming the page.
    """
    m = _minutes(duration_sec)
    if m <= 10:
        return 5
    return _clip(round(5 + (m - 10) / 3.5), 5, 24)


def glossary_max(duration_sec: float | int) -> int:
    m = _minutes(duration_sec)
    return _clip(round(8 + m / 3.0), 5, 30)


def review_questions_max(duration_sec: float | int) -> int:
    m = _minutes(duration_sec)
    return _clip(round(4 + m / 5.0), 3, 16)


def study_questions_target(duration_sec: float | int) -> tuple[int, int]:
    """Question-Driven extraction: how many study questions to pre-generate."""
    m = _minutes(duration_sec)
    if m < 10:
        return 4, 6
    if m < 30:
        return 5, 8
    if m < 60:
        return 6, 10
    return 7, 12


@dataclass(frozen=True)
class LengthBudget:
    """Snapshot of every duration-derived ceiling for a single video.

    Attached to ``LecturizeContext`` so prompts and post-processing read
    a single source of truth instead of recomputing piecemeal.
    """

    duration_sec: float
    chapters_min: int
    chapters_max: int
    mainline_max: int
    global_visual_limit: int
    glossary_max: int
    review_questions_max: int
    keyframe_min: int
    keyframe_max: int
    study_questions_min: int
    study_questions_max: int

    @classmethod
    def for_duration(
        cls,
        duration_sec: float | int,
        *,
        keyframe_base_min: int,
        keyframe_base_max: int,
    ) -> "LengthBudget":
        ch_min, ch_max = chapters_target(duration_sec)
        kf_min, kf_max = keyframe_window(
            duration_sec, base_min=keyframe_base_min, base_max=keyframe_base_max
        )
        sq_min, sq_max = study_questions_target(duration_sec)
        return cls(
            duration_sec=float(duration_sec or 0),
            chapters_min=ch_min,
            chapters_max=ch_max,
            mainline_max=mainline_max(duration_sec),
            global_visual_limit=global_visual_limit(duration_sec),
            glossary_max=glossary_max(duration_sec),
            review_questions_max=review_questions_max(duration_sec),
            keyframe_min=kf_min,
            keyframe_max=kf_max,
            study_questions_min=sq_min,
            study_questions_max=sq_max,
        )

    def density_hint(self) -> str:
        """Short Chinese phrase the prompts can splice in verbatim."""
        m = _minutes(self.duration_sec)
        if m < 10:
            return "短视频（<10 分钟），节奏紧凑，结构尽量精炼。"
        if m < 30:
            return "中等长度（10-30 分钟），按主线展开 4-6 章。"
        if m < 60:
            return "较长视频（30-60 分钟），需要更密的章节与知识单元，通常每 5-8 分钟应有独立章节支持。"
        return "长视频（>60 分钟），切勿过度压缩；每段 5-8 分钟应有独立章节支持。"
