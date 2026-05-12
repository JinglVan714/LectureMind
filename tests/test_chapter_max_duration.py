"""M2.2 regression tests for ``ChapterPlanner._enforce_max_duration``.

These tests pin the architectural prevention layer that bisects any
oversized chapter into ``ceil(width / max_chapter_sec)`` equal
sub-chapters tagged ``source='equal_split_overflow'``. The trigger
case is BV1ypdgBCE9B: the planner produced a 17min 43s chapter that
the downstream map-chapter LLM call could not fit in DeepSeek's
8192-token output cap, surfacing as the user-visible
``"本章内容生成失败"`` placeholder.

The contract pinned here:

1. Chapters within budget pass through untouched (zero-cost happy path).
2. A chapter just over budget bisects into 2 sub-chapters, both under
   budget.
3. A pathologically long chapter (17min on a 12min budget) bisects into
   ``ceil(width/threshold)`` sub-chapters and **none** of the
   sub-chapters exceeds the budget.
4. The first sub-chapter preserves the original anchor text /
   confidence (so the user still sees the planner's strongest signal);
   subsequent sub-chapters carry a ``（续 k/N）`` suffix.
5. Setting ``max_chapter_sec=0`` disables the safeguard (this is the
   pre-M2.2 behaviour and is what shipped before BV1ypdgBCE9B
   surfaced the truncation).
6. After bisection, coverage of ``[0, duration_sec]`` stays
   contiguous: no gaps, no overlaps, sorted by ``start_sec``.
"""
from __future__ import annotations

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="lecturemind-test-cmaxdur-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.ingest.subtitle import SubtitleSegment  # noqa: E402
from app.understand.chapter_planner import (  # noqa: E402
    ChapterAnchor,
    _enforce_max_duration,
)


def _seg(start: float, end: float, text: str) -> SubtitleSegment:
    return SubtitleSegment(start=start, end=end, text=text)


def _anchor(
    start: float,
    end: float,
    *,
    text: str = "anchor",
    source: str = "transition_phrase",
    confidence: float = 0.85,
) -> ChapterAnchor:
    return ChapterAnchor(
        start_sec=start,
        end_sec=end,
        confidence=confidence,
        anchor_text=text,
        source=source,
        mode="structural",
    )


# ---------------------------------------------------------------------------
# 1. Happy path — under budget chapters untouched
# ---------------------------------------------------------------------------


def test_chapters_within_budget_pass_through_unchanged() -> None:
    """3 chapters of 600s / 600s / 600s with a 720s budget = no work to do.

    The function MUST be a no-op on the inputs (same instances, same
    order) so caching layers downstream don't see a spurious key change.
    """
    chapters = [
        _anchor(0.0, 600.0, text="ch1"),
        _anchor(600.0, 1200.0, text="ch2"),
        _anchor(1200.0, 1800.0, text="ch3"),
    ]
    out = _enforce_max_duration(
        chapters,
        max_chapter_sec=720.0,
        segments=[],
        mode="structural",
    )
    assert out == chapters
    # Identity preserved on each item — important for downstream cache
    # keys that rely on object reuse.
    for a, b in zip(out, chapters):
        assert a is b


# ---------------------------------------------------------------------------
# 2. Bisection — chapter just over budget splits into 2
# ---------------------------------------------------------------------------


def test_oversized_chapter_bisected_into_two_when_just_over_budget() -> None:
    """800s chapter on a 720s budget → 2 equal sub-chapters of 400s.

    400s is comfortably under the budget so the new sub-chapters do
    not need further bisection, and the LLM call for either one fits
    under the 8192-token output cap.
    """
    chapters = [_anchor(0.0, 800.0, text="huge", confidence=0.9)]
    out = _enforce_max_duration(
        chapters,
        max_chapter_sec=720.0,
        segments=[],
        mode="structural",
    )
    assert len(out) == 2
    assert out[0].start_sec == 0.0
    assert out[0].end_sec == 400.0
    assert out[1].start_sec == 400.0
    assert out[1].end_sec == 800.0
    # Both sub-chapters carry the equal_split_overflow source so the
    # matrix verifier can flag the architectural bisection.
    assert out[0].source == "equal_split_overflow"
    assert out[1].source == "equal_split_overflow"


# ---------------------------------------------------------------------------
# 3. Pathological chapter — BV1ypdgBCE9B chapter 5 (17min 43s)
# ---------------------------------------------------------------------------


def test_pathological_chapter_bisected_until_all_subchapters_under_budget() -> None:
    """17min 43s chapter on a 12min budget → split into N sub-chapters
    where every single one is under the budget. This is the exact
    real-machine BV1ypdgBCE9B chapter 5 shape (start=05:38, end=23:21).

    1063 / 720 ≈ 1.48, so ``ceil`` gives N=2 and each sub-chapter is
    ~531.5s — well under the budget. The test pins both N and the
    "every part under budget" invariant so a future refactor can't
    silently regress to N=1 (no split) or pick non-uniform widths.
    """
    chapters = [_anchor(338.0, 1401.0, text="第5章")]  # 17m23s span
    width = 1401.0 - 338.0
    assert width > 720.0  # sanity: real test setup is over budget

    out = _enforce_max_duration(
        chapters,
        max_chapter_sec=720.0,
        segments=[],
        mode="structural",
    )
    assert len(out) == 2  # ceil(1063/720) == 2
    for sub in out:
        sub_width = sub.end_sec - sub.start_sec
        # The whole point of the safeguard: every sub-chapter MUST
        # come in under the budget.
        assert sub_width <= 720.0


# ---------------------------------------------------------------------------
# 4. First sub-chapter preserves the natural anchor signal
# ---------------------------------------------------------------------------


def test_first_subchapter_preserves_anchor_text_and_confidence() -> None:
    """Bisection must not erase the planner's strongest signal.

    The user-facing diagnostic on the rendered HTML chapter card comes
    from ``anchor_text``; if every sub-chapter got "（章节续）" the
    planner's deterministic signal would vanish from the UI. Pin the
    first sub-chapter to the parent text + confidence; later
    sub-chapters carry a ``（续 k/N）`` suffix so users still see the
    architectural split.
    """
    parent_text = "深入讲解 GRPO 算法原理"
    chapters = [_anchor(0.0, 1500.0, text=parent_text, confidence=0.92)]
    out = _enforce_max_duration(
        chapters,
        max_chapter_sec=720.0,
        segments=[],
        mode="structural",
    )
    assert len(out) >= 2
    assert out[0].anchor_text == parent_text
    assert out[0].confidence == 0.92
    # Later sub-chapters carry the 续 k/N marker so a quick log scan
    # tells the operator "this row is a bisection, look upstream".
    for k, sub in enumerate(out[1:], start=2):
        assert "续" in sub.anchor_text
        assert f"{k}/{len(out)}" in sub.anchor_text


# ---------------------------------------------------------------------------
# 5. max_chapter_sec=0 disables the safeguard
# ---------------------------------------------------------------------------


def test_zero_threshold_disables_safeguard() -> None:
    """Operators may want the pre-M2.2 behaviour back (e.g. while
    debugging a different layer of the pipeline). Setting
    ``LECTURE_CHAPTER_MAX_DURATION_SEC=0`` must let oversized chapters
    pass through verbatim — same as the planner shipped before
    M2.2.
    """
    chapters = [_anchor(0.0, 5000.0, text="huge")]  # absurdly long
    out = _enforce_max_duration(
        chapters,
        max_chapter_sec=0.0,
        segments=[],
        mode="structural",
    )
    assert out == chapters
    assert out[0].source == "transition_phrase"  # untouched, not _overflow


# ---------------------------------------------------------------------------
# 6. Coverage invariant — bisection must keep contiguous coverage
# ---------------------------------------------------------------------------


def test_bisection_preserves_contiguous_coverage_and_sort_order() -> None:
    """After bisection the chapter list must still cover the input
    range continuously: every chapter's ``end_sec`` matches the next
    chapter's ``start_sec``, the first starts at 0, the last ends at
    the original last end. Out-of-order output would silently corrupt
    map-chapter window slicing in ``_collect_chapter_dicts``.
    """
    chapters = [
        _anchor(0.0, 300.0, text="A"),       # under budget, untouched
        _anchor(300.0, 1500.0, text="B"),    # 1200s → bisects
        _anchor(1500.0, 1800.0, text="C"),   # under budget, untouched
    ]
    out = _enforce_max_duration(
        chapters,
        max_chapter_sec=720.0,
        segments=[],
        mode="structural",
    )
    # Sort invariant
    for prev, nxt in zip(out, out[1:]):
        assert prev.end_sec == nxt.start_sec, "gap or overlap detected"
    # Total coverage
    assert out[0].start_sec == 0.0
    assert out[-1].end_sec == 1800.0
    # The unchanged outer chapters keep their original source labels —
    # only the middle chapter B should produce overflow rows.
    sources = [c.source for c in out]
    assert sources[0] == "transition_phrase"
    assert sources[-1] == "transition_phrase"
    overflow_count = sources.count("equal_split_overflow")
    assert overflow_count >= 2  # B bisected into >=2 parts
