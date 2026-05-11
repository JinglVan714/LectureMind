"""Unit tests for ``app.understand.chapter_planner``.

P2 scope (M2 length-aware routing):

* ``plan_chapters`` is a deterministic, LLM-free chapter-anchor extractor.
* Three signals combine into anchor candidates: transition-phrase
  regex on subtitle text, silent gaps between segments, and visual-type
  shifts in adjacent keyframes.
* Two modes drive selection: ``hint`` (tiny/standard) keeps top-K
  candidates ≤ chapters_max+2 and lets the LLM finalise; ``structural``
  (long/epic) enforces chapters_min ≤ N ≤ chapters_max with equal-split
  fallback, greedy merging, boundary snapping and a per-chapter length
  floor.

These tests target the public surface (``plan_chapters`` + the
``ChapterAnchor`` schema) so refactoring the internal helpers later is
free as long as the contracts hold.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ingest.subtitle import SubtitleSegment


def _settings(**overrides):
    """Build a hermetic Settings; overrides flow through constructor kwargs.

    P1 already enabled ``populate_by_name=True``, so we can pass field
    names directly without remembering the env-var aliases.
    """
    from app.config import Settings

    return Settings(**overrides)


def _profile(mode: str = "hint", *, name: str = "standard", duration: float = 600.0):
    """Build a LectureProfile template with overridden chapter_planner_mode.

    P2 keeps the planner mode read from the profile dataclass — we
    deliberately don't add a config override for it because the spec
    pins the mapping (tiny/standard → hint; long/epic → structural).
    For tests we materialise a profile and ``dataclasses.replace`` the
    mode/name so we can drive both modes against the same fixtures.
    """
    from dataclasses import replace

    from app.understand.profile import _PROFILE_TABLE

    base = _PROFILE_TABLE[name].with_duration(duration)
    return replace(base, chapter_planner_mode=mode)


@dataclass
class _FakeFrame:
    """Lightweight stand-in for FrameDescription with only the fields
    the planner reads. Avoids importing the full VLM module in tests.
    """

    timestamp: float
    visual_type: str = "other"


def _seg(start: float, end: float, text: str) -> SubtitleSegment:
    return SubtitleSegment(start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _flat_subtitles(duration: float, every: float = 5.0):
    """Generate dense, gap-free subtitles with neutral wording.

    Useful as a baseline that produces *no* anchor candidates so the
    structural mode is forced into equal-split territory.
    """

    out: list[SubtitleSegment] = []
    t = 0.0
    while t < duration:
        out.append(_seg(t, min(t + every, duration), "讲解一些细节内容。"))
        t += every
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanChaptersGate:
    def test_plan_chapters_skipped_when_disabled(self):
        from app.understand.chapter_planner import plan_chapters

        s = _settings(lecture_chapter_planner_enabled=False)
        anchors = plan_chapters(
            profile=_profile("structural", name="long", duration=2000.0),
            duration_sec=2000.0,
            segments=_flat_subtitles(2000.0),
            frames=[],
            chapters_min=5,
            chapters_max=10,
            settings=s,
        )
        assert anchors == []

    def test_plan_chapters_skipped_for_videos_under_min_sec(self):
        from app.understand.chapter_planner import plan_chapters

        # Default min_sec is 60s; a 45s tiny video must not produce any
        # planned chapters even with the planner globally enabled.
        s = _settings()
        anchors = plan_chapters(
            profile=_profile("hint", name="tiny", duration=45.0),
            duration_sec=45.0,
            segments=_flat_subtitles(45.0),
            frames=[],
            chapters_min=2,
            chapters_max=5,
            settings=s,
        )
        assert anchors == []


class TestPlanChaptersHintMode:
    def test_plan_chapters_hint_mode_returns_within_max(self):
        """Hint mode tolerates over-extraction by chapters_max + 2.

        We feed 12 strong transition-phrase signals into a 1500s video
        with chapters_max=5; the planner must keep at most 5+2=7
        chapters in hint mode (mode-driven cap, not structural).
        """
        from app.understand.chapter_planner import plan_chapters

        # Build subtitles with 12 explicit transition phrases evenly
        # spread across 1500s; everything else is filler.
        segments: list[SubtitleSegment] = []
        for i in range(12):
            t = 100.0 + i * 110.0  # 100, 210, 320, ... < 1500
            segments.append(_seg(t, t + 4.0, f"接下来我们看第{i+1}部分。"))
            segments.append(_seg(t + 4.0, t + 100.0, "继续讲解。"))

        anchors = plan_chapters(
            profile=_profile("hint", name="standard", duration=1500.0),
            duration_sec=1500.0,
            segments=segments,
            frames=[],
            chapters_min=3,
            chapters_max=5,
            settings=_settings(),
        )
        assert 0 < len(anchors) <= 5 + 2, (
            f"hint mode must keep ≤ chapters_max+2 anchors, got {len(anchors)}"
        )
        # Every anchor must be tagged with mode=hint for downstream
        # diagnostics.
        assert all(a.mode == "hint" for a in anchors)


class TestPlanChaptersStructuralMode:
    def test_plan_chapters_structural_mode_satisfies_min(self):
        """Structural mode must reach chapters_min even when no
        deterministic signals exist (equal-split fills the deficit).
        """
        from app.understand.chapter_planner import plan_chapters

        anchors = plan_chapters(
            profile=_profile("structural", name="long", duration=2000.0),
            duration_sec=2000.0,
            segments=_flat_subtitles(2000.0),
            frames=[],
            chapters_min=6,
            chapters_max=10,
            settings=_settings(),
        )
        assert len(anchors) >= 6
        assert all(a.mode == "structural" for a in anchors)
        # Equal-split fallback should be the dominant source for a
        # signal-free input.
        assert any(a.source == "equal_split" for a in anchors)

    def test_plan_chapters_structural_mode_satisfies_max(self):
        """Structural mode must trim to chapters_max even if many
        strong signals exist.
        """
        from app.understand.chapter_planner import plan_chapters

        # Generate 20 strong transition phrases
        segments: list[SubtitleSegment] = []
        for i in range(20):
            t = 50.0 + i * 95.0  # 50, 145, 240, ... < 2000
            segments.append(_seg(t, t + 4.0, f"接下来我们看第{i+1}部分。"))
            segments.append(_seg(t + 4.0, t + 90.0, "讲解。"))

        anchors = plan_chapters(
            profile=_profile("structural", name="long", duration=2000.0),
            duration_sec=2000.0,
            segments=segments,
            frames=[],
            chapters_min=4,
            chapters_max=8,
            settings=_settings(),
        )
        assert 4 <= len(anchors) <= 8
        # Coverage: chapters together must span (≈) the full duration.
        # We enforce strict end-to-end coverage to catch off-by-one.
        assert anchors[0].start_sec == 0.0
        assert abs(anchors[-1].end_sec - 2000.0) < 0.5
        # No gaps between adjacent chapters.
        for prev, nxt in zip(anchors, anchors[1:]):
            assert abs(prev.end_sec - nxt.start_sec) < 0.5


class TestPlanChaptersSignalPriority:
    def test_plan_chapters_uses_strong_anchors_first(self):
        """When a transition phrase, a silence gap and a visual shift
        are all available, the high-confidence transition phrase wins
        the slot in hint mode (where K is small).
        """
        from app.understand.chapter_planner import plan_chapters

        # Transition phrase at t≈300 (high conf).
        # Silence gap between t=600 and t=607 (mid conf).
        # Visual shift at t≈900 (low conf).
        segments = [
            _seg(0.0, 300.0, "开场白。"),
            _seg(300.0, 305.0, "接下来我们进入主体内容。"),
            _seg(305.0, 600.0, "讲解一段。"),
            # 7 second silence gap (>= 6s default)
            _seg(607.0, 900.0, "继续讲解。"),
            _seg(900.0, 905.0, "再看一个例子。"),
            _seg(905.0, 1200.0, "结束。"),
        ]
        frames = [
            _FakeFrame(timestamp=100.0, visual_type="slide_text"),
            _FakeFrame(timestamp=500.0, visual_type="slide_text"),
            # Visual shift right at t=900 (within ±2s of segment boundary)
            _FakeFrame(timestamp=899.0, visual_type="slide_text"),
            _FakeFrame(timestamp=901.0, visual_type="code"),
        ]

        # Hint mode with chapters_max=2 → top-K = 4. Transition phrase
        # is the strongest signal so its anchor must appear.
        anchors = plan_chapters(
            profile=_profile("hint", name="standard", duration=1200.0),
            duration_sec=1200.0,
            segments=segments,
            frames=frames,
            chapters_min=2,
            chapters_max=2,
            settings=_settings(),
        )
        # The anchor near t=300 must be present (transition phrase).
        assert any(
            a.source == "transition_phrase" and 295.0 <= a.start_sec <= 310.0
            for a in anchors
        ), f"transition phrase anchor missing; got {[(a.source, a.start_sec) for a in anchors]}"


class TestPlanChaptersEqualSplitAndSnapping:
    def test_plan_chapters_equal_split_when_no_anchors(self):
        """Structural mode with zero candidates should produce exactly
        chapters_min equal-split chapters covering the full duration.
        """
        from app.understand.chapter_planner import plan_chapters

        anchors = plan_chapters(
            profile=_profile("structural", name="long", duration=1800.0),
            duration_sec=1800.0,
            segments=_flat_subtitles(1800.0),
            frames=[],
            chapters_min=4,
            chapters_max=8,
            settings=_settings(),
        )
        assert len(anchors) == 4
        assert all(a.source == "equal_split" for a in anchors)
        # Equal-split → roughly uniform chapter widths (within ±10%).
        widths = [a.end_sec - a.start_sec for a in anchors]
        avg = sum(widths) / len(widths)
        assert all(abs(w - avg) <= 0.15 * avg for w in widths)

    def test_plan_chapters_snaps_to_segment_boundary(self):
        """Anchors should snap their start_sec to the nearest segment
        boundary so chapter starts align with subtitle starts.
        """
        from app.understand.chapter_planner import plan_chapters

        # The transition phrase begins at exactly t=300.0 in the
        # subtitles below; the planner must therefore put the anchor
        # at start_sec=300.0 (not 299.x, not 300.x).
        segments = [
            _seg(0.0, 300.0, "前面的内容。"),
            _seg(300.0, 305.0, "接下来我们看下一部分的内容。"),
            _seg(305.0, 800.0, "继续讲解。"),
        ]
        anchors = plan_chapters(
            profile=_profile("structural", name="long", duration=800.0),
            duration_sec=800.0,
            segments=segments,
            frames=[],
            chapters_min=2,
            chapters_max=4,
            settings=_settings(),
        )
        # Find the anchor that starts at the transition phrase.
        transition_anchors = [
            a for a in anchors if a.source == "transition_phrase"
        ]
        assert transition_anchors, (
            "expected at least one transition_phrase anchor; "
            f"got sources={[a.source for a in anchors]}"
        )
        # start_sec must equal a real segment.start within 0.05s.
        seg_starts = {round(s.start, 3) for s in segments}
        for a in transition_anchors:
            assert any(
                abs(a.start_sec - s) < 0.05 for s in seg_starts
            ), f"anchor at {a.start_sec} did not snap to a segment boundary"
