"""Unit tests for ``app.understand.profile``.

P1 scope (M2 length-aware routing):
    * select_profile maps duration_sec to one of {tiny, standard, long, epic}
      using ``Settings.lecture_profile_thresholds_sec``.
    * LectureProfile carries the policy switches that downstream phases
      (ChapterPlanner / MapReduce / Critic / Reviser) will read.
    * Backward-compat: ``LECTURE_REVISER_ENABLED`` should still be able
      to force the legacy "reviser off" behaviour, but that compat
      lives in Settings and is exercised in ``test_smoke.py``. Here we
      only validate the deterministic profile selection.
"""
from __future__ import annotations


def _make_settings(thresholds=(180, 1500, 3600)):
    """Build a Settings instance with explicit thresholds.

    We monkey-patch the field rather than depend on the user's local
    ``.env`` so tests are hermetic.
    """
    from app.config import Settings

    # Pydantic BaseSettings honours constructor kwargs over env, so this
    # is the cleanest way to inject test thresholds without touching
    # process env (which would leak into other tests).
    return Settings(lecture_profile_thresholds_sec=thresholds)


class TestSelectProfile:
    def test_select_profile_tiny_below_180(self):
        from app.understand.profile import select_profile

        s = _make_settings()
        p = select_profile(120.0, s)
        assert p.name == "tiny"
        assert p.duration_sec == 120.0
        # tiny: planner on (>=60s), no map-reduce, no cache, agents off
        assert p.use_chapter_planner is True
        assert p.use_map_reduce is False
        assert p.use_chapter_cache is False
        assert p.critic_mode == "off"
        assert p.reviser_mode == "off"
        assert p.chapter_planner_mode == "hint"

    def test_select_profile_standard_in_range(self):
        from app.understand.profile import select_profile

        s = _make_settings()
        p = select_profile(900.0, s)  # 15min
        assert p.name == "standard"
        # standard: planner hint, single-call IR, full Critic, patch Reviser
        assert p.use_chapter_planner is True
        assert p.chapter_planner_mode == "hint"
        assert p.use_map_reduce is False
        assert p.use_chapter_cache is False
        assert p.critic_mode == "full"
        assert p.reviser_mode == "patch"
        assert p.study_question_mode == "multi_window"

    def test_select_profile_long_in_range(self):
        from app.understand.profile import select_profile

        s = _make_settings()
        p = select_profile(2640.0, s)  # 44min, BV1yX4aznE9s analogue
        assert p.name == "long"
        assert p.use_chapter_planner is True
        assert p.chapter_planner_mode == "structural"
        assert p.use_map_reduce is True
        assert p.use_chapter_cache is True
        # long uses projected critic to keep prompt size bounded
        assert p.critic_mode == "projected"
        assert p.reviser_mode == "patch"

    def test_select_profile_epic_above_3600(self):
        from app.understand.profile import select_profile

        s = _make_settings()
        p = select_profile(5400.0, s)  # 90min
        assert p.name == "epic"
        assert p.use_map_reduce is True
        assert p.use_chapter_cache is True
        assert p.chapter_planner_mode == "structural"
        assert p.critic_mode == "projected"
        assert p.reviser_mode == "patch"

    def test_select_profile_uses_settings_thresholds(self):
        """Custom thresholds in Settings must override defaults.

        With thresholds (120, 900, 2400):
            * 119s → tiny
            * 120s → standard (boundary belongs to the upper bucket)
            * 899s → standard
            * 900s → long
            * 2400s → epic
        """
        from app.understand.profile import select_profile

        s = _make_settings(thresholds=(120, 900, 2400))
        assert select_profile(119.0, s).name == "tiny"
        assert select_profile(120.0, s).name == "standard"
        assert select_profile(899.0, s).name == "standard"
        assert select_profile(900.0, s).name == "long"
        assert select_profile(2399.0, s).name == "long"
        assert select_profile(2400.0, s).name == "epic"
