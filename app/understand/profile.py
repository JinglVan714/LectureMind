"""Length-aware ``LectureProfile`` router for M2.

This module is the single source of truth for how downstream stages
(ChapterPlanner, MapReduceIRBuilder, ChapterCache, CriticContext,
patch Reviser) decide what to do for a given lecture. P1 only
introduces the data model and the deterministic ``select_profile``
function; later phases (P2-P7) will read from these fields.

Profile policy table (frozen for M2; see spec §1.2):

    | profile  | duration   | use_chapter_planner | planner_mode | use_map_reduce | use_chapter_cache | critic_mode | reviser_mode | study_question_mode |
    |----------|------------|---------------------|--------------|----------------|-------------------|-------------|--------------|---------------------|
    | tiny     |   <3 min   | True                | hint         | False          | False             | off         | off          | head_only           |
    | standard |  3-25 min  | True                | hint         | False          | False             | full        | patch        | multi_window        |
    | long     | 25-60 min  | True                | structural   | True           | True              | projected   | patch        | multi_window        |
    | epic     |  60+ min   | True                | structural   | True           | True              | projected   | patch        | multi_window        |

Thresholds default to ``(180, 1500, 3600)`` seconds and are configured
via :class:`app.config.Settings.lecture_profile_thresholds_sec`. The
boundary belongs to the **upper** bucket: ``duration_sec == tiny_th``
maps to ``standard``.

P1 deliberately keeps this module dependency-free (no LLM clients, no
SQLite) so it can be imported anywhere without side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.config import Settings


ProfileName = Literal["tiny", "standard", "long", "epic"]
CriticMode = Literal["off", "full", "projected"]
ReviserMode = Literal["off", "patch", "full"]
StudyQuestionMode = Literal["head_only", "multi_window"]
ChapterPlannerMode = Literal["hint", "structural"]


@dataclass(frozen=True)
class LectureProfile:
    """Length-aware policy bundle.

    All booleans/literals are pre-computed from the profile name; the
    only runtime-varying field is :attr:`duration_sec`, populated via
    :meth:`with_duration` when ``select_profile`` materialises the
    template for a specific video.
    """

    name: ProfileName
    duration_sec: float
    use_chapter_planner: bool
    use_map_reduce: bool
    use_chapter_cache: bool
    critic_mode: CriticMode
    reviser_mode: ReviserMode
    study_question_mode: StudyQuestionMode
    chapter_planner_mode: ChapterPlannerMode

    def with_duration(self, duration_sec: float) -> "LectureProfile":
        """Return a copy of this template bound to ``duration_sec``."""

        return replace(self, duration_sec=float(duration_sec))


# Frozen template table. ``duration_sec=0.0`` is a sentinel that callers
# overwrite via :meth:`LectureProfile.with_duration`. Treat this dict as
# read-only — mutating it would silently change every downstream stage.
_PROFILE_TABLE: dict[ProfileName, LectureProfile] = {
    "tiny": LectureProfile(
        name="tiny",
        duration_sec=0.0,
        use_chapter_planner=True,  # ChapterPlanner runs for >= 60s in P2
        use_map_reduce=False,
        use_chapter_cache=False,
        critic_mode="off",
        reviser_mode="off",
        study_question_mode="head_only",
        chapter_planner_mode="hint",
    ),
    "standard": LectureProfile(
        name="standard",
        duration_sec=0.0,
        use_chapter_planner=True,
        use_map_reduce=False,
        use_chapter_cache=False,
        critic_mode="full",
        reviser_mode="patch",
        study_question_mode="multi_window",
        chapter_planner_mode="hint",
    ),
    "long": LectureProfile(
        name="long",
        duration_sec=0.0,
        use_chapter_planner=True,
        use_map_reduce=True,
        use_chapter_cache=True,
        # long videos use the projected critic to keep prompt size
        # bounded; full critic is reserved for standard.
        critic_mode="projected",
        reviser_mode="patch",
        study_question_mode="multi_window",
        chapter_planner_mode="structural",
    ),
    "epic": LectureProfile(
        name="epic",
        duration_sec=0.0,
        use_chapter_planner=True,
        use_map_reduce=True,
        use_chapter_cache=True,
        critic_mode="projected",
        reviser_mode="patch",
        study_question_mode="multi_window",
        chapter_planner_mode="structural",
    ),
}


def select_profile(duration_sec: float, settings: "Settings") -> LectureProfile:
    """Pick a :class:`LectureProfile` for ``duration_sec``.

    Boundary policy: thresholds are exclusive on the lower side, i.e.
    ``duration_sec < tiny_th`` is tiny; ``duration_sec == tiny_th``
    upgrades to standard, and so on. This avoids a 4-5 minute video
    flapping between buckets across runs (decision D7).
    """

    tiny_th, std_th, long_th = settings.lecture_profile_thresholds_sec
    if duration_sec < tiny_th:
        name: ProfileName = "tiny"
    elif duration_sec < std_th:
        name = "standard"
    elif duration_sec < long_th:
        name = "long"
    else:
        name = "epic"
    return _PROFILE_TABLE[name].with_duration(duration_sec)


__all__ = [
    "LectureProfile",
    "ProfileName",
    "CriticMode",
    "ReviserMode",
    "StudyQuestionMode",
    "ChapterPlannerMode",
    "select_profile",
]
