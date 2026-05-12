"""M2.2 regression tests for ``MapFailure`` telemetry contract.

These tests pin the schema and serialisation of the structured failure
records that travel from ``MapReduceIRBuilder`` → ``LectureIRBuilder``
→ ``pipeline_stats['map_reduce']['map_failures']``. Pre-M2.2 the
field was ``tuple[int, ...]`` (chapter index only) and the user had
no way to triage failures without trawling logs.

Three behaviours pinned:

1. ``MapFailure`` carries the full diagnostic payload (window bounds,
   subtitle / frame counts, finish_reason, error_class, error_excerpt,
   attempts).
2. ``MapReduceStats.failure_indices`` returns the legacy ``tuple[int]``
   shape for downstream callers that haven't migrated yet.
3. ``ir_builder._serialise_map_failure`` projects the dataclass into a
   JSON-friendly dict with all expected keys, AND tolerates pre-M2.2
   ``int``-typed entries (legacy mocks in older tests) by degrading
   to a single ``{"chapter_index": <int>}`` shape.
"""
from __future__ import annotations

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="lecturemind-test-mapfail-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.understand.ir_builder import _serialise_map_failure  # noqa: E402
from app.understand.ir_map_reduce import (  # noqa: E402
    MapFailure,
    MapReduceStats,
)


# ---------------------------------------------------------------------------
# 1. MapFailure schema — full diagnostic context survives the dataclass
# ---------------------------------------------------------------------------


def test_map_failure_carries_full_diagnostic_context() -> None:
    """The new dataclass must expose every field the verifier / matrix
    tools were asking for in BV1ypdgBCE9B post-mortem: window bounds,
    duration, subtitle/frame counts, attempts, finish_reason,
    error_class, error_excerpt.

    A regression that drops or renames any field would break the
    ``"本章抽取失败：…"`` placeholder summary string, so we pin the
    exact attribute names here.
    """
    f = MapFailure(
        chapter_index=5,
        start_sec=338.0,
        end_sec=1401.0,
        duration_sec=1063.0,
        subtitle_count=412,
        frame_count=18,
        attempts=1,
        finish_reason="length",
        error_class="RuntimeError",
        error_excerpt="Map-chapter LLM truncated JSON (...)",
    )
    assert f.chapter_index == 5
    assert f.start_sec == 338.0
    assert f.end_sec == 1401.0
    assert f.duration_sec == 1063.0
    assert f.subtitle_count == 412
    assert f.frame_count == 18
    assert f.attempts == 1
    assert f.finish_reason == "length"
    assert f.error_class == "RuntimeError"
    assert "truncated" in f.error_excerpt


def test_map_failure_defaults_are_safe_for_partial_construction() -> None:
    """Construction with only ``chapter_index`` must succeed so old test
    fixtures that pre-date the M2.2 schema can still build minimal
    failure records during migration.
    """
    f = MapFailure(chapter_index=3)
    assert f.chapter_index == 3
    assert f.duration_sec == 0.0
    assert f.finish_reason == ""
    assert f.error_class == ""


# ---------------------------------------------------------------------------
# 2. failure_indices — backwards-compat shape
# ---------------------------------------------------------------------------


def test_failure_indices_returns_legacy_int_tuple() -> None:
    """Pre-M2.2 callers of ``stats.map_failures`` saw ``tuple[int, ...]``.
    The :prop:`failure_indices` property returns that exact legacy
    shape so log-scrapers / dashboards can be migrated incrementally.
    """
    stats = MapReduceStats(
        map_calls=4,
        map_failures=(
            MapFailure(chapter_index=2, finish_reason="length"),
            MapFailure(chapter_index=5, error_class="TimeoutError"),
        ),
    )
    assert stats.failure_indices == (2, 5)
    # Shape: tuple of ints, NOT MapFailure objects.
    assert all(isinstance(i, int) for i in stats.failure_indices)


def test_failure_indices_empty_when_no_failures() -> None:
    """Happy path — no failures means an empty tuple, not None or
    other falsy weirdness."""
    stats = MapReduceStats()
    assert stats.failure_indices == ()
    assert stats.map_failures == ()


# ---------------------------------------------------------------------------
# 3. _serialise_map_failure — JSON-friendly projection for telemetry
# ---------------------------------------------------------------------------


def test_serialise_map_failure_emits_all_expected_keys() -> None:
    """Every field needed for downstream triage must appear in the dict
    that ``ir_builder.build_with_agents`` injects into
    ``stats['map_reduce']['map_failures']``. The matrix verifier
    dispatches on ``finish_reason`` / ``duration_sec`` so missing
    either one would silently break debuggability.
    """
    f = MapFailure(
        chapter_index=5,
        start_sec=338.0,
        end_sec=1401.0,
        duration_sec=1063.0,
        subtitle_count=412,
        frame_count=18,
        attempts=1,
        finish_reason="length",
        error_class="RuntimeError",
        error_excerpt="trunc(…)",
    )
    d = _serialise_map_failure(f)
    expected_keys = {
        "chapter_index",
        "start_sec",
        "end_sec",
        "duration_sec",
        "subtitle_count",
        "frame_count",
        "attempts",
        "finish_reason",
        "error_class",
        "error_excerpt",
    }
    assert set(d.keys()) == expected_keys
    # Spot-check the values round-trip with sensible types so the dict
    # is JSON-serialisable on the pipeline_stats path.
    assert d["chapter_index"] == 5
    assert d["finish_reason"] == "length"
    assert d["error_class"] == "RuntimeError"
    # Floats are rounded to 1 decimal so noisy timing precision doesn't
    # bloat the matrix report.
    assert d["duration_sec"] == 1063.0


def test_serialise_map_failure_tolerates_legacy_int_entry() -> None:
    """Older mocks / tests may still emit ``int`` chapter indices into
    ``stats['map_failures']``. The serialiser must degrade gracefully
    instead of raising ``AttributeError`` and tanking the whole
    pipeline_stats dict.
    """
    d = _serialise_map_failure(7)
    assert d == {"chapter_index": 7}


def test_serialise_map_failure_handles_missing_optional_fields() -> None:
    """A bare ``MapFailure(chapter_index=2)`` should still serialise to
    the full key set with sensible defaults — empty strings / 0
    rather than ``None`` — so downstream JSON consumers don't have to
    branch on missing keys.
    """
    f = MapFailure(chapter_index=2)
    d = _serialise_map_failure(f)
    assert d["chapter_index"] == 2
    assert d["finish_reason"] == ""
    assert d["error_class"] == ""
    assert d["duration_sec"] == 0.0
    assert d["subtitle_count"] == 0
