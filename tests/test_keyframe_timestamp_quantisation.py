"""Regression test for the keyframe fresh-vs-cached timestamp invariant.

The map-reduce chapter cache (M2 P3) hashes ``round(frame.ts, 3)`` into
its ``prompt_hash``. Until commit ``<this-commit>`` the fresh extraction
path of :class:`app.ingest.keyframe.KeyframeExtractor._extract_frames`
stored PySceneDetect's microsecond-precision ``float`` on
``Keyframe.timestamp`` (e.g. ``12.345678``) while writing the file as
``f"{int(ts * 1000):08d}.jpg"`` (e.g. ``00012345.jpg``). A subsequent
run that hit the *cached* branch reconstructed
``int(p.stem) / 1000.0 = 12.345`` from the filename, drifting the
hash bucket by 1 ms and forcing a 100 %-miss on the chapter cache for
every fresh run that wrote the cache the same day (epic
``BV1WEovBjEpd`` 二跑 was exactly this scenario).

The fix quantises ``ts`` to millisecond precision *before* both the
filename and the in-memory ``Keyframe.timestamp``, so the two paths
become byte-equal. This test pins that invariant.
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any

import pytest

from app.ingest.keyframe import Keyframe, KeyframeExtractor


def _fake_ffmpeg(monkeypatch: pytest.MonkeyPatch, out_dir: Path) -> None:
    """Stub :func:`shutil.which` / :func:`subprocess.run` so we never
    actually call ffmpeg in the test. We just touch the file that
    ``_extract_frames`` is about to declare as the frame path."""

    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")

    class _Proc:
        returncode = 0
        stderr = ""

    def _run(cmd: list[str], **_kwargs: Any) -> _Proc:
        # The frame path is always the second-to-last positional arg
        # in our ffmpeg call list (see _extract_frames).
        out_path = Path(cmd[-3])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\xff\xd8\xff\xd9")  # 4-byte JPEG stub
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _run)


def test_fresh_extract_matches_cached_reload_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh ``_extract_frames`` and cached ``_frame_from_path`` must
    return :class:`Keyframe` instances with identical ``timestamp``
    values for every file written to disk — otherwise the chapter
    cache prompt_hash diverges between 首跑 (fresh) and 二跑 (cached).
    """

    _fake_ffmpeg(monkeypatch, tmp_path)
    extractor = KeyframeExtractor()

    # Span a mix of clean / sub-ms / boundary timestamps. The two
    # values flagged "drift" are the ones that used to round
    # asymmetrically under the legacy code (``round(ts, 3)`` of the
    # raw float vs ``int(ts * 1000) / 1000.0``).
    raw_timestamps = [
        0.0,
        1.0,
        12.345,
        12.3456,      # drift candidate (legacy: round → 12.346 vs 12.345)
        12.345678,    # drift candidate
        99.999,
        100.0001,     # drift candidate (legacy: 100.000 vs 100.000 — fine)
        3600.4999,    # near rounding boundary
    ]

    fresh = extractor._extract_frames(
        Path("/dev/null/fake.mp4"), raw_timestamps, tmp_path
    )

    assert len(fresh) == len(raw_timestamps), (
        "stub ffmpeg should have produced every requested file"
    )

    # Every fresh timestamp must equal the corresponding cached-reload
    # timestamp byte-for-byte. ``Keyframe`` is a plain dataclass with a
    # plain ``float``, so equality here is value equality.
    for kf_fresh, raw in zip(fresh, raw_timestamps):
        kf_cached = KeyframeExtractor._frame_from_path(kf_fresh.path)
        assert kf_fresh.timestamp == kf_cached.timestamp, (
            f"fresh vs cached drift for raw ts={raw!r}: "
            f"fresh={kf_fresh.timestamp!r}, cached={kf_cached.timestamp!r}"
        )
        # Also pin the quantisation contract directly: the fresh path
        # must be at most 1 ms below ``raw`` (truncation toward zero)
        # and never above it.
        assert 0 <= raw - kf_fresh.timestamp < 1e-3 + 1e-9


def test_fresh_extract_matches_round3_bucket_of_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end invariant for :func:`compute_prompt_hash`: the
    ``round(ts, 3)`` bucket of a fresh frame timestamp must equal the
    bucket of the same frame reloaded from disk. This is the property
    the chapter cache hash actually depends on.
    """
    from app.understand.chapter_cache import compute_prompt_hash

    _fake_ffmpeg(monkeypatch, tmp_path)
    extractor = KeyframeExtractor()

    raw_timestamps = [0.0, 12.3456, 12.345678, 3600.4999]
    fresh = extractor._extract_frames(
        Path("/dev/null/fake.mp4"), raw_timestamps, tmp_path
    )
    cached = [KeyframeExtractor._frame_from_path(kf.path) for kf in fresh]

    common = {
        "chapter_start_sec": 0.0,
        "chapter_end_sec": 4000.0,
        "chapter_segments": [],
        "model_id": "deepseek-v4-flash",
        "prompt_version": "m2-map-v1",
    }
    # Wrap as objects with ``.timestamp`` (matches ``_frame_ts`` lookup).
    h_fresh = compute_prompt_hash(chapter_frames=fresh, **common)
    h_cached = compute_prompt_hash(chapter_frames=cached, **common)
    assert h_fresh == h_cached, (
        "compute_prompt_hash must be invariant under fresh-vs-cached "
        "keyframe reload — otherwise the chapter cache 100 %-misses on "
        "二跑 (epic BV1WEovBjEpd regression)."
    )
