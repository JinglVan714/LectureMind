"""Unit tests for :mod:`app.understand.chapter_cache` (M2 P3).

The 8 cases cover the contract from impl plan §3.2:

1. ``get`` on an unknown key returns ``None`` (and counts as a miss).
2. ``put`` followed by ``get`` returns the original chapter payload.
3. Different segments / frames / boundaries → different ``prompt_hash``.
4. Whitespace-only differences in segment text are *normalised away*
   (same trimmed text → same hash; protects against silently re-rendering
   identical chapters with leading/trailing whitespace shifts).
5. Bumping ``prompt_version`` produces a new hash so old rows can no
   longer be hit (cheap "invalidate the world" knob for prompt edits).
6. Concurrent ``put`` calls under WAL never raise IntegrityError, even
   when two writers race the same primary key.
7. ``stats()`` reports per-instance ``hits`` / ``misses`` / ``writes``
   counters plus ``rows_total`` from the underlying table.
8. ``clear_by_bv_id`` only deletes rows for the requested BV id.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.understand.chapter_cache import ChapterCache, compute_prompt_hash


# ---------- helpers ---------------------------------------------------


@dataclass
class _Seg:
    start: float
    text: str


@dataclass
class _Frm:
    ts: float
    ocr: str
    caption: str


def _segs(*pairs: tuple[float, str]) -> list[_Seg]:
    return [_Seg(start=t, text=s) for t, s in pairs]


def _frms(*tuples: tuple[float, str, str]) -> list[_Frm]:
    return [_Frm(ts=t, ocr=o, caption=c) for t, o, c in tuples]


def _payload(idx: int = 0) -> dict:
    return {
        "title": f"Chapter {idx}",
        "start_sec": 0.0 + idx,
        "end_sec": 60.0 + idx,
        "summary": f"summary {idx}",
        "points": [{"text": f"point {idx}", "ts": idx, "quote": "q"}],
    }


def _put(
    cache: ChapterCache,
    *,
    prompt_hash: str,
    bv_id: str = "BVTEST",
    chapter_index: int = 0,
    chapter_start: float = 0.0,
    chapter_end: float = 60.0,
    model_id: str = "deepseek-v4-flash",
    prompt_version: str = "m2-map-v1",
    chapter_payload: dict | None = None,
) -> None:
    cache.put(
        prompt_hash=prompt_hash,
        chapter_payload=chapter_payload or _payload(chapter_index),
        model_id=model_id,
        prompt_version=prompt_version,
        bv_id=bv_id,
        chapter_index=chapter_index,
        chapter_start=chapter_start,
        chapter_end=chapter_end,
    )


@pytest.fixture()
def cache(tmp_path: Path) -> ChapterCache:
    return ChapterCache(tmp_path / "chapter_cache.sqlite")


# ---------- tests -----------------------------------------------------


def test_chapter_cache_get_miss_returns_none(cache: ChapterCache) -> None:
    assert cache.get("nonexistent-hash") is None
    s = cache.stats()
    assert s["misses"] == 1
    assert s["hits"] == 0
    assert s["writes"] == 0
    assert s["rows_total"] == 0


def test_chapter_cache_put_then_get_returns_cached_ir(cache: ChapterCache) -> None:
    payload = _payload(2)
    _put(cache, prompt_hash="hash-A", chapter_payload=payload, chapter_index=2)

    got = cache.get("hash-A")

    assert got == payload
    s = cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 0
    assert s["writes"] == 1
    assert s["rows_total"] == 1


def test_chapter_cache_different_segments_produces_different_hash() -> None:
    base_kwargs = dict(
        chapter_start_sec=0.0,
        chapter_end_sec=120.0,
        model_id="deepseek-v4-flash",
        prompt_version="m2-map-v1",
        chapter_frames=[],
    )
    h1 = compute_prompt_hash(
        chapter_segments=_segs((0.0, "hello"), (5.0, "world")), **base_kwargs
    )
    h2 = compute_prompt_hash(
        chapter_segments=_segs((0.0, "hello"), (5.0, "WORLD")), **base_kwargs
    )
    h3 = compute_prompt_hash(
        chapter_segments=_segs((0.0, "hello")), **base_kwargs
    )
    h4 = compute_prompt_hash(
        chapter_segments=_segs((0.0, "hello"), (5.0, "world")),
        chapter_frames=_frms((1.0, "ocr", "cap")),
        chapter_start_sec=0.0,
        chapter_end_sec=120.0,
        model_id="deepseek-v4-flash",
        prompt_version="m2-map-v1",
    )
    h5 = compute_prompt_hash(
        chapter_segments=_segs((0.0, "hello"), (5.0, "world")),
        chapter_frames=[],
        chapter_start_sec=0.001,  # 1ms shift → different hash
        chapter_end_sec=120.0,
        model_id="deepseek-v4-flash",
        prompt_version="m2-map-v1",
    )

    assert len({h1, h2, h3, h4, h5}) == 5


def test_chapter_cache_segment_text_normalization_affects_hash() -> None:
    """Whitespace-only diffs around segment text must not change the hash.

    Without normalisation, a downstream subtitle re-format that adds a
    trailing space would silently miss the cache. The hash strips
    leading / trailing whitespace from each segment text and OCR /
    caption fields so re-runs hit deterministically.
    """
    base = dict(
        chapter_start_sec=0.0,
        chapter_end_sec=60.0,
        model_id="m",
        prompt_version="v1",
    )
    h_clean = compute_prompt_hash(
        chapter_segments=_segs((0.0, "hello"), (1.0, "world")),
        chapter_frames=_frms((0.5, "code", "slide")),
        **base,
    )
    h_padded = compute_prompt_hash(
        chapter_segments=_segs((0.0, "  hello  "), (1.0, "\tworld\n")),
        chapter_frames=_frms((0.5, "  code\n", " slide ")),
        **base,
    )
    assert h_clean == h_padded


def test_chapter_cache_prompt_version_bump_invalidates_existing(
    cache: ChapterCache,
) -> None:
    args_v1 = dict(
        chapter_start_sec=0.0,
        chapter_end_sec=120.0,
        chapter_segments=_segs((0.0, "alpha"), (10.0, "beta")),
        chapter_frames=_frms((5.0, "ocr", "cap")),
        model_id="deepseek-v4-flash",
        prompt_version="m2-map-v1",
    )
    h1 = compute_prompt_hash(**args_v1)
    _put(cache, prompt_hash=h1, prompt_version="m2-map-v1")
    assert cache.get(h1) is not None

    args_v2 = {**args_v1, "prompt_version": "m2-map-v2"}
    h2 = compute_prompt_hash(**args_v2)

    assert h1 != h2
    assert cache.get(h2) is None  # old row no longer reachable


def test_chapter_cache_concurrent_writes_safe(tmp_path: Path) -> None:
    """Threaded writers under WAL must not raise IntegrityError.

    Each thread upserts the same primary key — INSERT OR REPLACE makes
    that safe but the regression we're guarding against is a former
    implementation that used a plain INSERT and would die on the second
    writer.
    """
    cache = ChapterCache(tmp_path / "cc.sqlite")

    def _writer(i: int) -> None:
        # Half the threads share a single hash to force PK collisions,
        # the other half use unique hashes to populate the table.
        ph = "shared-hash" if i % 2 == 0 else f"unique-{i}"
        _put(cache, prompt_hash=ph, chapter_index=i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_writer, range(32)))

    rows = cache.stats()["rows_total"]
    # 1 shared + 16 uniques = 17 rows.
    assert rows == 17
    # The shared row must still be readable after the write storm.
    assert cache.get("shared-hash") is not None


def test_chapter_cache_stats_reports_hits_misses_writes(cache: ChapterCache) -> None:
    _put(cache, prompt_hash="h1")
    _put(cache, prompt_hash="h2")

    cache.get("h1")
    cache.get("h1")
    cache.get("missing")

    s = cache.stats()
    assert s["writes"] == 2
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["rows_total"] == 2


def test_chapter_cache_clear_by_bv_id(cache: ChapterCache) -> None:
    _put(cache, prompt_hash="h-a1", bv_id="BV_AAA", chapter_index=0)
    _put(cache, prompt_hash="h-a2", bv_id="BV_AAA", chapter_index=1)
    _put(cache, prompt_hash="h-b1", bv_id="BV_BBB", chapter_index=0)

    deleted = cache.clear_by_bv_id("BV_AAA")
    assert deleted == 2

    assert cache.get("h-a1") is None
    assert cache.get("h-a2") is None
    assert cache.get("h-b1") is not None
    assert cache.stats()["rows_total"] == 1


def test_chapter_cache_disabled_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "should_not_exist.sqlite"
    cache = ChapterCache(db_path, enabled=False)

    assert cache.enabled is False
    _put(cache, prompt_hash="h-x")
    assert cache.get("h-x") is None
    assert not db_path.exists()


def test_chapter_cache_init_creates_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "cc.sqlite"
    ChapterCache(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chapter_cache'"
        ).fetchall()
        assert rows == [("chapter_cache",)]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        assert mode == "wal"
