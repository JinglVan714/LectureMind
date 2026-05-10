"""Unit tests for :mod:`app.understand.vlm_cache`.

Covers the five contract points from M3 Phase 1:

1. round-trip insert + lookup preserves payload bytewise
2. ``tier`` is part of the primary key — HIGH and LOW for the same
   sha + model coexist
3. ``batch_lookup`` reports each input key with hit / miss without
   silently dropping any
4. concurrent writes under WAL are durable and complete
5. ``enabled=False`` makes the cache a no-op and never touches the
   filesystem

Plus one extra test for :meth:`VLMCache.import_legacy_bv_json`
(covered in Phase 1 §1.4 of the impl plan).
"""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.understand.vlm_cache import VLMCache, sha256_file


@pytest.fixture()
def cache(tmp_path: Path) -> VLMCache:
    return VLMCache(tmp_path / "vlm_cache.sqlite")


def _payload(idx: int = 0) -> dict:
    return {
        "caption": f"frame {idx} caption",
        "ocr_text": f"some ocr {idx}",
        "visual_type": "slide",
        "importance_score": 0.42,
        "ocr_density": 0.5,
        "novelty_score": 0.1,
        "why_useful": "demonstrates feature X",
    }


def test_insert_then_lookup_round_trip(cache: VLMCache) -> None:
    payload = _payload(1)
    cache.batch_insert([("sha-A", "model-x", "high", payload)])

    got = cache.batch_lookup([("sha-A", "model-x", "high")])

    assert got == {("sha-A", "model-x", "high"): payload}


def test_tier_isolation(cache: VLMCache) -> None:
    """HIGH and LOW for the same (sha, model) coexist independently."""
    high_payload = _payload(1)
    low_payload = {"caption": "", "ocr_text": "ocr only", "visual_type": "other"}
    cache.batch_insert(
        [
            ("sha-A", "model-x", "high", high_payload),
            ("sha-A", "model-x", "low", low_payload),
        ]
    )

    got = cache.batch_lookup(
        [
            ("sha-A", "model-x", "high"),
            ("sha-A", "model-x", "low"),
        ]
    )

    assert got[("sha-A", "model-x", "high")] == high_payload
    assert got[("sha-A", "model-x", "low")] == low_payload
    # Different model → miss even with same sha + tier.
    assert cache.batch_lookup([("sha-A", "model-y", "high")]) == {
        ("sha-A", "model-y", "high"): None
    }


def test_batch_lookup_partial_hit(cache: VLMCache) -> None:
    cache.batch_insert(
        [
            ("sha-1", "m", "high", _payload(1)),
            ("sha-2", "m", "low", _payload(2)),
        ]
    )

    keys = [
        ("sha-1", "m", "high"),  # hit
        ("sha-1", "m", "low"),  # miss (different tier)
        ("sha-2", "m", "low"),  # hit
        ("sha-3", "m", "high"),  # miss
        ("sha-2", "m", "high"),  # miss (different tier)
        ("sha-1", "other", "high"),  # miss (different model)
    ]
    got = cache.batch_lookup(keys)

    assert set(got) == set(keys)
    assert got[("sha-1", "m", "high")] == _payload(1)
    assert got[("sha-2", "m", "low")] == _payload(2)
    for k in [
        ("sha-1", "m", "low"),
        ("sha-3", "m", "high"),
        ("sha-2", "m", "high"),
        ("sha-1", "other", "high"),
    ]:
        assert got[k] is None


def test_concurrent_writes_under_wal(tmp_path: Path) -> None:
    cache = VLMCache(tmp_path / "vlm_cache.sqlite")
    rows = [
        (f"sha-{i:03d}", "model-x", "high" if i % 2 == 0 else "low", _payload(i))
        for i in range(50)
    ]

    def _insert(row: tuple[str, str, str, dict]) -> None:
        cache.batch_insert([row])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_insert, rows))

    keys = [(s, m, t) for s, m, t, _ in rows]
    got = cache.batch_lookup(keys)
    for sha, model, tier, payload in rows:
        assert got[(sha, model, tier)] == payload, f"missing {sha}/{tier}"


def test_disabled_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "should_not_exist.sqlite"
    cache = VLMCache(db_path, enabled=False)

    assert cache.enabled is False
    cache.batch_insert([("sha-A", "model-x", "high", _payload(1))])
    got = cache.batch_lookup([("sha-A", "model-x", "high")])

    assert got == {("sha-A", "model-x", "high"): None}
    assert not db_path.exists(), "disabled cache must not touch the filesystem"
    # stats are still queryable (and report zeros).
    assert cache.stats() == {"row_count": 0, "db_size_bytes": 0}


def test_invalid_tier_is_dropped(cache: VLMCache) -> None:
    cache.batch_insert([("sha-A", "model-x", "medium", _payload(1))])
    assert cache.stats()["row_count"] == 0


def test_import_legacy_bv_json(tmp_path: Path) -> None:
    """Legacy per-BV JSON should land in SQLite as HIGH tier rows."""
    bv_dir = tmp_path / "frames" / "BVTEST"
    bv_dir.mkdir(parents=True)
    frame_paths: list[Path] = []
    for i in range(3):
        p = bv_dir / f"frame_{i}.jpg"
        # Distinct content per frame so SHAs differ.
        p.write_bytes(b"x" * (10 + i) + str(i).encode())
        frame_paths.append(p)

    legacy = [
        {
            "timestamp": 1.5 * i,
            "path": str(frame_paths[i]),
            "caption": f"cap {i}",
            "ocr_text": f"ocr {i}",
            "visual_type": "slide",
            "importance_score": 0.3,
            "ocr_density": 0.4,
            "novelty_score": 0.2,
            "why_useful": "test",
        }
        for i in range(3)
    ]
    legacy_path = tmp_path / "BVTEST.json"
    legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    cache = VLMCache(tmp_path / "vlm_cache.sqlite")
    imported = cache.import_legacy_bv_json(legacy_path, model="legacy-model")

    assert imported == 3
    assert cache.stats()["row_count"] == 3

    # Each frame should be retrievable as HIGH tier with its caption preserved.
    for i, p in enumerate(frame_paths):
        sha = sha256_file(p)
        got = cache.batch_lookup([(sha, "legacy-model", "high")])
        payload = got[(sha, "legacy-model", "high")]
        assert payload is not None
        assert payload["caption"] == f"cap {i}"
        assert payload["ocr_text"] == f"ocr {i}"


def test_import_legacy_bv_json_skips_missing_files(tmp_path: Path) -> None:
    legacy_path = tmp_path / "ghost.json"
    legacy_path.write_text(
        json.dumps([{"path": str(tmp_path / "missing.jpg"), "caption": "x"}]),
        encoding="utf-8",
    )

    cache = VLMCache(tmp_path / "vlm_cache.sqlite")
    assert cache.import_legacy_bv_json(legacy_path, model="m") == 0
    assert cache.stats()["row_count"] == 0


def test_overwrite_replaces_existing_payload(cache: VLMCache) -> None:
    cache.batch_insert([("sha-A", "m", "high", {"caption": "v1"})])
    cache.batch_insert([("sha-A", "m", "high", {"caption": "v2"})])

    got = cache.batch_lookup([("sha-A", "m", "high")])
    assert got[("sha-A", "m", "high")] == {"caption": "v2"}
    assert cache.stats()["row_count"] == 1


def test_stats_reports_db_size(tmp_path: Path) -> None:
    cache = VLMCache(tmp_path / "v.sqlite")
    cache.batch_insert([("sha-A", "m", "high", _payload(1))])
    stats = cache.stats()
    assert stats["row_count"] == 1
    assert stats["db_size_bytes"] > 0


def test_init_creates_table_with_pragmas(tmp_path: Path) -> None:
    db_path = tmp_path / "v.sqlite"
    VLMCache(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        # Schema is in place.
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vlm_cache'"
        ).fetchall()
        assert rows == [("vlm_cache",)]
        # WAL persists across connections (it is a per-DB property).
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        assert mode == "wal"
