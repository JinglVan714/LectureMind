"""Per-chapter SQLite cache for the map-reduce IR builder (M2 P3).

The map stage of :mod:`app.understand.ir_map_reduce` (P4) issues one
LLM call per chapter. Long / epic videos can have a dozen or more
chapters; re-running the same BV — or two BVs that happen to share a
chapter prompt verbatim — would otherwise re-pay the full latency
budget every time. ``ChapterCache`` makes that latency cost amortised
to "first call only" without leaking concerns into the higher-level
builder:

* Key is a single SHA-256 (``prompt_hash``) computed by
  :func:`compute_prompt_hash` over the chapter's normalised segments,
  frames, time window, model id and prompt version. Identical inputs
  → identical hash → cache hit; any structural change to the prompt
  template should be paired with a bump of
  ``LECTURE_MAP_PROMPT_VERSION`` to invalidate the world cheaply.
* SQLite with WAL is the same backing store as
  :mod:`app.understand.vlm_cache`, so deployment / file management is
  uniform; rows live in ``data/chapter_cache.sqlite`` by default.
* When ``enabled=False`` everything becomes a no-op — ``get`` returns
  ``None``, ``put`` silently drops the row, and the DB file is never
  created. Useful for A/B and ``LECTURE_CHAPTER_CACHE_ENABLED=false``
  emergency disable.

This module deliberately ships **only** the cache primitives
(``get`` / ``put`` / ``stats`` / ``clear_by_bv_id`` and the hash
function). Wiring into the actual map-reduce pipeline lives in P4 and
P7 — keeping the cache phase self-contained means we can validate the
WAL / hash invariants in isolation before any LLM code depends on it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---- prompt_hash ---------------------------------------------------


def _seg_text(seg: Any) -> str:
    """Best-effort attribute access for a subtitle segment.

    The hash function intentionally accepts duck-typed objects: we may
    feed it real :class:`app.ingest.subtitle.SubtitleSegment` instances,
    test fakes (see ``tests/test_chapter_cache.py``) or thin dataclass
    adapters from the map-reduce builder. We strip whitespace so the
    cache hits across whitespace-only re-renders of the same content.
    """
    text = getattr(seg, "text", "") or ""
    return text.strip()


def _seg_start(seg: Any) -> float:
    val = getattr(seg, "start", None)
    if val is None:
        val = getattr(seg, "start_sec", 0.0)
    return float(val)


def _frame_ts(frame: Any) -> float:
    val = getattr(frame, "ts", None)
    if val is None:
        val = getattr(frame, "timestamp", 0.0)
    return float(val)


def _frame_ocr(frame: Any) -> str:
    val = getattr(frame, "ocr", None)
    if val is None:
        val = getattr(frame, "ocr_text", "")
    return (val or "").strip()


def _frame_caption(frame: Any) -> str:
    val = getattr(frame, "caption", "") or ""
    return val.strip()


def compute_prompt_hash(
    *,
    chapter_start_sec: float,
    chapter_end_sec: float,
    chapter_segments: Iterable[Any],
    chapter_frames: Iterable[Any],
    model_id: str,
    prompt_version: str,
) -> str:
    """Stable SHA-256 over the per-chapter map-prompt inputs.

    The payload is canonicalised JSON (sorted keys, no whitespace) so
    that two semantically-identical inputs produce the same digest
    regardless of attribute ordering or Python dict layout. Boundaries
    are rounded to the millisecond — anything finer is below the
    timing precision of either subtitle ASR or our keyframe extractor
    and would cause spurious cache misses on rerun.
    """
    payload = {
        "v": prompt_version,
        "model": model_id,
        "start": round(float(chapter_start_sec), 3),
        "end": round(float(chapter_end_sec), 3),
        "segments": [
            {"ts": round(_seg_start(s), 3), "text": _seg_text(s)}
            for s in chapter_segments
        ],
        "frames": [
            {
                "ts": round(_frame_ts(f), 3),
                "ocr": _frame_ocr(f),
                "caption": _frame_caption(f),
            }
            for f in chapter_frames
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- ChapterCache --------------------------------------------------


class ChapterCache:
    """SQLite-backed cache mapping ``prompt_hash`` → chapter payload.

    Telemetry counters (``hits`` / ``misses`` / ``writes``) live on the
    instance — they reflect the activity of *this* process, not the
    cumulative DB lifetime — and are surfaced via :meth:`stats` for the
    pipeline ``pipeline_stats.chapter_cache`` block we will assemble in
    P7.
    """

    _CREATE_SQL = (
        "CREATE TABLE IF NOT EXISTS chapter_cache ("
        "    prompt_hash    TEXT PRIMARY KEY,"
        "    chapter_json   TEXT NOT NULL,"
        "    model_id       TEXT NOT NULL,"
        "    prompt_version TEXT NOT NULL,"
        "    bv_id          TEXT NOT NULL,"
        "    chapter_index  INTEGER NOT NULL,"
        "    chapter_start  REAL NOT NULL,"
        "    chapter_end    REAL NOT NULL,"
        "    created_at     REAL NOT NULL,"
        "    accessed_at    REAL NOT NULL,"
        "    hit_count      INTEGER NOT NULL DEFAULT 0"
        ")"
    )
    _CREATE_BV_INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS idx_chapter_cache_bv "
        "ON chapter_cache(bv_id)"
    )
    _CREATE_VERSION_INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS idx_chapter_cache_model_version "
        "ON chapter_cache(model_id, prompt_version)"
    )

    def __init__(self, db_path: Path, *, enabled: bool = True) -> None:
        self._db_path = Path(db_path)
        self._enabled = bool(enabled)
        # Per-instance counters; protected by ``_counter_lock`` so the
        # threaded WAL test (and any future concurrent caller) does not
        # tear writes apart on the GIL release boundary.
        self._counter_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        if not self._enabled:
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(self._CREATE_SQL)
                conn.execute(self._CREATE_BV_INDEX_SQL)
                conn.execute(self._CREATE_VERSION_INDEX_SQL)
                conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "ChapterCache init failed for %s: %s — disabling cache",
                self._db_path,
                exc,
            )
            self._enabled = False

    # ---- properties ------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._db_path

    # ---- public API ------------------------------------------------

    def get(self, prompt_hash: str) -> dict | None:
        """Return the cached chapter payload or ``None`` on miss.

        Hits also bump ``accessed_at`` and ``hit_count`` so a future
        cleanup script (P7+) can implement an LRU policy on top of the
        same schema without re-reading every row.
        """
        if not self._enabled or not prompt_hash:
            self._bump("misses")
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT chapter_json FROM chapter_cache "
                    "WHERE prompt_hash = ?",
                    (prompt_hash,),
                ).fetchone()
                if row is None:
                    self._bump("misses")
                    return None
                # Bookkeeping update — best-effort, never fail the
                # actual cache read on a write error.
                try:
                    conn.execute(
                        "UPDATE chapter_cache "
                        "SET hit_count = hit_count + 1, accessed_at = ? "
                        "WHERE prompt_hash = ?",
                        (time.time(), prompt_hash),
                    )
                    conn.commit()
                except sqlite3.OperationalError as exc:  # pragma: no cover
                    logger.debug(
                        "ChapterCache hit_count update failed: %s", exc
                    )
        except sqlite3.OperationalError as exc:
            logger.warning("ChapterCache get failed: %s — treating as miss", exc)
            self._bump("misses")
            return None

        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning(
                "ChapterCache row %s has corrupt JSON — treating as miss",
                prompt_hash[:12],
            )
            self._bump("misses")
            return None

        self._bump("hits")
        return payload

    def put(
        self,
        *,
        prompt_hash: str,
        chapter_payload: dict,
        model_id: str,
        prompt_version: str,
        bv_id: str,
        chapter_index: int,
        chapter_start: float,
        chapter_end: float,
    ) -> None:
        """Upsert a chapter payload under its ``prompt_hash``.

        Uses ``INSERT OR REPLACE`` so concurrent writers racing the
        same primary key never raise — the last writer simply wins,
        which is the right semantics for a deterministic-input cache.
        """
        if not self._enabled or not prompt_hash:
            return
        try:
            blob = json.dumps(chapter_payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "ChapterCache.put dropping non-JSON payload "
                "(hash=%s): %s",
                prompt_hash[:12],
                exc,
            )
            return
        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO chapter_cache "
                    "(prompt_hash, chapter_json, model_id, prompt_version, "
                    " bv_id, chapter_index, chapter_start, chapter_end, "
                    " created_at, accessed_at, hit_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        prompt_hash,
                        blob,
                        model_id,
                        prompt_version,
                        bv_id,
                        int(chapter_index),
                        float(chapter_start),
                        float(chapter_end),
                        now,
                        now,
                    ),
                )
                conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("ChapterCache put failed: %s", exc)
            return
        self._bump("writes")

    def clear_by_bv_id(self, bv_id: str) -> int:
        """Delete every row tagged with ``bv_id``; return delete count.

        Intended for the per-BV ``--force`` retry path — re-running a
        single video should not invalidate cached chapters from other
        videos that happen to live in the same SQLite file.
        """
        if not self._enabled or not bv_id:
            return 0
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM chapter_cache WHERE bv_id = ?",
                    (bv_id,),
                )
                conn.commit()
                return int(cur.rowcount or 0)
        except sqlite3.OperationalError as exc:
            logger.warning("ChapterCache.clear_by_bv_id failed: %s", exc)
            return 0

    def stats(self) -> dict[str, int]:
        """Return per-instance counters plus the live ``rows_total``.

        ``hits`` / ``misses`` / ``writes`` reflect this Python process
        only; ``rows_total`` is queried from the underlying table on
        every call so it is correct even when other processes write
        to the same SQLite file (rare in practice — pipeline workers
        run sequentially per BV — but worth it for cheap correctness).
        """
        rows_total = 0
        if self._enabled:
            try:
                with self._connect() as conn:
                    rows_total = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM chapter_cache"
                        ).fetchone()[0]
                    )
            except sqlite3.OperationalError:
                rows_total = 0
        with self._counter_lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
                "rows_total": rows_total,
            }

    # ---- internals -------------------------------------------------

    def _bump(self, kind: str) -> None:
        with self._counter_lock:
            if kind == "hits":
                self._hits += 1
            elif kind == "misses":
                self._misses += 1
            elif kind == "writes":
                self._writes += 1

    def _connect(self) -> sqlite3.Connection:
        # ``check_same_thread=False`` mirrors the VLMCache approach:
        # every call site opens its own connection inside a ``with``
        # block, never sharing across threads. WAL + timeout=30s gives
        # us safe concurrent writers without explicit locking.
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn


def make_chapter_cache_from_settings(settings: Any) -> ChapterCache:
    """Convenience constructor used by the pipeline (P7) and CLI tools.

    Resolves ``LECTURE_CHAPTER_CACHE_PATH`` against ``DATA_DIR`` when
    blank, mirroring the behaviour of :class:`app.understand.vlm_cache.VLMCache`'s
    home in ``app.understand.frame_describer``.
    """
    raw_path = getattr(settings, "lecture_chapter_cache_path", "") or ""
    if raw_path:
        db_path = Path(raw_path)
    else:
        data_dir = Path(getattr(settings, "data_dir", Path("./data")))
        db_path = data_dir / "chapter_cache.sqlite"
    enabled = bool(getattr(settings, "lecture_chapter_cache_enabled", True))
    return ChapterCache(db_path, enabled=enabled)
