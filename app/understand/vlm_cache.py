"""Content-addressed VLM frame description cache backed by SQLite.

Replaces the per-BV ``vlm_cache/<model>/<bv>.json`` files used by the
legacy :class:`FrameDescriber`. Two improvements over that scheme:

* **Per-frame granularity** — adding one new keyframe to a BV no longer
  invalidates the whole cache. Each row is keyed by
  ``(image_sha256, model, tier)`` so cache hits compose at the frame
  level.
* **Cross-BV reuse** — if the same screenshot reappears across videos
  (e.g. a recurring slide template) we already have its description
  cached.

The implementation is intentionally tiny:

* Standard library ``sqlite3`` only, no ORM.
* New connection per call → simplest thread / process model. SQLite's
  WAL mode handles concurrent writes safely.
* When ``enabled=False`` the cache is a no-op: ``batch_lookup`` returns
  all-misses and ``batch_insert`` does nothing. Useful for A/B or
  emergency disable via ``VLM_CACHE_ENABLED=false``.

Tier is typed as ``str`` here on purpose — the canonical
:class:`Tier` enum lives in :mod:`app.understand.frame_ranker` and
inherits from ``str``, so its members are accepted directly. Keeping
the typing as ``str`` avoids a circular import between the two
modules and preserves the "Phase 1 and Phase 2 are independent"
property of the M3 plan.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_VALID_TIERS = ("high", "low")


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file with SHA-256 and return the hex digest.

    Exposed at module level because :class:`FrameDescriber` (Phase 3)
    needs the same digest to look up rows. Streaming keeps memory flat
    on large keyframes (a 4K screenshot is ~3 MB but reading 50 of
    them at once would still spike).
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


class VLMCache:
    """SQLite-backed cache for VLM frame descriptions."""

    _CREATE_SQL = (
        "CREATE TABLE IF NOT EXISTS vlm_cache ("
        "    image_sha256 TEXT NOT NULL,"
        "    model        TEXT NOT NULL,"
        "    tier         TEXT NOT NULL CHECK (tier IN ('high','low')),"
        "    payload      TEXT NOT NULL,"
        "    created_at   INTEGER NOT NULL,"
        "    PRIMARY KEY (image_sha256, model, tier)"
        ")"
    )
    _CREATE_INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS idx_vlm_cache_created "
        "ON vlm_cache(created_at)"
    )

    def __init__(self, db_path: Path, *, enabled: bool = True) -> None:
        self._db_path = Path(db_path)
        self._enabled = bool(enabled)
        if not self._enabled:
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(self._CREATE_SQL)
                conn.execute(self._CREATE_INDEX_SQL)
                conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "VLMCache init failed for %s: %s — disabling cache",
                self._db_path,
                exc,
            )
            self._enabled = False

    # ---- Public API -------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._db_path

    def batch_lookup(
        self, items: list[tuple[str, str, str]]
    ) -> dict[tuple[str, str, str], dict | None]:
        """Look up many ``(sha256, model, tier)`` keys in one round-trip.

        Returns a dict from key tuple to either the decoded payload
        dict or ``None`` for misses. The order / completeness of the
        dict mirrors ``items``: every input key has an entry, even if
        the row is missing.

        Disabled cache → all-miss dict (not raise) so the caller can
        treat it transparently.
        """
        result: dict[tuple[str, str, str], dict | None] = {(s, m, t): None for s, m, t in items}
        if not self._enabled or not items:
            return result

        # De-dup keys for the SQL IN clause but keep the original
        # `items` list intact so downstream callers can iterate it in
        # their preferred order.
        unique_keys = list({(s, m, t) for s, m, t in items})
        if not unique_keys:
            return result

        try:
            with self._connect() as conn:
                placeholders = ",".join(["(?,?,?)"] * len(unique_keys))
                params: list[str] = []
                for s, m, t in unique_keys:
                    params.extend([s, m, t])
                # SQLite does not natively support tuple-IN, so we use
                # a row-value comparison: WHERE (a,b,c) IN ((?,?,?),...)
                sql = (
                    "SELECT image_sha256, model, tier, payload "
                    "FROM vlm_cache "
                    f"WHERE (image_sha256, model, tier) IN ({placeholders})"
                )
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("VLMCache lookup failed: %s — treating as miss", exc)
            return result

        for sha, model, tier, payload in rows:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning(
                    "VLMCache row has corrupt JSON for sha=%s tier=%s — ignoring",
                    sha,
                    tier,
                )
                continue
            result[(sha, model, tier)] = obj
        return result

    def batch_insert(
        self, rows: list[tuple[str, str, str, dict]]
    ) -> None:
        """Upsert a batch of rows in a single transaction.

        Each row is ``(sha256, model, tier, payload_dict)``. Existing
        rows with the same primary key are overwritten — useful when
        the model is bumped (different ``model`` string) and there is
        no compatibility, but also when a previously cached payload
        needs to be refreshed.
        """
        if not self._enabled or not rows:
            return
        now = int(time.time())
        records = []
        for sha, model, tier, payload in rows:
            if tier not in _VALID_TIERS:
                logger.warning(
                    "VLMCache.batch_insert dropping row with invalid tier=%s",
                    tier,
                )
                continue
            try:
                blob = json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "VLMCache.batch_insert dropping non-JSON payload "
                    "(sha=%s tier=%s): %s",
                    sha,
                    tier,
                    exc,
                )
                continue
            records.append((sha, model, tier, blob, now))
        if not records:
            return
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO vlm_cache "
                    "(image_sha256, model, tier, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    records,
                )
                conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("VLMCache insert failed (%d rows lost): %s", len(records), exc)

    def import_legacy_bv_json(self, json_path: Path, model: str) -> int:
        """Import a legacy per-BV cache file as HIGH tier rows.

        The legacy format is a list of objects, each containing at
        least ``path`` and the various caption / OCR fields. We hash
        the file pointed to by ``path`` and treat the whole row as the
        HIGH tier payload (because the legacy pipeline only had one
        prompt, equivalent to today's HIGH tier).

        Returns the number of rows successfully imported. Missing
        files, malformed JSON or already-cached entries are skipped
        silently. The caller decides whether to delete the legacy
        file afterwards.
        """
        if not self._enabled:
            return 0
        try:
            raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "import_legacy_bv_json: cannot read %s: %s", json_path, exc
            )
            return 0
        if not isinstance(raw, list):
            logger.warning(
                "import_legacy_bv_json: expected list in %s, got %s",
                json_path,
                type(raw).__name__,
            )
            return 0

        rows: list[tuple[str, str, str, dict]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            frame_path = item.get("path")
            if not frame_path:
                continue
            p = Path(str(frame_path))
            if not p.exists():
                continue
            try:
                sha = sha256_file(p)
            except OSError as exc:
                logger.warning("import_legacy_bv_json: hash failed for %s: %s", p, exc)
                continue
            payload = {
                "caption": str(item.get("caption", "")),
                "ocr_text": str(item.get("ocr_text", "")),
                "visual_type": str(item.get("visual_type", "other") or "other"),
                "importance_score": _safe_float(item.get("importance_score")),
                "ocr_density": _safe_float(item.get("ocr_density")),
                "novelty_score": _safe_float(item.get("novelty_score")),
                "why_useful": str(item.get("why_useful", "")),
            }
            rows.append((sha, model, "high", payload))

        if not rows:
            return 0
        before = self.stats().get("row_count", 0)
        self.batch_insert(rows)
        after = self.stats().get("row_count", 0)
        # batch_insert uses INSERT OR REPLACE, so the diff is the lower
        # bound on net new rows. We return len(rows) (rows touched) to
        # match the docstring contract.
        logger.info(
            "import_legacy_bv_json: touched %d rows from %s (db rows %d → %d)",
            len(rows),
            json_path,
            before,
            after,
        )
        return len(rows)

    def stats(self) -> dict[str, int]:
        """Return ``{row_count, db_size_bytes}`` for telemetry / inspection."""
        if not self._enabled:
            return {"row_count": 0, "db_size_bytes": 0}
        try:
            with self._connect() as conn:
                row_count = conn.execute(
                    "SELECT COUNT(*) FROM vlm_cache"
                ).fetchone()[0]
        except sqlite3.OperationalError:
            row_count = 0
        try:
            size = self._db_path.stat().st_size if self._db_path.exists() else 0
        except OSError:
            size = 0
        return {"row_count": int(row_count), "db_size_bytes": int(size)}

    # ---- internals --------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # ``check_same_thread=False`` is safe because each call site
        # opens its own connection inside a ``with`` block; the
        # connection is never shared across threads.
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        # WAL gives us concurrent readers + a single writer without
        # blocking the whole DB. ``synchronous=NORMAL`` is the standard
        # WAL-friendly compromise (slightly weaker fsync but safe under
        # WAL — a sudden crash may lose the most recent transaction
        # but never corrupt the DB).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
