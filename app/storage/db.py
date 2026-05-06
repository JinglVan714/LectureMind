"""SQLite schema, DAO, and connection helpers."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

logger = logging.getLogger(__name__)


# Step 1 — base CREATE TABLE statements only (no indexes referencing columns
# that may need to be migrated in).  Safe to run on both fresh and legacy DBs.
SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS summaries (
  bv_id        TEXT PRIMARY KEY,
  url          TEXT NOT NULL,
  title        TEXT,
  author       TEXT,
  duration     INTEGER,
  cover_url    TEXT,
  category     TEXT DEFAULT 'lecture',
  domain_tags  TEXT DEFAULT '[]',
  domain       TEXT,
  direction    TEXT,
  summary_json TEXT NOT NULL,
  report_path  TEXT NOT NULL,
  model_used   TEXT,
  token_cost   INTEGER DEFAULT 0,
  status       TEXT DEFAULT 'done',
  error_msg    TEXT,
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assets (
  bv_id TEXT NOT NULL,
  kind  TEXT NOT NULL,
  path  TEXT NOT NULL,
  meta  TEXT DEFAULT '{}',
  PRIMARY KEY (bv_id, kind, path)
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id     TEXT PRIMARY KEY,
  bv_id      TEXT,
  status     TEXT NOT NULL,
  progress   INTEGER DEFAULT 0,
  error_msg  TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ---- Copilot RAG layer (v1) ----
-- Structured chunk store; one row per teaching_note paragraph / point /
-- pitfall / knowledge_unit / frame_ocr.  See spec for kind taxonomy.
CREATE TABLE IF NOT EXISTS lecture_chunks (
  chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  bv_id        TEXT NOT NULL,
  kind         TEXT NOT NULL,
  chapter_idx  INTEGER,
  t_start      INTEGER,
  t_end        INTEGER,
  text         TEXT NOT NULL,
  meta_json    TEXT DEFAULT '{}',
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Generic key/value bag for RAG metadata (embedding_model, embedding_dim,
-- last_reindex_at, ...).  Keeps Stage-2 free to evolve schema without DDL.
CREATE TABLE IF NOT EXISTS kv (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- BM25 index over lecture_chunks.text.  Triggers (in SCHEMA_AUX) keep it in sync.
-- Note: lecture_chunk_vecs (sqlite-vec virtual table) is created lazily by
-- app/copilot/rag.py once the embedding dimension is known (Stage 2).
CREATE VIRTUAL TABLE IF NOT EXISTS lecture_chunk_fts USING fts5(
  text,
  content='lecture_chunks', content_rowid='chunk_id',
  tokenize='unicode61'
);
"""

# Step 2 — indexes + triggers.  Runs *after* legacy column migrations so any
# index/trigger that references a new column finds it.
SCHEMA_AUX = """
CREATE INDEX IF NOT EXISTS idx_summaries_created ON summaries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_domain ON summaries(domain);
CREATE INDEX IF NOT EXISTS idx_summaries_direction ON summaries(direction);

CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chunks_bv ON lecture_chunks(bv_id);
CREATE INDEX IF NOT EXISTS idx_chunks_bv_chapter ON lecture_chunks(bv_id, chapter_idx);

CREATE TRIGGER IF NOT EXISTS lecture_chunks_ai
AFTER INSERT ON lecture_chunks BEGIN
  INSERT INTO lecture_chunk_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS lecture_chunks_ad
AFTER DELETE ON lecture_chunks BEGIN
  INSERT INTO lecture_chunk_fts(lecture_chunk_fts, rowid, text)
  VALUES('delete', old.chunk_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS lecture_chunks_au
AFTER UPDATE ON lecture_chunks BEGIN
  INSERT INTO lecture_chunk_fts(lecture_chunk_fts, rowid, text)
  VALUES('delete', old.chunk_id, old.text);
  INSERT INTO lecture_chunk_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
"""

# Backwards-compat alias for any external caller that imported SCHEMA.
SCHEMA = SCHEMA_TABLES + SCHEMA_AUX


# Columns we may need to add to legacy `summaries` rows.  SQLite has no
# "ADD COLUMN IF NOT EXISTS" so we apply each ALTER conditionally.
_SUMMARIES_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("domain", "ALTER TABLE summaries ADD COLUMN domain TEXT"),
    ("direction", "ALTER TABLE summaries ADD COLUMN direction TEXT"),
)


@dataclass
class SummaryRow:
    bv_id: str
    url: str
    title: str | None
    author: str | None
    duration: int | None
    cover_url: str | None
    category: str
    domain_tags: list[str]
    summary_json: dict[str, Any]
    report_path: str
    model_used: str | None
    token_cost: int
    status: str
    error_msg: str | None
    created_at: str
    updated_at: str
    domain: str | None = None
    direction: str | None = None

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "SummaryRow":
        keys = row.keys()
        return cls(
            bv_id=row["bv_id"],
            url=row["url"],
            title=row["title"],
            author=row["author"],
            duration=row["duration"],
            cover_url=row["cover_url"],
            category=row["category"] or "lecture",
            domain_tags=json.loads(row["domain_tags"] or "[]"),
            summary_json=json.loads(row["summary_json"] or "{}"),
            report_path=row["report_path"],
            model_used=row["model_used"],
            token_cost=row["token_cost"] or 0,
            status=row["status"] or "done",
            error_msg=row["error_msg"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            domain=row["domain"] if "domain" in keys else None,
            direction=row["direction"] if "direction" in keys else None,
        )


@dataclass
class JobRow:
    job_id: str
    bv_id: str | None
    status: str
    progress: int
    error_msg: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "JobRow":
        return cls(
            job_id=row["job_id"],
            bv_id=row["bv_id"],
            status=row["status"],
            progress=row["progress"] or 0,
            error_msg=row["error_msg"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class Database:
    """Thin async DAO over SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_TABLES)
            await self._migrate_summaries(db)
            await db.executescript(SCHEMA_AUX)
            await db.commit()
        logger.info("SQLite schema initialised at %s", self.db_path)

    async def _migrate_summaries(self, db: aiosqlite.Connection) -> None:
        """Apply additive `summaries` migrations idempotently.

        SQLite lacks `ADD COLUMN IF NOT EXISTS`, so we inspect
        ``PRAGMA table_info`` and only run ALTER for missing columns. Running
        this against a freshly created schema (which already declares the
        columns) is a no-op.
        """
        cur = await db.execute("PRAGMA table_info(summaries)")
        existing = {row[1] for row in await cur.fetchall()}
        for column, ddl in _SUMMARIES_MIGRATIONS:
            if column not in existing:
                logger.info("Migrating summaries table: %s", ddl)
                await db.execute(ddl)

    def _conn(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.db_path)

    # ---- summaries ----

    async def get_summary(self, bv_id: str) -> SummaryRow | None:
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM summaries WHERE bv_id = ?", (bv_id,))
            row = await cur.fetchone()
            return SummaryRow.from_row(row) if row else None

    async def list_summaries(self, limit: int = 50, offset: int = 0) -> list[SummaryRow]:
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cur.fetchall()
            return [SummaryRow.from_row(r) for r in rows]

    async def upsert_summary(
        self,
        *,
        bv_id: str,
        url: str,
        title: str | None,
        author: str | None,
        duration: int | None,
        cover_url: str | None,
        category: str,
        domain_tags: Iterable[str],
        summary_json: dict[str, Any],
        report_path: str,
        model_used: str | None,
        token_cost: int = 0,
        status: str = "done",
        error_msg: str | None = None,
        domain: str | None = None,
        direction: str | None = None,
    ) -> None:
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO summaries (
                  bv_id, url, title, author, duration, cover_url,
                  category, domain_tags, domain, direction,
                  summary_json, report_path,
                  model_used, token_cost, status, error_msg, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bv_id) DO UPDATE SET
                  url=excluded.url, title=excluded.title, author=excluded.author,
                  duration=excluded.duration, cover_url=excluded.cover_url,
                  category=excluded.category, domain_tags=excluded.domain_tags,
                  domain=excluded.domain, direction=excluded.direction,
                  summary_json=excluded.summary_json, report_path=excluded.report_path,
                  model_used=excluded.model_used, token_cost=excluded.token_cost,
                  status=excluded.status, error_msg=excluded.error_msg,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (
                    bv_id, url, title, author, duration, cover_url,
                    category, json.dumps(list(domain_tags), ensure_ascii=False),
                    domain, direction,
                    json.dumps(summary_json, ensure_ascii=False), report_path,
                    model_used, token_cost, status, error_msg,
                ),
            )
            await db.commit()

    async def update_taxonomy(
        self,
        bv_id: str,
        *,
        domain: str | None = None,
        direction: str | None = None,
        domain_tags: Iterable[str] | None = None,
    ) -> bool:
        """Patch taxonomy columns without touching the rest of the row.

        Used by Stage-8 manual re-grouping and the ``--taxonomy-only``
        reindex path.  Returns ``True`` when a row was updated.
        """
        sets: list[str] = ["updated_at=CURRENT_TIMESTAMP"]
        args: list[Any] = []
        if domain is not None:
            sets.append("domain=?")
            args.append(domain)
        if direction is not None:
            sets.append("direction=?")
            args.append(direction)
        if domain_tags is not None:
            sets.append("domain_tags=?")
            args.append(json.dumps(list(domain_tags), ensure_ascii=False))
        if len(sets) == 1:
            return False
        args.append(bv_id)
        async with self._conn() as db:
            cur = await db.execute(
                f"UPDATE summaries SET {', '.join(sets)} WHERE bv_id=?",
                args,
            )
            await db.commit()
            return cur.rowcount > 0

    # ---- kv (Copilot RAG metadata) ----

    async def kv_get(self, key: str) -> str | None:
        async with self._conn() as db:
            cur = await db.execute("SELECT value FROM kv WHERE key=?", (key,))
            row = await cur.fetchone()
            return row[0] if row else None

    async def kv_set(self, key: str, value: str) -> None:
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            await db.commit()

    # ---- assets ----

    async def add_asset(self, bv_id: str, kind: str, path: str, meta: dict[str, Any] | None = None) -> None:
        async with self._conn() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO assets (bv_id, kind, path, meta)
                VALUES (?, ?, ?, ?)
                """,
                (bv_id, kind, path, json.dumps(meta or {}, ensure_ascii=False)),
            )
            await db.commit()

    # ---- jobs ----

    async def create_job(self, job_id: str, bv_id: str | None) -> None:
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO jobs (job_id, bv_id, status, progress) VALUES (?, ?, 'queued', 0)",
                (job_id, bv_id),
            )
            await db.commit()

    async def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        error_msg: str | None = None,
        bv_id: str | None = None,
    ) -> None:
        sets: list[str] = ["updated_at=CURRENT_TIMESTAMP"]
        args: list[Any] = []
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if progress is not None:
            sets.append("progress=?")
            args.append(progress)
        if error_msg is not None:
            sets.append("error_msg=?")
            args.append(error_msg)
        if bv_id is not None:
            sets.append("bv_id=?")
            args.append(bv_id)
        args.append(job_id)
        async with self._conn() as db:
            await db.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?",
                args,
            )
            await db.commit()

    async def get_job(self, job_id: str) -> JobRow | None:
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
            row = await cur.fetchone()
            return JobRow.from_row(row) if row else None
