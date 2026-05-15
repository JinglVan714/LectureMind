"""RAG layer for the LectureMind Copilot.

This module owns three concerns that the rest of the Copilot stack depends on:

1. **Chunking** — :class:`Chunk` dataclass and the pure
   :func:`chunk_lecture` function that turns a ``LectureJSON`` (and an
   optional ``LectureIR``) into a flat list of retrievable units.
   See spec section "RAG 索引策略" for the kind taxonomy.

2. **Embeddings** — :class:`EmbeddingClient` wraps the synchronous
   ``dashscope.MultiModalEmbedding`` SDK in :func:`asyncio.to_thread`
   calls.  The first successful call probes the model's embedding
   dimension and persists it to the ``kv`` table; subsequent calls
   validate the dimension to catch silent model swaps.

3. **Hybrid retrieval** — :class:`RAGStore` keeps a *single* synchronous
   :class:`sqlite3.Connection` (because ``aiosqlite`` does not expose
   ``enable_load_extension``), guarded by an ``asyncio.Lock``; it loads
   ``sqlite-vec`` lazily, creates the ``lecture_chunk_vecs`` virtual
   table on first ``upsert_chunks`` (so the ``FLOAT[<dim>]`` declaration
   matches the model), and exposes a
   ``vector top-20 + FTS5 top-20 → RRF → hydrate`` search.

Stage 2 only ships **text** embedding (``embed_text``);
``embed_image`` is implemented but not invoked from the indexer until
``COPILOT_FRAME_VISION_EMBED=true`` is enabled in a later phase.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, cast

from app.config import get_settings
from app.understand.ir import LectureIR
from app.understand.schema import LectureJSON

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ChunkKind = Literal[
    "teaching_note",
    "quote",
    "pitfall",
    "knowledge_unit",
    "frame_ocr",
    # v2 — code/formula first-class, plus study questions for "explain
    # how this lecture answers question X" RAG queries.
    "code_block",
    "formula_block",
    "study_question",
]

_VALID_CHUNK_KINDS = {
    "teaching_note",
    "quote",
    "pitfall",
    "knowledge_unit",
    "frame_ocr",
    "code_block",
    "formula_block",
    "study_question",
}


@dataclass
class Chunk:
    """One retrievable unit.

    Mirrors the columns of the ``lecture_chunks`` SQL table verbatim, with
    ``meta`` serialised to ``meta_json`` at insert time.
    """

    bv_id: str
    kind: ChunkKind
    text: str
    chapter_idx: int | None = None
    t_start: int | None = None
    t_end: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_lecture(
    lecture: LectureJSON,
    lecture_ir: LectureIR | None = None,
) -> list[Chunk]:
    """Turn a ``LectureJSON`` into chunks indexable by the RAG layer.

    The five chunk kinds (per spec § RAG / Chunking 表格):

    * ``teaching_note`` — one chunk per ``chapter.teaching_notes`` paragraph.
    * ``quote`` — one chunk per ``chapter.points`` entry; the chunk text
      combines the LLM's distilled point text with the verbatim subtitle
      quote so both vector and BM25 retrieval can hit it.
    * ``pitfall`` — one chunk per ``chapter.pitfalls`` entry.
    * ``knowledge_unit`` — one chunk per ``LectureJSON.knowledge_units``
      entry (these survived the LectureIR → LectureJSON conversion).
    * ``frame_ocr`` — one chunk per chapter ``Frame`` whose textual
      content (caption / insight / ocr_text) is non-trivial.  ``visual_evidence``
      at the lecture level supplements but is de-duplicated by ``path``.

    Empty / whitespace-only texts are skipped silently (otherwise FTS5
    would index empty rows and RRF would weight them).

    ``lecture_ir`` is currently unused but accepted in the signature so
    future callers (Stage 2 v2 / Stage 3) can pass IR-only metadata
    without changing the contract.
    """
    del lecture_ir  # reserved for future use
    evidence_chunks = _evidence_index_chunks(lecture)
    if evidence_chunks:
        return evidence_chunks
    bv_id = lecture.bv_id
    out: list[Chunk] = []
    seen_frame_paths: set[str] = set()

    for ch in lecture.chapters:
        ch_start = int(ch.start)
        ch_end = int(ch.end) if ch.end > ch.start else ch_start

        # 1. teaching_notes — already paragraph-level after Stage 1
        for note in ch.teaching_notes:
            text = (note or "").strip()
            if not text:
                continue
            out.append(
                Chunk(
                    bv_id=bv_id,
                    kind="teaching_note",
                    text=text,
                    chapter_idx=ch.index,
                    t_start=ch_start,
                    t_end=ch_end,
                )
            )

        # 2. points → quote chunks (point.text + subtitle quote)
        for p in ch.points:
            point_text = (p.text or "").strip()
            quote = (p.quote or "").strip()
            if not point_text and not quote:
                continue
            text = point_text if not quote else f"{point_text}\n字幕：{quote}"
            ts = int(p.ts)
            out.append(
                Chunk(
                    bv_id=bv_id,
                    kind="quote",
                    text=text,
                    chapter_idx=ch.index,
                    t_start=ts,
                    t_end=ts,
                    meta={"quote": quote, "point": point_text},
                )
            )

        # 3. pitfalls
        for pf in ch.pitfalls:
            text = (pf or "").strip()
            if not text:
                continue
            out.append(
                Chunk(
                    bv_id=bv_id,
                    kind="pitfall",
                    text=text,
                    chapter_idx=ch.index,
                    t_start=ch_start,
                    t_end=ch_end,
                )
            )

        # 3b. code_blocks — chapter-attached code excerpts
        for cb in getattr(ch, "code_blocks", []) or []:
            code = (getattr(cb, "code", "") or "").strip()
            if not code:
                continue
            language = (getattr(cb, "language", "") or "text").strip() or "text"
            explanation = (getattr(cb, "explanation", "") or "").strip()
            ts = int(getattr(cb, "ts", ch_start) or ch_start)
            # Embed the explanation alongside the code so vector search
            # on natural-language queries (e.g. "show me the loss code")
            # has something to match in addition to identifiers.
            text = f"[code:{language}]\n{code}"
            if explanation:
                text = f"{text}\n说明：{explanation}"
            out.append(
                Chunk(
                    bv_id=bv_id,
                    kind="code_block",
                    text=text,
                    chapter_idx=ch.index,
                    t_start=ts,
                    t_end=ts,
                    meta={
                        "language": language,
                        "code": code,
                        "source": getattr(cb, "source", "") or "",
                        "explanation": explanation,
                    },
                )
            )

        # 3c. formula_blocks — chapter-attached LaTeX formulas
        for fb in getattr(ch, "formula_blocks", []) or []:
            latex = (getattr(fb, "latex", "") or "").strip()
            if not latex:
                continue
            explanation = (getattr(fb, "explanation", "") or "").strip()
            ts = int(getattr(fb, "ts", ch_start) or ch_start)
            text = f"$$ {latex} $$"
            if explanation:
                text = f"{text}\n说明：{explanation}"
            out.append(
                Chunk(
                    bv_id=bv_id,
                    kind="formula_block",
                    text=text,
                    chapter_idx=ch.index,
                    t_start=ts,
                    t_end=ts,
                    meta={"latex": latex, "explanation": explanation},
                )
            )

        # 4. frame_ocr — chapter-attached frames
        for f in ch.frames:
            if not f.path or f.path in seen_frame_paths:
                continue
            text = _frame_text(f.caption, f.insight, f.ocr_text)
            if not text:
                continue
            seen_frame_paths.add(f.path)
            ts = int(f.ts)
            out.append(
                Chunk(
                    bv_id=bv_id,
                    kind="frame_ocr",
                    text=text,
                    chapter_idx=ch.index,
                    t_start=ts,
                    t_end=ts,
                    meta={
                        "path": f.path,
                        "caption": f.caption,
                        "ocr_text": f.ocr_text,
                        "insight": f.insight,
                        "visual_type": f.visual_type,
                    },
                )
            )

    # 5. knowledge_units (LectureJSON.knowledge_units already mapped from IR)
    for ku in lecture.knowledge_units:
        title = (ku.title or "").strip()
        explanation = (ku.explanation or "").strip()
        if not title and not explanation:
            continue
        text = title if not explanation else f"{title}\n{explanation}"
        ts = int(ku.ts)
        out.append(
            Chunk(
                bv_id=bv_id,
                kind="knowledge_unit",
                text=text,
                chapter_idx=ku.chapter_index,
                t_start=ts,
                t_end=ts,
                meta={
                    "ku_id": ku.id,
                    "ku_type": ku.type,
                    "quote": ku.quote,
                },
            )
        )

    # 6. lecture-level visual_evidence (selected frames not yet covered by chapter.frames)
    for v in lecture.visual_evidence:
        if not v.path or v.path in seen_frame_paths:
            continue
        text = _frame_text(v.caption, v.insight, v.ocr_text)
        if not text:
            continue
        seen_frame_paths.add(v.path)
        ts = int(v.ts)
        ch_idx = _chapter_for_ts(lecture, ts)
        out.append(
            Chunk(
                bv_id=bv_id,
                kind="frame_ocr",
                text=text,
                chapter_idx=ch_idx,
                t_start=ts,
                t_end=ts,
                meta={
                    "path": v.path,
                    "caption": v.caption,
                    "ocr_text": v.ocr_text,
                    "insight": v.insight,
                    "visual_type": v.visual_type,
                },
            )
        )

    # 7. study_questions — one chunk per question; chapter_idx is left
    #    unset because they apply to the whole lecture.  These give the
    #    Copilot a precise hook for "did the lecture answer X?" queries.
    for q in getattr(lecture, "study_questions", []) or []:
        text = (q or "").strip()
        if not text:
            continue
        out.append(
            Chunk(
                bv_id=bv_id,
                kind="study_question",
                text=text,
                chapter_idx=None,
                t_start=None,
                t_end=None,
            )
        )

    return out


def _evidence_index_chunks(lecture: LectureJSON) -> list[Chunk]:
    evidence = lecture.evidence_index
    if evidence is None or not evidence.evidence_objects:
        return []
    out: list[Chunk] = []
    for obj in evidence.evidence_objects:
        for rag_chunk in obj.rag_chunks:
            text = (rag_chunk.text or "").strip()
            if not text:
                continue
            kind = _coerce_chunk_kind(rag_chunk.kind)
            t_start = int(rag_chunk.t_start) if rag_chunk.t_start is not None else None
            t_end = int(rag_chunk.t_end) if rag_chunk.t_end is not None else t_start
            if t_start is not None and (t_end is None or t_end < t_start):
                t_end = t_start
            note_node_id = rag_chunk.note_node_id or (obj.note_node_ids[0] if obj.note_node_ids else "")
            meta = dict(rag_chunk.meta or {})
            meta.setdefault("evidence_id", obj.evidence_id)
            meta.setdefault("note_node_id", note_node_id)
            meta.setdefault("evidence_kind", obj.kind)
            if obj.quote and "quote" not in meta:
                meta["quote"] = obj.quote
            if obj.path and "path" not in meta:
                meta["path"] = obj.path
            out.append(
                Chunk(
                    bv_id=lecture.bv_id,
                    kind=kind,
                    text=text,
                    chapter_idx=rag_chunk.chapter_idx or obj.chapter_index,
                    t_start=t_start,
                    t_end=t_end,
                    meta=meta,
                )
            )
    return out


def _coerce_chunk_kind(kind: str) -> ChunkKind:
    if kind in _VALID_CHUNK_KINDS:
        return cast(ChunkKind, kind)
    return "teaching_note"


def _frame_text(caption: str, insight: str, ocr_text: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for piece in (caption, insight, ocr_text):
        text = (piece or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return " | ".join(parts)


def _chapter_for_ts(lecture: LectureJSON, ts: int) -> int | None:
    for ch in lecture.chapters:
        if ch.start <= ts <= ch.end:
            return ch.index
    return lecture.chapters[0].index if lecture.chapters else None


# ---------------------------------------------------------------------------
# Embeddings — DashScope MultiModalEmbedding wrapper
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """Raised when the upstream DashScope call fails or returns garbage."""


class EmbeddingClient:
    """Async wrapper around ``dashscope.MultiModalEmbedding``.

    The DashScope multimodal SDK is synchronous, so each call is dispatched
    to a worker thread.  An :class:`asyncio.Semaphore` (default 4) bounds
    concurrent in-flight calls per spec.

    ``embed_text(["hello", "world"])`` issues one HTTP call per text and
    returns a list of equal length.  We deliberately avoid relying on
    multi-item batching in a single ``MultiModalEmbedding.call`` because
    its semantics depend on whether the model returns N embeddings or one
    factor-weighted average; per-item is correct in both cases and the
    Semaphore covers throughput.

    ``embed_image`` is implemented for completeness but the indexer does
    not invoke it until ``COPILOT_FRAME_VISION_EMBED`` is enabled.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        s = get_settings()
        self._model = model or s.qwen_embedding_model
        self._api_key = api_key or s.dashscope_api_key
        self._sem = asyncio.Semaphore(max_concurrent or s.copilot_max_concurrent or 4)
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int | None:
        """Embedding dimension once probed; ``None`` before the first call."""
        return self._dim

    async def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)

        async def _run(i: int, text: str) -> None:
            async with self._sem:
                vec = await asyncio.to_thread(self._call_text, text)
            results[i] = vec

        await asyncio.gather(*(_run(i, t) for i, t in enumerate(texts)))
        out = [v for v in results if v is not None]
        if len(out) != len(texts):
            raise EmbeddingError("embed_text returned partial results")
        return out

    async def embed_image(self, paths: Sequence[Path]) -> list[list[float]]:
        if not paths:
            return []
        results: list[list[float] | None] = [None] * len(paths)

        async def _run(i: int, path: Path) -> None:
            async with self._sem:
                vec = await asyncio.to_thread(self._call_image, path)
            results[i] = vec

        await asyncio.gather(*(_run(i, p) for i, p in enumerate(paths)))
        out = [v for v in results if v is not None]
        if len(out) != len(paths):
            raise EmbeddingError("embed_image returned partial results")
        return out

    # -- sync helpers (run inside asyncio.to_thread) -------------------------

    def _call_text(self, text: str) -> list[float]:
        from dashscope import MultiModalEmbedding

        items = [{"text": text}]
        resp = MultiModalEmbedding.call(model=self._model, input=items, api_key=self._api_key)
        return self._extract_vec(resp)

    def _call_image(self, path: Path) -> list[float]:
        from dashscope import MultiModalEmbedding

        items = [{"image": str(path)}]
        resp = MultiModalEmbedding.call(model=self._model, input=items, api_key=self._api_key)
        return self._extract_vec(resp)

    def _extract_vec(self, resp: Any) -> list[float]:
        status_code = getattr(resp, "status_code", None)
        if status_code is not None and status_code != 200:
            code = getattr(resp, "code", "?")
            msg = getattr(resp, "message", "?")
            raise EmbeddingError(f"DashScope embedding failed: {status_code} {code} {msg}")
        output = getattr(resp, "output", None) or {}
        embeddings = output.get("embeddings") if isinstance(output, dict) else None
        if not embeddings:
            raise EmbeddingError(f"DashScope returned no embeddings: {resp!r}")
        first = embeddings[0]
        vec = first.get("embedding") if isinstance(first, dict) else None
        if not vec or not isinstance(vec, list):
            raise EmbeddingError(f"unexpected embedding payload: {first!r}")
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise EmbeddingError(
                f"embedding dim drift: expected {self._dim}, got {len(vec)}"
            )
        return [float(x) for x in vec]


# ---------------------------------------------------------------------------
# Vector store — sqlite-vec + FTS5 hybrid
# ---------------------------------------------------------------------------


_RRF_K = 60  # standard RRF constant
_VEC_TOPK = 20  # candidates pulled from the vector index per query
_FTS_TOPK = 20


class RAGStore:
    """Hybrid sqlite-vec + FTS5 retriever, scoped to one ``app.db``.

    Lifecycle:

    * Construct with the path to the same ``app.db`` used by
      :class:`app.storage.db.Database`.
    * Call :py:meth:`init` once; this opens a synchronous
      :class:`sqlite3.Connection`, attempts to load ``sqlite-vec``, and
      reads the persisted ``embedding_dim`` from ``kv``.  If the
      extension fails to load (e.g. on an environment without the wheel),
      :attr:`vec_available` flips to ``False`` and all subsequent
      ``upsert_chunks`` skip vector inserts; ``search`` quietly degrades
      to FTS-only.
    * Call :py:meth:`upsert_chunks` to (re)index a BV.  The first
      successful call also creates the ``lecture_chunk_vecs`` virtual
      table at the model's embedding dimension.
    * Call :py:meth:`search` — vec top-20 + FTS top-20 → RRF → hydrate.

    All public methods are async and mutually exclusive via
    ``self._lock``; the underlying connection is single-threaded.
    """

    def __init__(self, db_path: Path, embedder: EmbeddingClient | None = None) -> None:
        self._db_path = db_path
        self._embedder = embedder or EmbeddingClient()
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._vec_available = False
        self._vec_table_ready = False
        self._dim: int | None = None
        self._initialised = False

    # -- properties ----------------------------------------------------------

    @property
    def vec_available(self) -> bool:
        """Whether ``sqlite-vec`` loaded successfully on this connection."""
        return self._vec_available

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def embedder(self) -> EmbeddingClient:
        return self._embedder

    # -- init ----------------------------------------------------------------

    async def init(self) -> None:
        """Open the connection, try to load sqlite-vec, restore state from kv."""
        if self._initialised:
            return
        async with self._lock:
            if self._initialised:
                return
            await asyncio.to_thread(self._sync_init)
            self._initialised = True

    def _sync_init(self) -> None:
        # ``check_same_thread=False`` because the connection is reused across
        # different worker threads dispatched by ``asyncio.to_thread``; the
        # asyncio.Lock guarantees one-at-a-time access from our side.
        conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.enable_load_extension(True)
        except sqlite3.NotSupportedError as exc:
            logger.warning(
                "sqlite3 build does not support enable_load_extension (%s); "
                "Copilot RAG will run in FTS-only mode.",
                exc,
            )
            self._vec_available = False
            self._conn = conn
            self._restore_dim_from_kv(conn)
            return
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
            try:
                conn.enable_load_extension(False)
            except sqlite3.NotSupportedError:
                pass
            (vec_version,) = conn.execute("SELECT vec_version()").fetchone()
            logger.info("sqlite-vec %s loaded for Copilot RAG", vec_version)
            self._vec_available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load sqlite-vec extension: %s. RAG will run FTS-only.",
                exc,
            )
            self._vec_available = False
        self._conn = conn
        self._restore_dim_from_kv(conn)
        if self._vec_available and self._dim is not None:
            self._ensure_vec_table(conn, self._dim)
            self._vec_table_ready = True

    def _restore_dim_from_kv(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM kv WHERE key='embedding_dim'").fetchone()
        if row:
            try:
                self._dim = int(row[0])
            except (TypeError, ValueError):
                self._dim = None

    def _ensure_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        # vec0 declarations cannot use parameter binding for the column type;
        # the dimension is interpolated.
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS lecture_chunk_vecs USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}]
            )
            """
        )

    # -- upsert --------------------------------------------------------------

    async def upsert_chunks(self, bv_id: str, chunks: Sequence[Chunk]) -> int:
        """Replace all chunks for ``bv_id`` with ``chunks`` and (re)embed.

        Returns the number of chunks inserted into ``lecture_chunks``.
        Vector inserts are skipped when sqlite-vec is unavailable; FTS5
        inserts are driven by triggers in :mod:`app.storage.db`.
        """
        await self.init()
        if not chunks:
            await asyncio.to_thread(self._sync_purge_bv, bv_id)
            return 0

        # Embed first so that an EmbeddingError aborts before we mutate state.
        vectors: list[list[float]] | None = None
        if self._vec_available:
            vectors = await self._embedder.embed_text([c.text for c in chunks])
            await self._record_dim(self._embedder.dim)

        return await asyncio.to_thread(self._sync_upsert, bv_id, list(chunks), vectors)

    def _sync_purge_bv(self, bv_id: str) -> None:
        assert self._conn is not None
        conn = self._conn
        old_ids = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM lecture_chunks WHERE bv_id=?", (bv_id,)
        ).fetchall()]
        if old_ids:
            placeholders = ",".join(["?"] * len(old_ids))
            conn.execute("BEGIN")
            try:
                conn.execute(f"DELETE FROM lecture_chunks WHERE chunk_id IN ({placeholders})", old_ids)
                if self._vec_available and self._vec_table_ready:
                    conn.execute(
                        f"DELETE FROM lecture_chunk_vecs WHERE chunk_id IN ({placeholders})",
                        old_ids,
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _sync_upsert(
        self,
        bv_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]] | None,
    ) -> int:
        assert self._conn is not None
        conn = self._conn
        # Make sure the vec0 table exists at the right dim before inserting.
        if vectors is not None and not self._vec_table_ready and self._dim is not None:
            self._ensure_vec_table(conn, self._dim)
            self._vec_table_ready = True

        old_ids = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM lecture_chunks WHERE bv_id=?", (bv_id,)
        ).fetchall()]

        conn.execute("BEGIN")
        try:
            if old_ids:
                placeholders = ",".join(["?"] * len(old_ids))
                conn.execute(
                    f"DELETE FROM lecture_chunks WHERE chunk_id IN ({placeholders})",
                    old_ids,
                )
                if self._vec_available and self._vec_table_ready:
                    conn.execute(
                        f"DELETE FROM lecture_chunk_vecs WHERE chunk_id IN ({placeholders})",
                        old_ids,
                    )

            new_ids: list[int] = []
            for chunk in chunks:
                cur = conn.execute(
                    """
                    INSERT INTO lecture_chunks (
                        bv_id, kind, chapter_idx, t_start, t_end, text, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.bv_id,
                        chunk.kind,
                        chunk.chapter_idx,
                        chunk.t_start,
                        chunk.t_end,
                        chunk.text,
                        json.dumps(chunk.meta, ensure_ascii=False) if chunk.meta else "{}",
                    ),
                )
                new_ids.append(cur.lastrowid)

            if vectors is not None and self._vec_available and self._vec_table_ready:
                for chunk_id, vec in zip(new_ids, vectors):
                    blob = struct.pack(f"{len(vec)}f", *vec)
                    conn.execute(
                        "INSERT INTO lecture_chunk_vecs (chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, blob),
                    )

            conn.execute("COMMIT")
            return len(new_ids)
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def _record_dim(self, dim: int | None) -> None:
        if dim is None or dim == self._dim:
            return
        if self._dim is not None and dim != self._dim:
            raise EmbeddingError(
                f"embedding_dim mismatch: stored={self._dim}, new={dim}; "
                f"run scripts/reindex_copilot.py --reset-vectors to rebuild."
            )
        self._dim = dim
        await asyncio.to_thread(self._sync_record_dim, dim, self._embedder.model)

    def _sync_record_dim(self, dim: int, model: str) -> None:
        assert self._conn is not None
        conn = self._conn
        conn.execute(
            "INSERT INTO kv(key,value) VALUES('embedding_dim', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(dim),),
        )
        conn.execute(
            "INSERT INTO kv(key,value) VALUES('embedding_model', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (model,),
        )

    # -- search --------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        bv_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Hybrid vector + FTS5 retrieval, scoped to ``bv_id`` when given.

        Returns a list of dicts with the spec contract:
        ``{chunk_id, bv_id, kind, chapter_idx, t_start, t_end, text, quote, score, meta}``.
        ``quote`` is hoisted out of ``meta`` for ``quote`` / ``frame_ocr``
        chunks so consumers can render anchors without re-parsing JSON.
        """
        await self.init()
        if not query.strip():
            return []

        # Vector branch (skipped if extension missing)
        vec_hits: list[tuple[int, float]] = []
        if self._vec_available and self._vec_table_ready:
            try:
                qvec = (await self._embedder.embed_text([query]))[0]
                await self._record_dim(self._embedder.dim)
                vec_hits = await asyncio.to_thread(
                    self._sync_vec_search, qvec, bv_id, _VEC_TOPK
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vector search failed; falling back to FTS only: %s", exc)
                vec_hits = []

        fts_hits = await asyncio.to_thread(self._sync_fts_search, query, bv_id, _FTS_TOPK)

        if not vec_hits and not fts_hits:
            return []

        merged = _rrf_merge(vec_hits, fts_hits, k=_RRF_K)
        chunk_ids = [cid for cid, _ in merged[:top_k]]
        if not chunk_ids:
            return []

        rows = await asyncio.to_thread(self._sync_hydrate, chunk_ids)
        score_by_id = {cid: score for cid, score in merged}
        for r in rows:
            r["score"] = score_by_id.get(r["chunk_id"], 0.0)
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_k]

    def _sync_vec_search(
        self,
        qvec: list[float],
        bv_id: str | None,
        k: int,
    ) -> list[tuple[int, float]]:
        assert self._conn is not None
        conn = self._conn
        # Over-sample so post-filtering by bv_id still returns enough.
        oversample = k if bv_id is None else max(k, k * 5)
        blob = struct.pack(f"{len(qvec)}f", *qvec)
        params: list[Any] = [blob, oversample]
        sql = (
            "SELECT v.chunk_id AS chunk_id, v.distance AS distance "
            "FROM lecture_chunk_vecs v "
            "WHERE v.embedding MATCH ? AND k = ?"
        )
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        if bv_id is None or not rows:
            return [(int(r["chunk_id"]), float(r["distance"])) for r in rows[:k]]
        ids = [int(r["chunk_id"]) for r in rows]
        placeholders = ",".join(["?"] * len(ids))
        kept = {
            int(r[0])
            for r in conn.execute(
                f"SELECT chunk_id FROM lecture_chunks WHERE bv_id=? AND chunk_id IN ({placeholders})",
                [bv_id, *ids],
            ).fetchall()
        }
        out = [(int(r["chunk_id"]), float(r["distance"])) for r in rows if int(r["chunk_id"]) in kept]
        return out[:k]

    def _sync_fts_search(
        self,
        query: str,
        bv_id: str | None,
        k: int,
    ) -> list[tuple[int, float]]:
        assert self._conn is not None
        conn = self._conn
        fts_q = _escape_fts_query(query)
        if not fts_q:
            return []
        params: list[Any] = [fts_q]
        sql = (
            "SELECT lecture_chunk_fts.rowid AS chunk_id, "
            "       bm25(lecture_chunk_fts) AS rank "
            "FROM lecture_chunk_fts "
            "JOIN lecture_chunks ON lecture_chunks.chunk_id = lecture_chunk_fts.rowid "
            "WHERE lecture_chunk_fts MATCH ?"
        )
        if bv_id is not None:
            sql += " AND lecture_chunks.bv_id = ?"
            params.append(bv_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed for %r: %s", query, exc)
            return []
        return [(int(r["chunk_id"]), float(r["rank"])) for r in rows]

    def _sync_hydrate(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        assert self._conn is not None
        conn = self._conn
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = conn.execute(
            f"SELECT chunk_id, bv_id, kind, chapter_idx, t_start, t_end, text, meta_json "
            f"FROM lecture_chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            meta = _safe_json(r["meta_json"]) or {}
            quote = meta.get("quote") or ""
            out.append(
                {
                    "chunk_id": int(r["chunk_id"]),
                    "bv_id": r["bv_id"],
                    "kind": r["kind"],
                    "chapter_idx": r["chapter_idx"],
                    "t_start": r["t_start"],
                    "t_end": r["t_end"],
                    "text": r["text"],
                    "quote": quote,
                    "meta": meta,
                }
            )
        return out

    # -- maintenance ---------------------------------------------------------

    async def reset_vectors(self) -> None:
        """Drop ``lecture_chunk_vecs`` and clear ``kv.embedding_dim``.

        Used when switching to a different embedding model (different dim).
        Caller is expected to follow up with ``upsert_chunks`` for every BV.
        """
        await self.init()
        async with self._lock:
            await asyncio.to_thread(self._sync_reset_vectors)
            self._vec_table_ready = False
            self._dim = None

    def _sync_reset_vectors(self) -> None:
        assert self._conn is not None
        conn = self._conn
        conn.execute("DROP TABLE IF EXISTS lecture_chunk_vecs")
        conn.execute("DELETE FROM kv WHERE key IN ('embedding_dim', 'embedding_model')")

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None
                self._initialised = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rrf_merge(
    vec: Iterable[tuple[int, float]],
    fts: Iterable[tuple[int, float]],
    *,
    k: int = _RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over (id, distance/rank) pairs.

    The numeric score in either input is *not* used as a weight; only the
    ordering matters, so we re-rank both lists by the input order.
    """
    score: dict[int, float] = {}
    for rank, (cid, _) in enumerate(vec):
        score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(fts):
        score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def _escape_fts_query(query: str) -> str:
    """Convert a free-form user query into a safe FTS5 MATCH expression.

    Strategy: split on whitespace, drop tokens that contain only FTS5
    operator characters, wrap each remaining token in double-quotes (so
    inner punctuation does not blow up the parser), and join with spaces
    (implicit AND).  Empty result returns the empty string and the caller
    should skip the query.
    """
    cleaned = []
    for token in query.replace('"', " ").split():
        token = token.strip(" \t\r\n")
        if not token:
            continue
        # Drop FTS5 reserved bareword operators
        if token.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        cleaned.append(f'"{token}"')
    return " ".join(cleaned)


def _safe_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Public dataclass exposure for tools / tests
# ---------------------------------------------------------------------------


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return asdict(chunk)
