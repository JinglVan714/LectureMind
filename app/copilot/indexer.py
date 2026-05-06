"""Indexing glue between the pipeline and the RAG store.

The indexer is intentionally tiny: it fans out the chunking + embedding
+ upsert sequence and converts every failure into a ``WARNING`` log so
that lecture rendering never blocks on RAG availability.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.understand.ir import LectureIR
from app.understand.schema import LectureJSON

from .rag import RAGStore, chunk_lecture

logger = logging.getLogger(__name__)


async def index_lecture(
    rag: RAGStore,
    bv_id: str,
    lecture: LectureJSON,
    lecture_ir: Optional[LectureIR] = None,
) -> int:
    """Chunk + embed + upsert one lecture.

    Returns the number of chunks indexed.  All errors are caught and
    logged at WARN level: indexing is best-effort, the lecture report is
    still produced even if RAG misbehaves.  Callers should not rely on
    the return value for correctness, only for observability.
    """
    try:
        chunks = chunk_lecture(lecture, lecture_ir)
        if not chunks:
            logger.info("No chunks to index for %s; skipping.", bv_id)
            return 0
        n = await rag.upsert_chunks(bv_id, chunks)
        logger.info("Indexed %d chunk(s) for %s into Copilot RAG.", n, bv_id)
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("Copilot indexing failed for %s: %s", bv_id, exc, exc_info=True)
        return 0
