"""Backfill / rebuild the Copilot RAG index over existing summaries.

Typical workflows::

    # full rebuild on the existing 11 BVs
    python scripts/reindex_copilot.py --all

    # rebuild a single lecture (useful while tuning chunking)
    python scripts/reindex_copilot.py --bv BV1o2421A7Dr

    # change embedding model in .env, then drop the vec table + re-index
    python scripts/reindex_copilot.py --reset-vectors --all

    # only refresh summaries.domain / direction (Stage 8 manual fix-up,
    # without re-embedding)
    python scripts/reindex_copilot.py --all --taxonomy-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow running as ``python scripts/reindex_copilot.py``
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.copilot.indexer import index_lecture  # noqa: E402
from app.copilot.rag import RAGStore  # noqa: E402
from app.copilot.taxonomy import normalize as normalize_taxonomy  # noqa: E402
from app.storage.db import Database  # noqa: E402
from app.understand.ir import LectureIR  # noqa: E402
from app.understand.schema import LectureJSON  # noqa: E402

logger = logging.getLogger("reindex_copilot")


def _load_lecture(summary_json: dict) -> LectureJSON | None:
    try:
        return LectureJSON.model_validate(summary_json)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LectureJSON validation failed: %s", exc)
        return None


def _load_lecture_ir(bv_id: str, debug_dir: Path) -> LectureIR | None:
    path = debug_dir / f"{bv_id}.lecture_ir.json"
    if not path.exists():
        return None
    try:
        return LectureIR.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LectureIR debug dump invalid for %s: %s", bv_id, exc)
        return None


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    db_path = settings.data_dir / "app.db"
    debug_dir = settings.data_dir / "debug"

    db = Database(db_path)
    await db.init()

    if args.bv:
        rows = []
        s = await db.get_summary(args.bv)
        if s is None:
            logger.error("BV %s not found in summaries", args.bv)
            return 2
        rows = [s]
    else:
        rows = await db.list_summaries(limit=10000)

    if not rows:
        logger.warning("No summaries to process; nothing to do.")
        return 0

    rag = RAGStore(db_path)
    await rag.init()

    if args.reset_vectors:
        logger.info("Resetting vector table and embedding_dim …")
        await rag.reset_vectors()

    success = 0
    skipped = 0
    failed = 0
    for s in rows:
        bv = s.bv_id
        lecture = _load_lecture(s.summary_json)
        if lecture is None:
            failed += 1
            continue

        # 1. Re-normalise the taxonomy so any prompt drift / off-list values
        #    in stored summary_json get cleaned up alongside the index.
        if lecture.taxonomy is not None:
            normalized = normalize_taxonomy(lecture.taxonomy)
            if normalized is not None and (
                normalized.domain != s.domain
                or normalized.direction != s.direction
                or sorted(normalized.tags) != sorted(s.domain_tags)
            ):
                await db.update_taxonomy(
                    bv,
                    domain=normalized.domain or None,
                    direction=normalized.direction or None,
                    domain_tags=normalized.tags,
                )
                logger.info(
                    "Updated taxonomy for %s: %s / %s (tags=%d)",
                    bv,
                    normalized.domain,
                    normalized.direction,
                    len(normalized.tags),
                )

        if args.taxonomy_only:
            success += 1
            continue

        lecture_ir = _load_lecture_ir(bv, debug_dir)
        n = await index_lecture(rag, bv, lecture, lecture_ir)
        if n > 0:
            success += 1
            logger.info("Indexed %d chunks for %s", n, bv)
        else:
            skipped += 1
            logger.info("No chunks produced for %s", bv)

    await rag.close()
    logger.info(
        "Done. processed=%d success=%d skipped=%d failed=%d",
        len(rows), success, skipped, failed,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bv", help="single BV id to (re)index")
    g.add_argument("--all", action="store_true", help="(re)index every summaries row")
    p.add_argument(
        "--reset-vectors",
        action="store_true",
        help="drop lecture_chunk_vecs + clear embedding_dim before re-indexing",
    )
    p.add_argument(
        "--taxonomy-only",
        action="store_true",
        help="only refresh summaries.domain / direction / domain_tags; skip embeddings",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
