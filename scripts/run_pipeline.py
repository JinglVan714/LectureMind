"""CLI to run the full pipeline against a single BV without the web UI.

Useful for debugging individual stages without authenticating through HTTP.

Usage:
    python -m scripts.run_pipeline https://www.bilibili.com/video/BVxxxxxxxxxx
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app.config import get_settings
from app.pipeline import Pipeline
from app.storage.db import Database


async def _async_main(url: str, force_refresh: bool) -> int:
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    s = get_settings()
    db = Database(s.db_path)
    await db.init()
    pipeline = Pipeline(db)

    async def progress(p: int, msg: str) -> None:
        print(f"  [{p:3d}%] {msg}", file=sys.stderr)

    report = await pipeline.run(url, force_refresh=force_refresh, progress_cb=progress)
    print(f"OK · report at {report}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="B 站视频链接或 BV 号")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新生成")
    args = ap.parse_args()
    rc = asyncio.run(_async_main(args.url, args.force))
    sys.exit(rc)


if __name__ == "__main__":
    main()
