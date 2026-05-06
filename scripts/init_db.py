"""One-shot SQLite schema initialiser.

Useful when you want to verify the DB layout without spinning up the
FastAPI app (the app itself also runs init() on startup).

Usage:
    python -m scripts.init_db
"""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.storage.db import Database


async def main() -> None:
    s = get_settings()
    db = Database(s.db_path)
    await db.init()
    print(f"OK · schema initialised at {s.db_path}")


if __name__ == "__main__":
    asyncio.run(main())
