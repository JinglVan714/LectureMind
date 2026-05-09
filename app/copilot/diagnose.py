from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings


def write_timing_json(bv_id: str, stats: dict[str, Any], *, debug_dir: Path | None = None) -> Path:
    base = debug_dir or (get_settings().data_dir / "debug")
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "bv_id": bv_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    path = base / f"{bv_id}.timing.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
