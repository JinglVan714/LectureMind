"""Frame description via Qwen-VL-Max (DashScope OpenAI-compatible API)."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..ingest.keyframe import Keyframe
from .prompts import VLM_FRAME_DESCRIBE_SYSTEM, VLM_FRAME_DESCRIBE_USER

logger = logging.getLogger(__name__)


@dataclass
class FrameDescription:
    timestamp: float
    path: Path
    caption: str
    ocr_text: str
    visual_type: str = "other"
    importance_score: float = 0.0
    ocr_density: float = 0.0
    novelty_score: float = 0.0
    why_useful: str = ""


class FrameDescriber:
    """Send each keyframe to Qwen-VL and collect a short caption + OCR text."""

    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        self._client = AsyncOpenAI(
            api_key=s.dashscope_api_key,
            base_url=s.dashscope_base_url,
        )
        self._model = s.qwen_vl_model
        self._concurrency = 4

    async def describe_all(self, frames: list[Keyframe]) -> list[FrameDescription]:
        if not frames:
            return []
        cached = self._load_cache(frames)
        if cached is not None:
            return cached
        sem = asyncio.Semaphore(self._concurrency)

        async def _one(f: Keyframe) -> FrameDescription:
            async with sem:
                return await self._describe_one(f)

        descriptions = list(await asyncio.gather(*[_one(f) for f in frames]))
        self._write_cache(descriptions)
        return descriptions

    def _cache_path(self, frames: list[Keyframe]) -> Path | None:
        if not frames:
            return None
        try:
            bv_id = frames[0].path.parent.name
        except IndexError:
            return None
        return self._settings.data_dir / "vlm_cache" / f"{bv_id}.json"

    def _load_cache(self, frames: list[Keyframe]) -> list[FrameDescription] | None:
        cache_path = self._cache_path(frames)
        if not cache_path or not cache_path.exists():
            return None
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid VLM cache %s: %s", cache_path, exc)
            return None
        by_path = {
            str(item.get("path")): item
            for item in raw
            if isinstance(item, dict) and item.get("path")
        }
        descriptions: list[FrameDescription] = []
        for frame in frames:
            item = by_path.get(str(frame.path))
            if not item:
                return None
            descriptions.append(
                FrameDescription(
                    timestamp=float(item.get("timestamp", frame.timestamp)),
                    path=frame.path,
                    caption=str(item.get("caption", "")),
                    ocr_text=str(item.get("ocr_text", "")),
                    visual_type=str(item.get("visual_type", "other")),
                    importance_score=float(item.get("importance_score", 0.0) or 0.0),
                    ocr_density=float(item.get("ocr_density", 0.0) or 0.0),
                    novelty_score=float(item.get("novelty_score", 0.0) or 0.0),
                    why_useful=str(item.get("why_useful", "")),
                )
            )
        logger.info(
            "Loaded %d cached VLM frame descriptions from %s",
            len(descriptions),
            cache_path,
        )
        return descriptions

    def _write_cache(self, descriptions: list[FrameDescription]) -> None:
        if not descriptions:
            return
        cache_path = (
            self._settings.data_dir
            / "vlm_cache"
            / f"{descriptions[0].path.parent.name}.json"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "timestamp": d.timestamp,
                        "path": str(d.path),
                        "caption": d.caption,
                        "ocr_text": d.ocr_text,
                        "visual_type": d.visual_type,
                        "importance_score": d.importance_score,
                        "ocr_density": d.ocr_density,
                        "novelty_score": d.novelty_score,
                        "why_useful": d.why_useful,
                    }
                    for d in descriptions
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
    async def _describe_one(self, frame: Keyframe) -> FrameDescription:
        b64 = base64.b64encode(frame.path.read_bytes()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": VLM_FRAME_DESCRIBE_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": VLM_FRAME_DESCRIBE_USER},
                        ],
                    },
                ],
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_vlm_response(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM call failed for frame %s: %s", frame.path.name, exc)
            parsed = {
                "caption": "",
                "ocr_text": "",
                "visual_type": "other",
                "importance_score": 0.0,
                "ocr_density": 0.0,
                "novelty_score": 0.0,
                "why_useful": "",
            }

        return FrameDescription(
            timestamp=frame.timestamp,
            path=frame.path,
            caption=parsed["caption"],
            ocr_text=parsed["ocr_text"],
            visual_type=parsed["visual_type"],
            importance_score=parsed["importance_score"],
            ocr_density=parsed["ocr_density"],
            novelty_score=parsed["novelty_score"],
            why_useful=parsed["why_useful"],
        )


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_vlm_response(content: str) -> dict[str, Any]:
    """Best-effort extraction of {caption, ocr_text}.

    Models occasionally wrap JSON in markdown fences or precede it with prose;
    tolerate both.
    """
    if not content:
        return _empty_parse()
    # Strip ```json fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
    candidates: list[str] = [cleaned]
    candidates.extend(_JSON_OBJ_RE.findall(cleaned))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return {
                    "caption": str(obj.get("caption", "")).strip(),
                    "ocr_text": str(obj.get("ocr_text", "")).strip(),
                    "visual_type": str(obj.get("visual_type", "other") or "other").strip(),
                    "importance_score": _coerce_score(obj.get("importance_score")),
                    "ocr_density": _coerce_score(obj.get("ocr_density")),
                    "novelty_score": _coerce_score(obj.get("novelty_score")),
                    "why_useful": str(obj.get("why_useful", "")).strip(),
                }
        except json.JSONDecodeError:
            continue
    # Fallback: treat entire content as caption
    parsed = _empty_parse()
    parsed["caption"] = cleaned[:120]
    return parsed


def _empty_parse() -> dict[str, Any]:
    return {
        "caption": "",
        "ocr_text": "",
        "visual_type": "other",
        "importance_score": 0.0,
        "ocr_density": 0.0,
        "novelty_score": 0.0,
        "why_useful": "",
    }


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 1.0))
