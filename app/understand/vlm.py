"""Frame description via Qwen-VL (DashScope OpenAI-compatible API).

Reworked in M3 to:

* Two-tier prompting via :class:`KeyframeRanker` — HIGH frames get the
  full caption + OCR + scoring prompt, LOW frames collapse to an
  OCR-only prompt with short max_tokens.
* Content-addressed SQLite cache (:class:`VLMCache`) keyed by
  ``(sha256(image), model, tier)``. Adding one new keyframe to a BV
  no longer forces a full re-run; identical slides across videos hit
  the same row.
* Per-call telemetry exposed as ``last_telemetry`` for the pipeline
  to surface in ``pipeline_stats``.

The legacy per-BV ``vlm_cache/<model>/<bv>.json`` files are still
honoured for backward compatibility: if a BV is processed for the
first time after the upgrade and the new SQLite cache is empty for
that BV, we lazy-import the old JSON as HIGH-tier rows. The legacy
file is preserved on disk so users can still roll back by setting
``VLM_TIERING_ENABLED=false`` + ``VLM_CACHE_ENABLED=false``.
"""
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
from .frame_ranker import KeyframeRanker, Tier
from .prompts import (
    VLM_FRAME_DESCRIBE_SYSTEM,
    VLM_FRAME_DESCRIBE_USER,
    VLM_FRAME_OCR_ONLY_USER,
)
from .vlm_cache import VLMCache, sha256_file

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


# Approximate per-frame cost weights, used only for the
# ``estimated_token_savings_pct`` telemetry. The 1.0 / 0.4 split is
# the spec's empirical estimate; once we have real DashScope token
# counts we can recalibrate without changing semantics elsewhere.
_TOKEN_WEIGHT_HIGH = 1.0
_TOKEN_WEIGHT_LOW = 0.4


class FrameDescriber:
    """Send each keyframe to Qwen-VL and collect a short caption + OCR text."""

    def __init__(
        self,
        *,
        ranker: KeyframeRanker | None = None,
        cache: VLMCache | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        s = get_settings()
        self._settings = s
        self._client = client or AsyncOpenAI(
            api_key=s.dashscope_api_key,
            base_url=s.dashscope_base_url,
        )
        self._model = s.qwen_vl_model
        self._concurrency = max(1, int(getattr(s, "qwen_vl_concurrency", 4) or 4))

        self._ranker = ranker or KeyframeRanker(
            junk_sim_threshold=s.vlm_junk_sim_threshold,
            junk_entropy_threshold=s.vlm_junk_entropy_threshold,
            high_floor_ratio=s.vlm_tiering_high_floor_ratio,
            enabled=s.vlm_tiering_enabled,
        )
        if cache is not None:
            self._cache = cache
        else:
            cache_path = (
                Path(s.vlm_cache_path)
                if s.vlm_cache_path
                else s.data_dir / "vlm_cache.sqlite"
            )
            self._cache = VLMCache(cache_path, enabled=s.vlm_cache_enabled)

        self._telemetry: dict[str, Any] = {}

    # ---- public helpers -------------------------------------------------

    @property
    def last_telemetry(self) -> dict[str, Any]:
        """Snapshot of the most recent ``describe_all`` run.

        Returned by reference is unsafe — pipeline.py serialises this
        into JSON and would otherwise mutate our internal state.
        """
        return dict(self._telemetry)

    # ---- main entrypoint ------------------------------------------------

    async def describe_all(self, frames: list[Keyframe]) -> list[FrameDescription]:
        if not frames:
            self._telemetry = self._empty_telemetry()
            return []

        # 1. Tier classification (CPU only).
        rank_result = self._ranker.classify(frames)
        tiers = rank_result.tiers
        junk_count = sum(1 for j in rank_result.is_junk if j)

        # 2. Per-frame SHA-256.
        hashes = [self._safe_sha(f.path) for f in frames]

        # 3. Cache lookup. Tier values are str-Enum members so they
        #    pass through unchanged.
        keys = [(h, self._model, str(t.value)) for h, t in zip(hashes, tiers)]
        cache_hits = self._cache.batch_lookup(keys)

        # 3a. Lazy migration of the legacy per-BV JSON. Only attempt
        #     when the SQLite cache is *fully* empty for this batch
        #     AND the legacy file actually exists — otherwise we'd
        #     pay an OS check on every run.
        if all(v is None for v in cache_hits.values()):
            legacy_path = self._legacy_bv_json_path(frames)
            if legacy_path is not None and legacy_path.exists():
                imported = self._cache.import_legacy_bv_json(legacy_path, self._model)
                if imported:
                    logger.info(
                        "vlm: lazy-imported %d rows from legacy BV cache %s",
                        imported,
                        legacy_path,
                    )
                    cache_hits = self._cache.batch_lookup(keys)

        # 4. Identify misses for VLM call.
        todo: list[tuple[int, Keyframe, Tier, str]] = []
        for idx, (frame, tier, sha) in enumerate(zip(frames, tiers, hashes)):
            if cache_hits.get((sha, self._model, str(tier.value))) is None:
                todo.append((idx, frame, tier, sha))

        # 5. Concurrent VLM calls.
        sem = asyncio.Semaphore(self._concurrency)

        async def _one(frame: Keyframe, tier: Tier) -> FrameDescription:
            async with sem:
                return await self._describe_one(frame, tier=tier)

        fresh_results: list[FrameDescription] = list(
            await asyncio.gather(*[_one(f, t) for _, f, t, _ in todo])
        )

        # 6. Cache write — only payloads from successful calls. We
        #    detect failure heuristically: an empty caption and empty
        #    ocr_text on a HIGH-tier call means the model retry chain
        #    gave up, so caching that would poison future runs.
        rows_to_cache: list[tuple[str, str, str, dict]] = []
        for (idx, frame, tier, sha), desc in zip(todo, fresh_results):
            if _is_failed_response(desc, tier):
                continue
            rows_to_cache.append(
                (sha, self._model, str(tier.value), _payload_of(desc))
            )
        if rows_to_cache:
            self._cache.batch_insert(rows_to_cache)

        # 7. Merge cache hits + fresh results back into input order.
        result: list[FrameDescription | None] = [None] * len(frames)
        for idx, frame, tier, sha in todo:
            pass  # handled below via fresh_results in same order
        for (idx, frame, tier, sha), desc in zip(todo, fresh_results):
            result[idx] = desc
        for idx, (frame, tier, sha) in enumerate(zip(frames, tiers, hashes)):
            if result[idx] is not None:
                continue
            payload = cache_hits.get((sha, self._model, str(tier.value)))
            assert payload is not None  # by construction: every miss is in todo
            result[idx] = _description_from_payload(frame, payload)

        # 8. Telemetry.
        cache_hit_count = len(frames) - len(todo)
        vlm_calls_high = sum(1 for _, _, t, _ in todo if t == Tier.HIGH)
        vlm_calls_low = sum(1 for _, _, t, _ in todo if t == Tier.LOW)
        total = len(frames)
        if total > 0:
            est_cost = vlm_calls_high * _TOKEN_WEIGHT_HIGH + vlm_calls_low * _TOKEN_WEIGHT_LOW
            savings_pct = max(0.0, 100.0 * (1.0 - est_cost / total))
        else:
            savings_pct = 0.0
        self._telemetry = {
            "total_frames": total,
            "junk_filtered": junk_count,
            "high_tier": sum(1 for t in tiers if t == Tier.HIGH),
            "low_tier": sum(1 for t in tiers if t == Tier.LOW),
            "cache_hits": cache_hit_count,
            "vlm_calls_made": len(todo),
            "vlm_calls_high": vlm_calls_high,
            "vlm_calls_low": vlm_calls_low,
            "estimated_token_savings_pct": round(savings_pct, 2),
            "tiering_enabled": self._ranker.enabled,
            "cache_enabled": self._cache.enabled,
        }
        return [d for d in result if d is not None]  # all set by construction

    # ---- internals ------------------------------------------------------

    def _legacy_bv_json_path(self, frames: list[Keyframe]) -> Path | None:
        if not frames:
            return None
        try:
            bv_id = frames[0].path.parent.name
        except (AttributeError, IndexError):
            return None
        if not bv_id:
            return None
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._model)
        return self._settings.data_dir / "vlm_cache" / safe_model / f"{bv_id}.json"

    @staticmethod
    def _safe_sha(path: Path) -> str:
        try:
            return sha256_file(path)
        except OSError as exc:
            logger.warning("vlm: cannot hash frame %s: %s — using path-string fallback", path, exc)
            # Fallback to a stable token derived from the path so the
            # downstream lookup still has a deterministic key. Cache
            # writes will deduplicate naturally.
            return f"NOHASH::{path}"

    def _empty_telemetry(self) -> dict[str, Any]:
        return {
            "total_frames": 0,
            "junk_filtered": 0,
            "high_tier": 0,
            "low_tier": 0,
            "cache_hits": 0,
            "vlm_calls_made": 0,
            "vlm_calls_high": 0,
            "vlm_calls_low": 0,
            "estimated_token_savings_pct": 0.0,
            "tiering_enabled": self._ranker.enabled,
            "cache_enabled": self._cache.enabled,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
    async def _describe_one(self, frame: Keyframe, *, tier: Tier = Tier.HIGH) -> FrameDescription:
        b64 = base64.b64encode(frame.path.read_bytes()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        if tier == Tier.LOW:
            user_prompt = VLM_FRAME_OCR_ONLY_USER
            extra_kwargs: dict[str, Any] = {"max_tokens": 200, "temperature": 0.0}
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
        else:
            user_prompt = VLM_FRAME_DESCRIBE_USER
            extra_kwargs = {"temperature": 0.2}
            messages = [
                {"role": "system", "content": VLM_FRAME_DESCRIBE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **extra_kwargs,
            )
            content = (resp.choices[0].message.content or "").strip()
            if tier == Tier.LOW:
                parsed = _parse_low_tier_response(content)
            else:
                parsed = _parse_vlm_response(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VLM call failed for frame %s (tier=%s): %s",
                frame.path.name,
                tier.value,
                exc,
            )
            parsed = _empty_parse()

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


# ---- payload helpers (cache <-> FrameDescription) -------------------


def _payload_of(desc: FrameDescription) -> dict[str, Any]:
    """Project a :class:`FrameDescription` into the cache payload.

    ``timestamp`` and ``path`` are intentionally excluded — they are
    per-instance metadata, not properties of the visual content. The
    cache payload is meant to be reusable across frames that happen
    to share a SHA-256 digest (same image content reused across
    timestamps / videos).
    """
    return {
        "caption": desc.caption,
        "ocr_text": desc.ocr_text,
        "visual_type": desc.visual_type,
        "importance_score": desc.importance_score,
        "ocr_density": desc.ocr_density,
        "novelty_score": desc.novelty_score,
        "why_useful": desc.why_useful,
    }


def _description_from_payload(frame: Keyframe, payload: dict[str, Any]) -> FrameDescription:
    return FrameDescription(
        timestamp=frame.timestamp,
        path=frame.path,
        caption=str(payload.get("caption", "")),
        ocr_text=str(payload.get("ocr_text", "")),
        visual_type=str(payload.get("visual_type", "other") or "other"),
        importance_score=_coerce_score(payload.get("importance_score")),
        ocr_density=_coerce_score(payload.get("ocr_density")),
        novelty_score=_coerce_score(payload.get("novelty_score")),
        why_useful=str(payload.get("why_useful", "")),
    )


def _is_failed_response(desc: FrameDescription, tier: Tier) -> bool:
    """Heuristic: treat fully empty HIGH-tier responses as failures.

    LOW-tier responses are allowed to be empty (a frame with no
    visible text legitimately has ocr_text=""), so we only guard
    HIGH tier where caption + ocr_text + why_useful all empty +
    importance==0 strongly suggests the retry chain bailed out and
    parser hit ``_empty_parse``.
    """
    if tier != Tier.HIGH:
        return False
    return (
        not desc.caption
        and not desc.ocr_text
        and not desc.why_useful
        and desc.importance_score == 0.0
    )


# ---- parsers --------------------------------------------------------


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_vlm_response(content: str) -> dict[str, Any]:
    """Best-effort extraction of {caption, ocr_text} for HIGH tier.

    Models occasionally wrap JSON in markdown fences or precede it
    with prose; tolerate both.
    """
    if not content:
        return _empty_parse()
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
    parsed = _empty_parse()
    parsed["caption"] = cleaned[:120]
    return parsed


def _parse_low_tier_response(content: str) -> dict[str, Any]:
    """Extract ``ocr_text`` only from a LOW-tier response.

    LOW-tier responses are *supposed* to be ``{"ocr_text": "..."}``
    but models occasionally drift (extra fields, a stray Markdown
    fence, or — worst case — a single unwrapped sentence). We accept
    any of those, project to ocr_text only, and fill the remaining
    FrameDescription fields with neutral defaults so downstream
    consumers do not need to special-case LOW-tier data.

    ``ocr_density`` is computed from the captured ``ocr_text`` length
    (cap at 200 chars → 1.0). It is purely a ranking aid for the IR
    builder; LOW-tier frames already have ``importance_score=0`` so
    they sit below HIGH-tier frames in any visual_evidence ordering.
    """
    parsed = _empty_parse()
    if not content:
        return parsed
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
    ocr_text = ""
    candidates: list[str] = [cleaned]
    candidates.extend(_JSON_OBJ_RE.findall(cleaned))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "ocr_text" in obj:
            ocr_text = str(obj.get("ocr_text", "")).strip()
            break
    if not ocr_text:
        # Model returned a free-form blob; treat the whole thing as
        # OCR text (truncated). Better than nothing for downstream
        # search even if the formatting is messy.
        ocr_text = cleaned[:400]
    parsed["ocr_text"] = ocr_text
    parsed["ocr_density"] = min(len(ocr_text) / 200.0, 1.0)
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
