"""Behavioural tests for the M3 :class:`FrameDescriber` rework.

Verifies that:

* HIGH-tier frames are sent the full caption / OCR prompt and LOW-tier
  frames are sent the cheaper OCR-only prompt
* both tiers' results land in the SQLite cache under their tier key
* a re-run hits the cache for every frame (zero new VLM calls) and
  the telemetry reflects that
* legacy per-BV JSON is lazy-imported as HIGH tier on first run
* telemetry numbers are self-consistent (tiers + junk + cache hits +
  vlm calls all add up)

The OpenAI client is stubbed; no network is touched. The
:class:`KeyframeRanker` is also stubbed in most tests so we exercise
the tier dispatch precisely instead of relying on synthetic image
properties to land on the right side of the threshold.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.understand.frame_ranker import (
    FrameSignals,
    KeyframeRanker,
    RankResult,
    Tier,
)
from app.understand.vlm import FrameDescriber
from app.understand.vlm_cache import VLMCache, sha256_file


# ---- helpers --------------------------------------------------------


@dataclass
class _Frame:
    """Stand-in for ``app.ingest.keyframe.Keyframe`` (same shape)."""

    timestamp: float
    path: Path


def _make_jpeg(path: Path, fill: tuple[int, int, int] = (10, 20, 30)) -> _Frame:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=fill)
    img.save(path, format="JPEG", quality=85)
    return _Frame(timestamp=0.0, path=path)


class _StubRanker:
    """KeyframeRanker stand-in returning a fixed tier list."""

    def __init__(self, tiers: list[Tier], junk: list[bool] | None = None) -> None:
        self._tiers = tiers
        self._junk = junk if junk is not None else [False] * len(tiers)
        self.enabled = True

    def classify(self, frames):  # noqa: D401 - mirror real signature
        sigs = [FrameSignals(0.0, 7.0, 0.5) for _ in frames]
        return RankResult(tiers=self._tiers[: len(frames)], signals=sigs, is_junk=self._junk[: len(frames)])

    def rank(self, frames):
        return self.classify(frames).tiers


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]


class _StubCompletions:
    def __init__(self, recorder: list[dict[str, Any]], responder) -> None:
        self._recorder = recorder
        self._responder = responder

    async def create(self, **kwargs) -> _StubResponse:
        self._recorder.append(kwargs)
        return self._responder(kwargs)


class _StubChat:
    def __init__(self, recorder: list[dict[str, Any]], responder) -> None:
        self.completions = _StubCompletions(recorder, responder)


class _StubClient:
    """Drop-in async OpenAI client stand-in.

    Records every call into ``calls`` (in order) and dispatches the
    response through ``responder(kwargs) -> str``.
    """

    def __init__(self, responder=None) -> None:
        self.calls: list[dict[str, Any]] = []
        responder = responder or (lambda kwargs: _default_responder(kwargs))
        self.chat = _StubChat(self.calls, responder)


def _default_responder(kwargs: dict[str, Any]) -> _StubResponse:
    """Return JSON shaped per the prompt detected in the user message.

    Routes by inspecting the last text part of the user message — if
    it contains the OCR-only marker we return a LOW-tier shaped JSON,
    otherwise the full HIGH-tier JSON.
    """
    text = ""
    for msg in kwargs.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
            else:
                text = str(content)
    if "仅返回 JSON" in text:  # LOW-tier prompt marker
        return _StubResponse(json.dumps({"ocr_text": "low-tier ocr text"}, ensure_ascii=False))
    # HIGH-tier shape
    return _StubResponse(
        json.dumps(
            {
                "caption": "a slide",
                "ocr_text": "high tier ocr",
                "visual_type": "slide_text",
                "importance_score": 0.8,
                "ocr_density": 0.5,
                "novelty_score": 0.4,
                "why_useful": "useful",
            },
            ensure_ascii=False,
        )
    )


def _make_describer(
    tmp_path: Path,
    *,
    tiers: list[Tier],
    junk: list[bool] | None = None,
    monkeypatch: pytest.MonkeyPatch,
    responder=None,
) -> tuple[FrameDescriber, _StubClient, VLMCache]:
    # Force the data dir to tmp so VLMCache lands somewhere we control,
    # and make the cache visible across runs.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_CACHE_PATH", str(tmp_path / "v.sqlite"))
    # Reset the lru_cache so settings re-read the env.
    from app import config

    config.get_settings.cache_clear()

    cache = VLMCache(tmp_path / "v.sqlite")
    client = _StubClient(responder=responder)
    describer = FrameDescriber(
        ranker=_StubRanker(tiers, junk=junk),
        cache=cache,
        client=client,
    )
    return describer, client, cache


# ---- tests ----------------------------------------------------------


def test_empty_input_records_zero_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    describer, _, _ = _make_describer(tmp_path, tiers=[], monkeypatch=monkeypatch)
    out = asyncio.run(describer.describe_all([]))
    assert out == []
    t = describer.last_telemetry
    assert t["total_frames"] == 0
    assert t["vlm_calls_made"] == 0


def test_high_and_low_tier_use_distinct_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [
        _make_jpeg(tmp_path / "BVTEST" / f"f{i}.jpg", fill=(i * 30, 50, 50))
        for i in range(2)
    ]
    describer, client, cache = _make_describer(
        tmp_path, tiers=[Tier.HIGH, Tier.LOW], monkeypatch=monkeypatch
    )

    descs = asyncio.run(describer.describe_all(frames))

    # Two calls were made (one per frame, no cache hits on first run).
    assert len(client.calls) == 2

    # Detect prompt by looking at user-message text content.
    def _user_text(call: dict[str, Any]) -> str:
        for msg in call["messages"]:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "text":
                        return part["text"]
        return ""

    user_texts = [_user_text(c) for c in client.calls]
    # HIGH-tier prompt is "请描述这一帧。", LOW-tier prompt mentions "仅返回 JSON".
    assert any("请描述这一帧" in t for t in user_texts)
    assert any("仅返回 JSON" in t for t in user_texts)

    # HIGH-tier call has a system message; LOW-tier does not.
    has_system = ["system" in {m["role"] for m in c["messages"]} for c in client.calls]
    assert sum(has_system) == 1

    # LOW-tier call must set max_tokens; HIGH-tier must not.
    low_call = client.calls[user_texts.index(next(t for t in user_texts if "仅返回 JSON" in t))]
    high_call = client.calls[user_texts.index(next(t for t in user_texts if "请描述这一帧" in t))]
    assert low_call.get("max_tokens") == 200
    assert "max_tokens" not in high_call

    # Both tiers' payloads landed in the SQLite cache.
    keys = [
        (sha256_file(frames[0].path), describer._model, "high"),
        (sha256_file(frames[1].path), describer._model, "low"),
    ]
    hits = cache.batch_lookup(keys)
    assert hits[keys[0]] is not None
    assert hits[keys[1]] is not None
    # The LOW-tier cache row carries the OCR text we returned.
    assert hits[keys[1]]["ocr_text"] == "low-tier ocr text"

    # FrameDescription objects round-trip the data.
    assert len(descs) == 2
    assert descs[0].caption == "a slide"
    assert descs[1].caption == ""  # LOW-tier descriptions have empty caption
    assert descs[1].ocr_text == "low-tier ocr text"


def test_second_run_is_full_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [_make_jpeg(tmp_path / "BV1" / f"f{i}.jpg", fill=(i * 10, 0, 0)) for i in range(3)]
    describer, client, _ = _make_describer(
        tmp_path,
        tiers=[Tier.HIGH, Tier.LOW, Tier.HIGH],
        monkeypatch=monkeypatch,
    )

    asyncio.run(describer.describe_all(frames))
    assert len(client.calls) == 3
    first_telemetry = describer.last_telemetry
    assert first_telemetry["cache_hits"] == 0
    assert first_telemetry["vlm_calls_made"] == 3

    # Second pass — every frame should hit the cache and the stub
    # client should not be called again.
    asyncio.run(describer.describe_all(frames))
    assert len(client.calls) == 3, "second run must not call VLM"
    second_telemetry = describer.last_telemetry
    assert second_telemetry["cache_hits"] == 3
    assert second_telemetry["vlm_calls_made"] == 0
    assert second_telemetry["estimated_token_savings_pct"] == 100.0


def test_telemetry_fields_self_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [_make_jpeg(tmp_path / "BV2" / f"f{i}.jpg", fill=(i * 8, 0, 0)) for i in range(4)]
    describer, _, _ = _make_describer(
        tmp_path,
        tiers=[Tier.HIGH, Tier.LOW, Tier.LOW, Tier.HIGH],
        junk=[False, True, False, False],
        monkeypatch=monkeypatch,
    )

    asyncio.run(describer.describe_all(frames))
    t = describer.last_telemetry

    assert t["total_frames"] == 4
    assert t["high_tier"] == 2
    assert t["low_tier"] == 2
    assert t["high_tier"] + t["low_tier"] == t["total_frames"]
    assert t["junk_filtered"] == 1
    assert t["vlm_calls_high"] == 2
    assert t["vlm_calls_low"] == 2
    assert t["vlm_calls_made"] == 4
    assert t["cache_hits"] == 0
    # estimated_token_savings_pct = 1 - (2*1.0 + 2*0.4) / 4 = 30%
    assert t["estimated_token_savings_pct"] == pytest.approx(30.0, abs=0.01)


def test_partial_cache_hit_only_calls_for_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [_make_jpeg(tmp_path / "BV3" / f"f{i}.jpg", fill=(i * 9, 0, 0)) for i in range(3)]
    describer, client, cache = _make_describer(
        tmp_path,
        tiers=[Tier.HIGH, Tier.HIGH, Tier.HIGH],
        monkeypatch=monkeypatch,
    )

    # Pre-populate cache for frame[1] only.
    cache.batch_insert(
        [
            (
                sha256_file(frames[1].path),
                describer._model,
                "high",
                {
                    "caption": "preloaded",
                    "ocr_text": "preloaded ocr",
                    "visual_type": "slide_text",
                    "importance_score": 0.5,
                    "ocr_density": 0.3,
                    "novelty_score": 0.2,
                    "why_useful": "x",
                },
            )
        ]
    )

    descs = asyncio.run(describer.describe_all(frames))
    # Only frames 0 and 2 needed VLM calls.
    assert len(client.calls) == 2
    t = describer.last_telemetry
    assert t["cache_hits"] == 1
    assert t["vlm_calls_made"] == 2
    # The preloaded payload made it back into the FrameDescription.
    assert descs[1].caption == "preloaded"
    assert descs[1].ocr_text == "preloaded ocr"


def test_legacy_bv_json_lazy_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [_make_jpeg(tmp_path / "BVLEG" / f"f{i}.jpg", fill=(i * 11, 0, 0)) for i in range(2)]

    describer, client, cache = _make_describer(
        tmp_path,
        tiers=[Tier.HIGH, Tier.HIGH],
        monkeypatch=monkeypatch,
    )

    # Build a legacy per-BV JSON at the path FrameDescriber expects.
    legacy_dir = tmp_path / "vlm_cache" / describer._model.replace(".", ".")
    # The model name might contain dots (e.g., "qwen3.5-omni-plus");
    # FrameDescriber._legacy_bv_json_path sanitises only forbidden chars.
    import re

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", describer._model)
    legacy_dir = tmp_path / "vlm_cache" / safe_model
    legacy_dir.mkdir(parents=True)
    legacy_payload = [
        {
            "timestamp": frames[i].timestamp,
            "path": str(frames[i].path),
            "caption": f"legacy caption {i}",
            "ocr_text": f"legacy ocr {i}",
            "visual_type": "slide",
            "importance_score": 0.7,
            "ocr_density": 0.6,
            "novelty_score": 0.4,
            "why_useful": "from legacy json",
        }
        for i in range(2)
    ]
    (legacy_dir / "BVLEG.json").write_text(
        json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8"
    )

    descs = asyncio.run(describer.describe_all(frames))

    # Lazy migration → no VLM calls (both HIGH-tier rows came from legacy).
    assert len(client.calls) == 0
    assert descs[0].caption == "legacy caption 0"
    assert descs[1].caption == "legacy caption 1"
    t = describer.last_telemetry
    assert t["cache_hits"] == 2
    assert t["vlm_calls_made"] == 0


def test_low_tier_response_falls_back_to_freeform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the model ignores the JSON contract, LOW tier still captures
    the text as ocr_text instead of erroring out."""
    frames = [_make_jpeg(tmp_path / "BVFF" / "f0.jpg")]

    def _bad_responder(kwargs: dict[str, Any]) -> _StubResponse:
        return _StubResponse("just plain text without json wrapper")

    describer, _, _ = _make_describer(
        tmp_path,
        tiers=[Tier.LOW],
        monkeypatch=monkeypatch,
        responder=_bad_responder,
    )
    descs = asyncio.run(describer.describe_all(frames))
    assert descs[0].ocr_text.startswith("just plain text")
    assert descs[0].caption == ""


def test_real_ranker_with_cache_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke with real KeyframeRanker + cache disabled.

    Guards against regressions in the wiring (settings → ranker
    construction → tier dispatch) that the stubbed-ranker tests would
    miss.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_CACHE_ENABLED", "false")
    monkeypatch.setenv("VLM_CACHE_PATH", str(tmp_path / "v.sqlite"))
    from app import config

    config.get_settings.cache_clear()

    frames = [_make_jpeg(tmp_path / "BV4" / f"f{i}.jpg", fill=(i * 30, 0, 0)) for i in range(2)]
    client = _StubClient()

    describer = FrameDescriber(
        ranker=KeyframeRanker(high_floor_ratio=1.0),  # all survivors -> HIGH
        cache=VLMCache(tmp_path / "v.sqlite", enabled=False),
        client=client,
    )

    descs = asyncio.run(describer.describe_all(frames))
    assert len(descs) == 2
    assert describer.last_telemetry["total_frames"] == 2
    assert describer.last_telemetry["cache_enabled"] is False
