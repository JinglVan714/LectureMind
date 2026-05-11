"""Unit tests for :mod:`app.understand.ir_map_reduce` (M2 P4).

The 10 cases follow the impl plan §4.2 step 1 list verbatim. They
exercise :class:`MapReduceIRBuilder` against an in-process stub
``AsyncOpenAI`` client (no network) and a real :class:`ChapterCache`
on ``tmp_path``. The chosen profile is ``long`` (mode='structural',
``use_map_reduce=True``) so the builder runs end-to-end.

Tested invariants:

1. Output chapters are ordered by ``start_sec`` even if the
   ``chapter_plan`` arrives shuffled.
2. A pre-populated ``ChapterCache`` row short-circuits the LLM call
   for that chapter (``map_calls`` does not increment).
3. Successful map calls write the chapter payload back into the cache
   under the ``compute_prompt_hash`` digest.
4. Two consecutive LLM failures collapse into a placeholder chapter
   (``unverified=True``, generated title) instead of crashing.
5. ``MapReduceStats.map_failures`` records the indices of placeholder
   chapters (and only those).
6. Placeholder chapters must not be cached — re-running would bypass
   the retry logic and silently ship a bad chapter.
7. ``reduce_local`` deduplicates ``knowledge_units`` by term, keeping
   the highest-confidence definition; both candidates are recorded
   in the conflict digest passed to reduce-global.
8. ``reduce_global_pass`` only overwrites the four top-level fields
   (``final_synthesis`` / ``mainline`` / ``knowledge_units`` /
   ``cross_references``); chapter internals come straight from map.
9. Two consecutive reduce-global failures raise ``ReduceGlobalError``
   (fail-fast — spec §3.3.3 forbids a degraded fallback).
10. When reduce-global raises, the per-chapter cache rows from the
    map stage stay intact so a re-run pays only the global cost.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from app.understand.chapter_cache import ChapterCache, compute_prompt_hash
from app.understand.chapter_planner import ChapterAnchor
from app.understand.ir_map_reduce import (
    MapReduceIRBuilder,
    MapReduceStats,
    ReduceGlobalError,
)
from app.understand.profile import _PROFILE_TABLE


# ---------- duck-typed inputs ----------------------------------------


@dataclass
class _Seg:
    start: float
    end: float
    text: str


@dataclass
class _Frm:
    timestamp: float
    ocr_text: str = ""
    caption: str = ""
    visual_type: str = "slide_text"
    path: str = ""


@dataclass
class _Meta:
    bv_id: str = "BVTEST"
    title: str = "Sample lecture"
    author: str = "U"
    duration: float = 1800.0


# ---------- async OpenAI stub ----------------------------------------


class _StubMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMsg(content)


class _StubResp:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]
        self.usage = None


class _StubCompletions:
    def __init__(self, parent: "_StubClient") -> None:
        self._p = parent

    async def create(self, **kwargs: Any) -> _StubResp:
        msgs = kwargs.get("messages", [])
        sys = ""
        user = ""
        for m in msgs:
            if m.get("role") == "system":
                sys = m.get("content", "")
            elif m.get("role") == "user":
                user = m.get("content", "")
        kind = "global" if "REDUCE_GLOBAL" in sys or "讲义全局编辑" in sys else "map"
        self._p.calls.append({"kind": kind, "user": user, "kwargs": kwargs})
        if kind == "global":
            result = self._p.global_responder(user, kwargs, self._p)
        else:
            result = self._p.map_responder(user, kwargs, self._p)
        if isinstance(result, BaseException):
            raise result
        return _StubResp(json.dumps(result, ensure_ascii=False))


class _StubChat:
    def __init__(self, parent: "_StubClient") -> None:
        self.completions = _StubCompletions(parent)


class _StubClient:
    def __init__(
        self,
        *,
        map_responder: Callable[..., Any] | None = None,
        global_responder: Callable[..., Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.map_responder = map_responder or _default_map
        self.global_responder = global_responder or _default_global
        self.chat = _StubChat(self)

    @property
    def map_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["kind"] == "map"]

    @property
    def global_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["kind"] == "global"]


def _chapter_index_from_user(user: str) -> int:
    """Extract ``chapter_index`` from a map user prompt for the stub.

    The map user template renders ``章节序号: 第 N 章`` so we just regex
    the integer out. Returns 1 on parse failure (defensive).
    """
    import re

    m = re.search(r"第\s*(\d+)\s*章", user)
    return int(m.group(1)) if m else 1


def _default_map(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
    idx = _chapter_index_from_user(user)
    return {
        "title": f"章节 {idx}",
        "summary": f"summary {idx}",
        "learning_goal": f"goal {idx}",
        "teaching_notes": [f"note {idx}-1"],
        "process_steps": [],
        "points": [
            {"text": f"P{idx}", "ts": 0.0, "quote": f"q-{idx}"}
        ],
        "code_blocks": [],
        "formula_blocks": [],
        "pitfalls": [],
        "key_takeaways": [],
        "knowledge_units": [],
    }


def _default_global(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
    return {
        "lecture_summary": "global lecture summary",
        "mainline": [
            {"step": 1, "title": "step1", "ts": 0, "chapter_index": 1},
            {"step": 2, "title": "step2", "ts": 600, "chapter_index": 2},
        ],
        "glossary_resolved": [],
        "cross_references": [],
    }


# ---------- minimal Settings stub ------------------------------------


@dataclass
class _StubSettings:
    qwen_text_model: str = "stub-model"
    deepseek_request_timeout: float = 30.0
    lecture_map_prompt_version: str = "m2-map-v1"
    lecture_map_chapter_max_retries: int = 1
    lecture_map_chapter_timeout: float = 30.0
    lecture_reduce_global_timeout: float = 30.0
    lecture_reduce_global_max_retries: int = 1
    lecture_reduce_global_max_tokens: int = 4000
    keyframe_min: int = 8
    keyframe_max: int = 20

    def text_extra_body(self) -> dict[str, Any]:
        return {}


def _anchor(start: float, end: float, *, text: str = "anchor", source: str = "equal_split") -> ChapterAnchor:
    return ChapterAnchor(
        start_sec=start,
        end_sec=end,
        confidence=1.0,
        anchor_text=text,
        source=source,
        mode="structural",
    )


def _segments_full() -> list[_Seg]:
    return [
        _Seg(0.0, 5.0, "intro one"),
        _Seg(5.0, 10.0, "intro two"),
        _Seg(610.0, 615.0, "body one"),
        _Seg(615.0, 620.0, "body two"),
    ]


def _frames_full() -> list[_Frm]:
    return [
        _Frm(timestamp=2.5, caption="cap-A", ocr_text="ocrA"),
        _Frm(timestamp=612.0, caption="cap-B", ocr_text="ocrB"),
    ]


def _builder(
    *,
    client: _StubClient,
    cache: ChapterCache,
    profile_name: str = "long",
    duration_sec: float = 1800.0,
    settings: _StubSettings | None = None,
) -> MapReduceIRBuilder:
    profile = _PROFILE_TABLE[profile_name].with_duration(duration_sec)
    return MapReduceIRBuilder(
        client=client,
        settings=settings or _StubSettings(),
        profile=profile,
        chapter_cache=cache,
    )


# ---------- tests -----------------------------------------------------


def test_map_reduce_orders_chapters_by_start_sec(tmp_path: Path) -> None:
    """Anchors arrive shuffled — output must still be sorted by start_sec."""
    client = _StubClient()
    cache = ChapterCache(tmp_path / "cc.sqlite")
    builder = _builder(client=client, cache=cache)
    # Shuffle: second chapter comes first in the input list.
    chapter_plan = [
        _anchor(600.0, 1800.0, text="body"),
        _anchor(0.0, 600.0, text="intro"),
    ]
    ir, stats = asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            study_questions=["Q1?"],
        )
    )
    starts = [ch.start for ch in ir.chapters]
    assert starts == sorted(starts)
    assert [ch.start for ch in ir.chapters] == [0.0, 600.0]
    assert isinstance(stats, MapReduceStats)
    assert stats.map_calls == 2  # both chapters hit LLM (no cache)


def test_map_reduce_uses_chapter_cache_when_hit(tmp_path: Path) -> None:
    """Pre-populated cache row → no LLM call for that chapter."""
    client = _StubClient()
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [_anchor(0.0, 600.0, text="intro"), _anchor(600.0, 1800.0, text="body")]
    settings = _StubSettings()

    # Pre-compute the prompt hash for chapter 1 with the same inputs the
    # builder will use; pre-populate the cache so it short-circuits.
    segs = _segments_full()
    frames = _frames_full()
    chap_segs = [s for s in segs if 0.0 <= s.start < 600.0]
    chap_frames = [f for f in frames if 0.0 <= f.timestamp < 600.0]
    cached_payload = {
        "title": "cached intro",
        "summary": "cached summary",
        "learning_goal": "cached goal",
        "teaching_notes": ["cached note"],
        "process_steps": [],
        "points": [{"text": "cached point", "ts": 0.0, "quote": "cached quote"}],
        "code_blocks": [],
        "formula_blocks": [],
        "pitfalls": [],
        "key_takeaways": [],
        "knowledge_units": [],
    }
    h = compute_prompt_hash(
        chapter_start_sec=0.0,
        chapter_end_sec=600.0,
        chapter_segments=chap_segs,
        chapter_frames=chap_frames,
        model_id=settings.qwen_text_model,
        prompt_version=settings.lecture_map_prompt_version,
    )
    cache.put(
        prompt_hash=h,
        chapter_payload=cached_payload,
        model_id=settings.qwen_text_model,
        prompt_version=settings.lecture_map_prompt_version,
        bv_id="BVTEST",
        chapter_index=1,
        chapter_start=0.0,
        chapter_end=600.0,
    )

    builder = _builder(client=client, cache=cache, settings=settings)
    ir, stats = asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=segs,
            frames=frames,
            meta=_Meta(),
            study_questions=["Q1?"],
        )
    )

    # Chapter 1 must come from the cache: title is "cached intro".
    assert ir.chapters[0].title == "cached intro"
    # Only chapter 2 should have hit the LLM.
    assert len(client.map_calls) == 1
    assert stats.map_calls == 1
    assert stats.cache_hits == 1


def test_map_reduce_writes_chapter_cache_after_success(tmp_path: Path) -> None:
    """A successful map call must persist to the chapter cache."""
    client = _StubClient()
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [_anchor(0.0, 1800.0, text="solo")]
    builder = _builder(client=client, cache=cache)
    ir, stats = asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            study_questions=[],
        )
    )
    assert len(ir.chapters) == 1
    assert stats.map_calls == 1
    assert stats.cache_writes == 1
    assert cache.stats()["rows_total"] == 1


def test_map_reduce_inserts_placeholder_after_max_retries(tmp_path: Path) -> None:
    """Two consecutive map failures → placeholder chapter (no crash)."""

    def _broken_map(user: str, kwargs: dict[str, Any], client: _StubClient) -> Any:
        raise RuntimeError("boom")

    client = _StubClient(map_responder=_broken_map)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [
        _anchor(0.0, 600.0, text="intro"),
        _anchor(600.0, 1800.0, text="body"),
    ]
    builder = _builder(client=client, cache=cache)
    ir, stats = asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            study_questions=[],
        )
    )
    assert len(ir.chapters) == 2
    # Both chapters are placeholders (failure title from spec §3.3.1).
    for ch in ir.chapters:
        assert "生成失败" in ch.title
    # Two attempts per chapter (1 + max_retries) × 2 chapters = 4 calls.
    assert len(client.map_calls) == 4


def test_map_reduce_records_map_failures_in_stats(tmp_path: Path) -> None:
    """Placeholder chapter indexes appear in ``stats.map_failures``.

    Only the chapter that fell back is recorded — successful ones must
    not bleed into the failures list.
    """
    state = {"calls": 0}

    def _half_broken(user: str, kwargs: dict[str, Any], client: _StubClient) -> Any:
        state["calls"] += 1
        idx = _chapter_index_from_user(user)
        # Fail every attempt for chapter 2 only; chapter 1 succeeds.
        if idx == 2:
            raise RuntimeError("boom")
        return _default_map(user, kwargs, client)

    client = _StubClient(map_responder=_half_broken)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [
        _anchor(0.0, 600.0, text="intro"),
        _anchor(600.0, 1800.0, text="body"),
    ]
    builder = _builder(client=client, cache=cache)
    _, stats = asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            study_questions=[],
        )
    )
    assert stats.map_failures == (2,)


def test_map_reduce_does_not_cache_placeholder(tmp_path: Path) -> None:
    """Placeholder chapters must not be written to the cache.

    Regression guard: a previous implementation rounded the cache write
    inside the ``finally`` block of ``_map_chapter_with_retry``, which
    would happily memoise the placeholder so subsequent runs short-
    circuit on a known-bad payload.
    """

    def _broken_map(user: str, kwargs: dict[str, Any], client: _StubClient) -> Any:
        raise RuntimeError("boom")

    client = _StubClient(map_responder=_broken_map)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [_anchor(0.0, 1800.0, text="solo")]
    builder = _builder(client=client, cache=cache)
    asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            study_questions=[],
        )
    )
    assert cache.stats()["rows_total"] == 0


def test_reduce_local_dedups_glossary_by_confidence(tmp_path: Path) -> None:
    """When two chapters share a knowledge-unit term, the higher
    confidence definition wins; the loser is recorded in the
    glossary-conflict digest passed to the reduce-global stage.
    """

    def _kunit_map(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
        idx = _chapter_index_from_user(user)
        if idx == 1:
            return {
                "title": "Intro",
                "summary": "intro",
                "learning_goal": "g",
                "teaching_notes": [],
                "process_steps": [],
                "points": [{"text": "P1", "ts": 0.0, "quote": "q"}],
                "code_blocks": [],
                "formula_blocks": [],
                "pitfalls": [],
                "key_takeaways": [],
                "knowledge_units": [
                    {"term": "Attention", "definition": "weak intro definition", "confidence": 0.4, "ts": 0.0}
                ],
            }
        return {
            "title": "Body",
            "summary": "body",
            "learning_goal": "g",
            "teaching_notes": [],
            "process_steps": [],
            "points": [{"text": "P2", "ts": 600.0, "quote": "q"}],
            "code_blocks": [],
            "formula_blocks": [],
            "pitfalls": [],
            "key_takeaways": [],
            "knowledge_units": [
                {"term": "Attention", "definition": "strong body definition", "confidence": 0.9, "ts": 600.0}
            ],
        }

    client = _StubClient(map_responder=_kunit_map)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [_anchor(0.0, 600.0), _anchor(600.0, 1800.0)]
    builder = _builder(client=client, cache=cache)

    chapter_dicts, conflicts = asyncio.run(
        builder._collect_chapter_dicts(  # type: ignore[attr-defined]
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            stats={
                "map_calls": 0,
                "cache_hits": 0,
                "cache_writes": 0,
                "map_failures": [],
            },
        )
    )
    local_ir, glossary_conflicts = builder.reduce_local(
        chapter_dicts=chapter_dicts,
        meta=_Meta(),
    )
    units = local_ir["knowledge_units"]
    assert len(units) == 1
    winner = units[0]
    assert winner["title"] == "Attention"
    assert winner["explanation"] == "strong body definition"
    # Conflict digest carries both candidates with their chapter ids.
    assert any(c.get("definition") == "weak intro definition" for c in glossary_conflicts)
    assert any(c.get("definition") == "strong body definition" for c in glossary_conflicts)


def test_reduce_global_pass_only_overwrites_top_level_fields(tmp_path: Path) -> None:
    """reduce-global writes ``final_synthesis`` / ``mainline`` /
    ``knowledge_units`` / ``cross_references`` only — chapter
    internals (points / code / formulas / summary / title) come
    straight from the map output.
    """

    def _rich_map(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
        idx = _chapter_index_from_user(user)
        return {
            "title": f"map title {idx}",
            "summary": f"map summary {idx}",
            "learning_goal": f"goal {idx}",
            "teaching_notes": [f"note {idx}"],
            "process_steps": [],
            "points": [{"text": f"map point {idx}", "ts": float(idx), "quote": f"q{idx}"}],
            "code_blocks": [
                {"language": "python", "code": f"x = {idx}", "ts": float(idx), "explanation": "e", "source": "ocr"}
            ],
            "formula_blocks": [],
            "pitfalls": [f"pitfall {idx}"],
            "key_takeaways": [f"take {idx}"],
            "knowledge_units": [
                {"term": f"Term{idx}", "definition": f"def{idx}", "confidence": 0.7, "ts": float(idx)}
            ],
        }

    def _rich_global(user: str, kwargs: dict[str, Any], client: _StubClient) -> dict[str, Any]:
        return {
            "lecture_summary": "GLOBAL final synthesis",
            "mainline": [
                {"step": 1, "title": "GLOBAL step", "ts": 0, "chapter_index": 1}
            ],
            "glossary_resolved": [
                {"term": "Term1", "definition": "GLOBAL def1", "winning_chapter_index": 1},
                {"term": "Term2", "definition": "GLOBAL def2", "winning_chapter_index": 2},
            ],
            "cross_references": [
                {"from_chapter": 1, "to_chapter": 2, "relation": "depends_on"}
            ],
        }

    client = _StubClient(map_responder=_rich_map, global_responder=_rich_global)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [_anchor(0.0, 600.0), _anchor(600.0, 1800.0)]
    builder = _builder(client=client, cache=cache)
    ir, _ = asyncio.run(
        builder.build(
            chapter_plan=chapter_plan,
            segments=_segments_full(),
            frames=_frames_full(),
            meta=_Meta(),
            study_questions=[],
        )
    )
    # Top-level overwrites applied:
    assert ir.final_synthesis == "GLOBAL final synthesis"
    assert "GLOBAL step" in " ".join(ir.mainline)
    titles = {u.title for u in ir.knowledge_units}
    assert titles == {"Term1", "Term2"}
    assert any(u.explanation == "GLOBAL def1" for u in ir.knowledge_units)
    # Chapter internals untouched: titles still come from map.
    assert [ch.title for ch in ir.chapters] == ["map title 1", "map title 2"]
    assert ir.chapters[0].summary == "map summary 1"
    assert ir.chapters[0].points[0].text == "map point 1"
    assert ir.chapters[0].points[0].quote == "q1"
    # Code blocks stayed under the originating chapter, not lifted to lecture-level.
    assert ir.chapters[0].code_blocks
    assert ir.chapters[0].code_blocks[0].code == "x = 1"


def test_reduce_global_pass_raises_on_timeout_after_retries(tmp_path: Path) -> None:
    """Two consecutive timeouts on the global pass → ReduceGlobalError.

    Spec §3.3.3 says the pipeline must fail-fast rather than degrade.
    """
    state = {"calls": 0}

    def _timeout_global(user: str, kwargs: dict[str, Any], client: _StubClient) -> Any:
        state["calls"] += 1
        return asyncio.TimeoutError("simulated timeout")

    client = _StubClient(global_responder=_timeout_global)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [_anchor(0.0, 1800.0)]
    builder = _builder(client=client, cache=cache)

    with pytest.raises(ReduceGlobalError):
        asyncio.run(
            builder.build(
                chapter_plan=chapter_plan,
                segments=_segments_full(),
                frames=_frames_full(),
                meta=_Meta(),
                study_questions=[],
            )
        )
    # Initial attempt + 1 retry == 2 calls.
    assert state["calls"] == 2


def test_reduce_global_failure_does_not_invalidate_chapter_cache(tmp_path: Path) -> None:
    """When reduce-global fails the per-chapter cache must survive.

    Re-running an epic video should pay only the global pass cost
    again; map output stays cached because the chapter-level work was
    successful.
    """

    def _timeout_global(user: str, kwargs: dict[str, Any], client: _StubClient) -> Any:
        return asyncio.TimeoutError("simulated timeout")

    client = _StubClient(global_responder=_timeout_global)
    cache = ChapterCache(tmp_path / "cc.sqlite")
    chapter_plan = [
        _anchor(0.0, 600.0, text="intro"),
        _anchor(600.0, 1800.0, text="body"),
    ]
    builder = _builder(client=client, cache=cache)

    with pytest.raises(ReduceGlobalError):
        asyncio.run(
            builder.build(
                chapter_plan=chapter_plan,
                segments=_segments_full(),
                frames=_frames_full(),
                meta=_Meta(),
                study_questions=[],
            )
        )
    # The two map calls succeeded → both rows are persisted.
    assert cache.stats()["rows_total"] == 2
