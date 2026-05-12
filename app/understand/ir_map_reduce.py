"""Map-reduce LectureIR builder for the M2 long / epic profiles (P4).

The single-shot :class:`~app.understand.ir_builder.LectureIRBuilder`
fits a 30-minute video into one prompt, but a 60+ minute lecture blows
through any model's context window. This module implements the
deterministic-router contract from spec §3.3 — given an
already-computed :class:`~app.understand.chapter_planner.ChapterAnchor`
plan, it issues one LLM call per chapter, stitches the results
together with a pure-Python ``reduce_local`` pass, then runs a single
lightweight ``reduce_global_pass`` LLM call to fill the four
lecture-level fields (``final_synthesis`` / ``mainline`` /
``knowledge_units`` / ``cross_references``) the per-chapter calls were
forbidden to touch.

Three invariants drive the design:

* **Cacheable**: every map call is keyed by
  :func:`compute_prompt_hash`, so re-running a video re-uses the map
  output and only the global pass re-pays its 30-s cost.
* **Resilient**: a chapter that fails after ``LECTURE_MAP_CHAPTER_MAX_RETRIES``
  drops a deterministic placeholder (``unverified=True``) instead of
  crashing the pipeline; the placeholder is **not** cached so the
  next run gets to retry it.
* **Fail-fast on global**: the global pass must succeed — when it
  fails after retries, ``ReduceGlobalError`` propagates and the
  pipeline aborts. The chapter cache stays intact (rationale in spec
  §3.3.3) so a re-run only repays the global cost.

P4 ships the builder primitives only; pipeline integration (router
selection, telemetry, ``ir_builder.build()`` dispatch) lands in P7.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from .chapter_cache import ChapterCache, compute_prompt_hash
from .chapter_planner import ChapterAnchor
from .ir import LectureIR
from .prompts import (
    LECTURE_IR_MAP_CHAPTER_SYSTEM,
    LECTURE_IR_MAP_CHAPTER_USER_TEMPLATE,
    LECTURE_IR_REDUCE_GLOBAL_SYSTEM,
    LECTURE_IR_REDUCE_GLOBAL_USER_TEMPLATE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .profile import LectureProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses & exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapReduceStats:
    """Per-build telemetry surface.

    All counters reflect a single ``build()`` invocation. P7 forwards
    them under ``pipeline_stats.map_reduce`` for the matrix tooling.
    """

    map_calls: int = 0
    cache_hits: int = 0
    cache_writes: int = 0
    map_failures: tuple[int, ...] = ()
    map_total_sec: float = 0.0
    reduce_local_sec: float = 0.0
    reduce_global_sec: float = 0.0
    reduce_global_attempts: int = 0


class ReduceGlobalError(RuntimeError):
    """Raised when the reduce-global LLM call fails after every retry.

    Spec §3.3.3 mandates fail-fast over a degraded fallback — the
    chapter cache survives so a re-run only repays the global cost
    (~30s on epic videos).
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_PLACEHOLDER_KEY = "_lm_placeholder"


def _segments_in_window(segments: Sequence[Any], start: float, end: float) -> list[Any]:
    """Slice subtitle segments whose ``start`` falls in ``[start, end]``.

    Inclusive on both sides: a segment whose ``start == end`` (i.e.
    coincident with the chapter boundary) belongs to the trailing
    chapter — that matches how :class:`ChapterPlanner` materialises
    half-open chapters but still attaches edge-of-window text.
    """
    out: list[Any] = []
    for s in segments:
        ts = float(getattr(s, "start", getattr(s, "start_sec", 0.0)) or 0.0)
        if start <= ts <= end:
            out.append(s)
    return out


def _frames_in_window(frames: Sequence[Any], start: float, end: float) -> list[Any]:
    out: list[Any] = []
    for f in frames:
        ts = float(getattr(f, "timestamp", getattr(f, "ts", 0.0)) or 0.0)
        if start <= ts <= end:
            out.append(f)
    return out


def _format_segments(segments: Iterable[Any]) -> str:
    rows: list[str] = []
    for s in segments:
        ts = float(getattr(s, "start", getattr(s, "start_sec", 0.0)) or 0.0)
        text = str(getattr(s, "text", "") or "")
        rows.append(f"[ts={ts:.1f}s | {int(ts // 60):02d}:{int(ts % 60):02d}] {text}")
    return "\n".join(rows)


def _format_frames(frames: Iterable[Any]) -> str:
    rows: list[str] = []
    for f in frames:
        ts = float(getattr(f, "timestamp", getattr(f, "ts", 0.0)) or 0.0)
        caption = str(getattr(f, "caption", "") or "")
        ocr = str(getattr(f, "ocr_text", getattr(f, "ocr", "")) or "")
        path = str(getattr(f, "path", "") or "")
        line = f"[ts={ts:.1f}s | {int(ts // 60):02d}:{int(ts % 60):02d}] path={path} · {caption}"
        if ocr:
            line += f"  · OCR: {ocr[:200]}"
        rows.append(line)
    return "\n".join(rows)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty response")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(raw)
        if not m:
            raise ValueError("no JSON object found in response")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class MapReduceIRBuilder:
    """Long / epic profile chapter map-reduce LectureIR builder.

    Stateless across ``build()`` calls — the only mutable state is on
    the injected ``chapter_cache``. The constructor accepts a duck-typed
    ``client`` (anything exposing ``client.chat.completions.create(...)``)
    so production callers pass an :class:`openai.AsyncOpenAI` while
    tests inject the local stub from ``test_ir_map_reduce``.
    """

    def __init__(
        self,
        *,
        client: Any,
        settings: Any,
        profile: "LectureProfile",
        chapter_cache: ChapterCache,
    ) -> None:
        self._client = client
        self._settings = settings
        self._profile = profile
        self._cache = chapter_cache
        self._model = getattr(settings, "qwen_text_model", "")
        self._extra_body = (
            settings.text_extra_body() if hasattr(settings, "text_extra_body") else {}
        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    async def build(
        self,
        *,
        chapter_plan: Sequence[ChapterAnchor],
        segments: Sequence[Any],
        frames: Sequence[Any],
        meta: Any,
        study_questions: Sequence[str] | None = None,
    ) -> tuple[LectureIR, MapReduceStats]:
        """Run map → reduce-local → reduce-global and return ``(LectureIR, stats)``.

        Raises :class:`ReduceGlobalError` when the global pass fails
        after retries.
        """
        stats: dict[str, Any] = {
            "map_calls": 0,
            "cache_hits": 0,
            "cache_writes": 0,
            "map_failures": [],
            "map_total_sec": 0.0,
            "reduce_local_sec": 0.0,
            "reduce_global_sec": 0.0,
            "reduce_global_attempts": 0,
        }

        chapter_dicts, _ = await self._collect_chapter_dicts(
            chapter_plan=chapter_plan,
            segments=segments,
            frames=frames,
            meta=meta,
            stats=stats,
        )

        t_local = time.perf_counter()
        local_ir, glossary_conflicts = self.reduce_local(
            chapter_dicts=chapter_dicts,
            meta=meta,
        )
        stats["reduce_local_sec"] = time.perf_counter() - t_local

        local_ir = await self.reduce_global_pass(
            local_ir=local_ir,
            glossary_conflicts=glossary_conflicts,
            meta=meta,
            study_questions=list(study_questions or []),
            stats=stats,
        )

        # Hand-built dict goes through ``LectureIR.model_validate`` so
        # downstream renderers (which type against the pydantic model)
        # see the exact same shape they get from the single-shot path.
        local_ir["bv_id"] = getattr(meta, "bv_id", local_ir.get("bv_id", ""))
        local_ir["url"] = getattr(meta, "url", local_ir.get("url", ""))
        local_ir["title"] = getattr(meta, "title", local_ir.get("title", ""))
        local_ir["author"] = getattr(meta, "author", local_ir.get("author", ""))
        local_ir["duration"] = float(getattr(meta, "duration", local_ir.get("duration", 0.0)) or 0.0)
        local_ir["cover"] = getattr(meta, "cover_url", local_ir.get("cover", ""))
        local_ir["study_questions"] = list(study_questions or [])

        ir = LectureIR.model_validate(local_ir)
        return ir, MapReduceStats(
            map_calls=int(stats["map_calls"]),
            cache_hits=int(stats["cache_hits"]),
            cache_writes=int(stats["cache_writes"]),
            map_failures=tuple(stats["map_failures"]),
            map_total_sec=round(float(stats["map_total_sec"]), 3),
            reduce_local_sec=round(float(stats["reduce_local_sec"]), 3),
            reduce_global_sec=round(float(stats["reduce_global_sec"]), 3),
            reduce_global_attempts=int(stats["reduce_global_attempts"]),
        )

    # ------------------------------------------------------------------
    # Map stage
    # ------------------------------------------------------------------

    async def _collect_chapter_dicts(
        self,
        *,
        chapter_plan: Sequence[ChapterAnchor],
        segments: Sequence[Any],
        frames: Sequence[Any],
        meta: Any,
        stats: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Issue map calls (or hit cache) for each anchor.

        Anchors are processed in **start_sec order** so the resulting
        chapter list is monotonically increasing regardless of how the
        caller ordered ``chapter_plan``. The second return value is
        kept as a forward-compat hook for future per-chapter conflict
        digests; current code only populates it inside ``reduce_local``.
        """
        # Snapshot + sort: we don't mutate the caller's list.
        ordered = sorted(
            list(chapter_plan), key=lambda a: float(getattr(a, "start_sec", 0.0))
        )
        out: list[dict[str, Any]] = []
        for idx0, anchor in enumerate(ordered):
            chapter_index = idx0 + 1
            start = float(anchor.start_sec)
            end = float(anchor.end_sec)
            chap_segs = _segments_in_window(segments, start, end)
            chap_frames = _frames_in_window(frames, start, end)
            prompt_hash = compute_prompt_hash(
                chapter_start_sec=start,
                chapter_end_sec=end,
                chapter_segments=chap_segs,
                chapter_frames=chap_frames,
                model_id=self._model,
                prompt_version=getattr(self._settings, "lecture_map_prompt_version", "m2-map-v1"),
            )

            cached = self._cache.get(prompt_hash) if self._cache.enabled else None
            if cached is not None:
                stats["cache_hits"] = int(stats["cache_hits"]) + 1
                chapter_dict = self._stamp_chapter(
                    cached, anchor=anchor, chapter_index=chapter_index
                )
                out.append(chapter_dict)
                continue

            t0 = time.perf_counter()
            payload, is_placeholder = await self._map_chapter_with_retry(
                anchor=anchor,
                chapter_index=chapter_index,
                chap_segs=chap_segs,
                chap_frames=chap_frames,
                meta=meta,
                stats=stats,
            )
            stats["map_total_sec"] = float(stats.get("map_total_sec", 0.0)) + (
                time.perf_counter() - t0
            )
            chapter_dict = self._stamp_chapter(
                payload, anchor=anchor, chapter_index=chapter_index
            )
            if is_placeholder:
                stats["map_failures"].append(chapter_index)
            else:
                # Spec §3.3.1: only cache verified chapter outputs so a
                # placeholder never short-circuits a future re-run.
                if self._cache.enabled:
                    self._cache.put(
                        prompt_hash=prompt_hash,
                        chapter_payload=payload,
                        model_id=self._model,
                        prompt_version=getattr(self._settings, "lecture_map_prompt_version", "m2-map-v1"),
                        bv_id=getattr(meta, "bv_id", ""),
                        chapter_index=chapter_index,
                        chapter_start=start,
                        chapter_end=end,
                    )
                    stats["cache_writes"] = int(stats["cache_writes"]) + 1
            out.append(chapter_dict)
        return out, []

    async def _map_chapter_with_retry(
        self,
        *,
        anchor: ChapterAnchor,
        chapter_index: int,
        chap_segs: list[Any],
        chap_frames: list[Any],
        meta: Any,
        stats: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Try the map call up to ``max_retries + 1`` times.

        Returns ``(payload, is_placeholder)``. The placeholder payload
        carries the spec §3.3.1 "本章生成失败" title and an empty
        ``points`` list so downstream renderers degrade gracefully.
        """
        max_retries = int(getattr(self._settings, "lecture_map_chapter_max_retries", 1) or 0)
        timeout = float(getattr(self._settings, "lecture_map_chapter_timeout", 180.0) or 180.0)
        user = self._format_map_user(
            anchor=anchor,
            chapter_index=chapter_index,
            chap_segs=chap_segs,
            chap_frames=chap_frames,
            meta=meta,
        )
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            stats["map_calls"] = int(stats["map_calls"]) + 1
            try:
                content, _finish_reason = await self._call_llm(
                    system=LECTURE_IR_MAP_CHAPTER_SYSTEM,
                    user=user,
                    timeout=timeout,
                )
                payload = _extract_json(content)
                # Drop disallowed lecture-level keys defensively in case
                # the model ignored the system prompt; chapter internals
                # are all that matters here.
                for k in ("mainline", "lecture_summary", "cross_references"):
                    payload.pop(k, None)
                return payload, False
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Map-chapter %d attempt %d/%d failed: %s",
                    chapter_index,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                continue
        # All attempts failed: emit placeholder.
        logger.warning(
            "Map-chapter %d exhausted retries (%s); inserting placeholder",
            chapter_index,
            last_exc,
        )
        return self._placeholder_payload(anchor=anchor, chapter_index=chapter_index), True

    def _placeholder_payload(
        self, *, anchor: ChapterAnchor, chapter_index: int
    ) -> dict[str, Any]:
        return {
            _PLACEHOLDER_KEY: True,
            "title": f"第 {chapter_index} 章（生成失败）",
            "summary": "本章内容生成失败，请手动重跑或检查日志。",
            "learning_goal": "",
            "teaching_notes": [],
            "process_steps": [],
            "points": [],
            "code_blocks": [],
            "formula_blocks": [],
            "pitfalls": [],
            "key_takeaways": [],
            "knowledge_units": [],
        }

    def _stamp_chapter(
        self,
        payload: dict[str, Any],
        *,
        anchor: ChapterAnchor,
        chapter_index: int,
    ) -> dict[str, Any]:
        """Attach ``index`` / ``start`` / ``end`` from the anchor.

        Map output is intentionally chapter-window-agnostic so that
        the cache key (which already encodes the time window) can be
        re-used across BVs. We re-stamp boundaries here so the IR
        chapter list is consistent with the planner output, even on
        cache hits.
        """
        chapter = dict(payload)
        chapter["index"] = chapter_index
        chapter["start"] = float(anchor.start_sec)
        chapter["end"] = float(anchor.end_sec)
        if not chapter.get("title"):
            chapter["title"] = f"第 {chapter_index} 章"
        if not chapter.get("summary"):
            chapter["summary"] = chapter["title"]
        # ``IRChapter.points`` requires text+ts+quote on every entry; map
        # output already conforms but we tolerate sparse dicts here.
        chapter["points"] = [
            self._coerce_point(p, chapter_start=chapter["start"]) for p in chapter.get("points", [])
            if isinstance(p, dict)
        ]
        return chapter

    @staticmethod
    def _coerce_point(p: dict[str, Any], *, chapter_start: float) -> dict[str, Any]:
        return {
            "text": str(p.get("text") or "").strip(),
            "ts": max(0.0, float(p.get("ts", chapter_start) or chapter_start)),
            "quote": str(p.get("quote") or "").strip(),
        }

    def _format_map_user(
        self,
        *,
        anchor: ChapterAnchor,
        chapter_index: int,
        chap_segs: list[Any],
        chap_frames: list[Any],
        meta: Any,
    ) -> str:
        return LECTURE_IR_MAP_CHAPTER_USER_TEMPLATE.format(
            bv_id=getattr(meta, "bv_id", ""),
            title=getattr(meta, "title", ""),
            duration_sec=int(getattr(meta, "duration", 0) or 0),
            chapter_index=chapter_index,
            chapter_start_sec=int(round(float(anchor.start_sec))),
            chapter_end_sec=int(round(float(anchor.end_sec))),
            anchor_text=str(anchor.anchor_text or "")[:80],
            subtitle_block=_format_segments(chap_segs) or "(无)",
            frames_block=_format_frames(chap_frames) or "(无)",
        )

    # ------------------------------------------------------------------
    # Reduce-local (pure script)
    # ------------------------------------------------------------------

    def reduce_local(
        self,
        *,
        chapter_dicts: list[dict[str, Any]],
        meta: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Stitch chapter dicts into a partial LectureIR.

        Returns ``(local_ir_dict, glossary_conflicts)``. ``local_ir`` has
        every required ``LectureIR`` field except the four owned by
        the global pass (``final_synthesis`` / ``mainline`` /
        ``knowledge_units`` / ``cross_references``) which start with
        sensible placeholders for the global call to overwrite.

        Per spec §3.3.2:
        * ``glossary`` (==``knowledge_units``) deduplicates by term,
          keeping the highest-confidence definition.
        * ``mainline`` is provisionally each chapter summary in
          sequence — overwritten by the global pass.
        * ``code_blocks`` / ``formula_blocks`` stay attached to their
          owning chapter (no lecture-level aggregation).
        """
        chapters: list[dict[str, Any]] = []
        knowledge_units_by_term: dict[str, dict[str, Any]] = {}
        glossary_conflicts: list[dict[str, Any]] = []

        for raw in sorted(chapter_dicts, key=lambda c: float(c.get("start", 0.0))):
            chapter = self._build_chapter_dict(raw)
            chapters.append(chapter)
            for ku in raw.get("knowledge_units") or []:
                if not isinstance(ku, dict):
                    continue
                term = str(ku.get("term") or "").strip()
                if not term:
                    continue
                conf = float(ku.get("confidence") or 0.0)
                definition = str(ku.get("definition") or "").strip()
                ts_val = float(ku.get("ts") or chapter["start"] or 0.0)
                candidate = {
                    "term": term,
                    "definition": definition,
                    "confidence": conf,
                    "chapter_index": chapter["index"],
                    "ts": ts_val,
                }
                # Always record the candidate in the conflict digest so
                # the global pass sees both winners and losers.
                glossary_conflicts.append(candidate)
                existing = knowledge_units_by_term.get(term)
                if existing is None or conf > float(existing.get("confidence", 0.0)):
                    knowledge_units_by_term[term] = candidate

        # Project deduped winners into ``LectureIR.knowledge_units`` shape.
        knowledge_units: list[dict[str, Any]] = []
        for i, candidate in enumerate(knowledge_units_by_term.values(), start=1):
            knowledge_units.append(
                {
                    "id": f"ku-{i}",
                    "type": "concept",
                    "title": candidate["term"],
                    "explanation": candidate["definition"],
                    "ts": candidate["ts"],
                    "quote": "",
                    "chapter_index": candidate["chapter_index"],
                }
            )

        provisional_mainline = [
            (ch.get("summary") or ch.get("title") or "").strip()
            for ch in chapters
            if (ch.get("summary") or ch.get("title"))
        ]
        provisional_synthesis = "（待 reduce-global 生成）"

        local_ir: dict[str, Any] = {
            "bv_id": getattr(meta, "bv_id", ""),
            "title": getattr(meta, "title", ""),
            "author": getattr(meta, "author", ""),
            "duration": float(getattr(meta, "duration", 0.0) or 0.0),
            "core_question": "",
            "mainline": provisional_mainline,
            "knowledge_units": knowledge_units,
            "chapters": chapters,
            "final_synthesis": provisional_synthesis,
            "review_questions": [],
            "study_questions": [],
        }
        return local_ir, glossary_conflicts

    def _build_chapter_dict(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Coerce a stamped chapter into a strict ``IRChapter`` dict."""
        idx = int(raw.get("index") or 1)
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or start)
        if end < start:
            end = start
        chapter = {
            "index": idx,
            "title": str(raw.get("title") or f"第 {idx} 章"),
            "start": start,
            "end": end,
            "summary": str(raw.get("summary") or ""),
            "learning_goal": str(raw.get("learning_goal") or ""),
            "teaching_notes": [
                str(t).strip() for t in (raw.get("teaching_notes") or []) if str(t).strip()
            ],
            "process_steps": [
                str(t).strip() for t in (raw.get("process_steps") or []) if str(t).strip()
            ],
            "points": [
                p for p in (raw.get("points") or []) if isinstance(p, dict)
            ],
            "frames": [],
            "pitfalls": [
                str(t).strip() for t in (raw.get("pitfalls") or []) if str(t).strip()
            ],
            "key_takeaways": [
                str(t).strip() for t in (raw.get("key_takeaways") or []) if str(t).strip()
            ],
            "knowledge_unit_ids": [],
            "code_blocks": [
                cb for cb in (raw.get("code_blocks") or []) if isinstance(cb, dict)
            ],
            "formula_blocks": [
                fb for fb in (raw.get("formula_blocks") or []) if isinstance(fb, dict)
            ],
        }
        return chapter

    # ------------------------------------------------------------------
    # Reduce-global (1 LLM call + retries)
    # ------------------------------------------------------------------

    async def reduce_global_pass(
        self,
        *,
        local_ir: dict[str, Any],
        glossary_conflicts: list[dict[str, Any]],
        meta: Any,
        study_questions: list[str],
        stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the global pass and overwrite the four lecture-level fields.

        Raises :class:`ReduceGlobalError` after ``max_retries+1``
        consecutive failures. Chapter internals in ``local_ir`` are
        **not** touched — that's the whole point of the map / reduce
        split.
        """
        max_retries = int(getattr(self._settings, "lecture_reduce_global_max_retries", 1) or 0)
        timeout = float(getattr(self._settings, "lecture_reduce_global_timeout", 120.0) or 120.0)
        max_tokens = int(getattr(self._settings, "lecture_reduce_global_max_tokens", 0) or 0)
        user = self._format_global_user(
            local_ir=local_ir,
            glossary_conflicts=glossary_conflicts,
            meta=meta,
            study_questions=study_questions,
        )
        last_exc: BaseException | None = None
        t0 = time.perf_counter()
        for attempt in range(max_retries + 1):
            if stats is not None:
                stats["reduce_global_attempts"] = int(stats["reduce_global_attempts"]) + 1
            try:
                content, finish_reason = await self._call_llm(
                    system=LECTURE_IR_REDUCE_GLOBAL_SYSTEM,
                    user=user,
                    timeout=timeout,
                    max_tokens=max_tokens or None,
                )
                try:
                    obj = _extract_json(content)
                except ValueError as parse_exc:
                    # Distinguish backend truncation from prompt-leak /
                    # hallucination so the failure log + final
                    # ReduceGlobalError tell the user where to look.
                    # See ir_builder._parse_to_dict for the same pattern.
                    if finish_reason == "length":
                        raise RuntimeError(
                            "Reduce-global truncated lecture-level JSON "
                            f"(finish_reason=length, raw {len(content)} "
                            f"chars): {parse_exc}. Raise "
                            "LECTURE_REDUCE_GLOBAL_MAX_TOKENS or shorten "
                            "the chapter digest (fewer / shorter chapters)."
                        ) from parse_exc
                    raise
                self._apply_global_pass(local_ir, obj)
                if stats is not None:
                    stats["reduce_global_sec"] = float(stats["reduce_global_sec"]) + (time.perf_counter() - t0)
                return local_ir
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Reduce-global attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                continue
        if stats is not None:
            stats["reduce_global_sec"] = float(stats["reduce_global_sec"]) + (time.perf_counter() - t0)
        raise ReduceGlobalError(
            f"reduce-global failed after {max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def _apply_global_pass(
        self, local_ir: dict[str, Any], obj: dict[str, Any]
    ) -> None:
        """Overwrite the 4 lecture-level fields. Chapters stay frozen.

        ``glossary_resolved`` is projected into ``LectureIR.knowledge_units``
        by replacing the prior ``explanation`` with the global definition
        (when present) while preserving each unit's ``chapter_index``.
        """
        lecture_summary = obj.get("lecture_summary")
        if isinstance(lecture_summary, str) and lecture_summary.strip():
            local_ir["final_synthesis"] = lecture_summary.strip()

        mainline = obj.get("mainline")
        if isinstance(mainline, list) and mainline:
            local_ir["mainline"] = [
                self._format_mainline_step(step) for step in mainline if step
            ]

        glossary = obj.get("glossary_resolved")
        if isinstance(glossary, list):
            local_ir["knowledge_units"] = self._merge_global_glossary(
                local_ir.get("knowledge_units") or [], glossary
            )

        cross_refs = obj.get("cross_references")
        if isinstance(cross_refs, list):
            # ``LectureIR`` does not (yet) expose a ``cross_references``
            # field. Stash on a non-validated alias so future callers
            # can read the resolved relations without the schema
            # rejecting the dict at validation time.
            local_ir["_cross_references"] = [c for c in cross_refs if isinstance(c, dict)]

    @staticmethod
    def _format_mainline_step(step: Any) -> str:
        if isinstance(step, str):
            return step.strip()
        if isinstance(step, dict):
            title = str(step.get("title") or "").strip()
            chap = step.get("chapter_index")
            if chap is not None and title:
                return f"第 {int(chap)} 章 · {title}"
            return title or json.dumps(step, ensure_ascii=False)
        return str(step).strip()

    @staticmethod
    def _merge_global_glossary(
        existing: list[dict[str, Any]],
        global_items: list[Any],
    ) -> list[dict[str, Any]]:
        by_term: dict[str, dict[str, Any]] = {}
        for ku in existing:
            term = str(ku.get("title") or "").strip()
            if term:
                by_term[term] = dict(ku)

        next_id = len(by_term) + 1
        for item in global_items:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term") or "").strip()
            if not term:
                continue
            definition = str(item.get("definition") or "").strip()
            chap = item.get("winning_chapter_index") or item.get("chapter_index")
            base = by_term.get(term)
            if base is None:
                base = {
                    "id": f"ku-{next_id}",
                    "type": "concept",
                    "title": term,
                    "explanation": definition,
                    "ts": 0.0,
                    "quote": "",
                    "chapter_index": int(chap) if chap is not None else 1,
                }
                next_id += 1
            else:
                if definition:
                    base["explanation"] = definition
                if chap is not None:
                    base["chapter_index"] = int(chap)
            by_term[term] = base
        return list(by_term.values())

    def _format_global_user(
        self,
        *,
        local_ir: dict[str, Any],
        glossary_conflicts: list[dict[str, Any]],
        meta: Any,
        study_questions: list[str],
    ) -> str:
        digest_lines: list[str] = []
        for ch in local_ir.get("chapters") or []:
            head_points = [
                str(p.get("text", "")).strip()
                for p in (ch.get("points") or [])[:3]
                if isinstance(p, dict)
            ]
            digest_lines.append(
                f"- 第 {ch.get('index')} 章 [{ch.get('start', 0):.0f}s-{ch.get('end', 0):.0f}s] "
                f"《{ch.get('title', '')}》：{ch.get('summary', '')}"
            )
            for hp in head_points:
                if hp:
                    digest_lines.append(f"  · {hp}")
        chapter_digest = "\n".join(digest_lines) or "(无章节)"

        if glossary_conflicts:
            glossary_lines = [
                f"- 「{c['term']}」第{c['chapter_index']}章 conf={c['confidence']:.2f} → {c['definition']}"
                for c in glossary_conflicts
            ]
            glossary_block = "\n".join(glossary_lines)
        else:
            glossary_block = "(无)"

        questions_block = (
            "\n".join(f"- {q}" for q in study_questions) if study_questions else "(无)"
        )
        return LECTURE_IR_REDUCE_GLOBAL_USER_TEMPLATE.format(
            bv_id=getattr(meta, "bv_id", ""),
            title=getattr(meta, "title", ""),
            duration_sec=int(getattr(meta, "duration", 0) or 0),
            chapter_digest=chapter_digest,
            glossary_conflicts=glossary_block,
            study_questions_block=questions_block,
        )

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        *,
        system: str,
        user: str,
        timeout: float,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """Single ``chat.completions.create`` call wrapped in ``wait_for``.

        ``asyncio.wait_for`` enforces the per-call timeout independently
        of whatever the underlying client may already do — keeps test
        responders that just raise ``TimeoutError`` honest.

        Returns ``(content, finish_reason)``. ``finish_reason`` is the
        empty string when the backend (or a test stub) does not surface
        the field, so callers can branch on ``== 'length'`` for the
        explicit truncation case without crashing on legacy stubs.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.25,
            "response_format": {"type": "json_object"},
            "timeout": timeout,
        }
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        coro = self._client.chat.completions.create(**kwargs)
        resp = await asyncio.wait_for(coro, timeout=timeout)
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None) or ""
        return content, finish_reason


__all__ = [
    "MapReduceIRBuilder",
    "MapReduceStats",
    "ReduceGlobalError",
]
