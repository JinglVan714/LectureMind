"""End-to-end pipeline glue: BV link → LectureJSON → HTML.

Kept in its own module so api.py stays thin and the pipeline can be
exercised from a CLI / unit test without spinning up FastAPI.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import get_settings
from .copilot.indexer import index_lecture
from .copilot.rag import RAGStore
from .copilot.taxonomy import normalize as normalize_taxonomy
from .ingest.bilibili import BilibiliIngest, VideoMeta
from .ingest.cover import CoverCache
from .ingest.keyframe import KeyframeExtractor
from .ingest.subtitle import SubtitleExtractor
from .render.renderer import Renderer
from .runtime.contracts import RunRecord
from .runtime.observe import (
    create_run_record,
    emit_artifact,
    issue_verdict,
    write_run_record,
)
from .storage.db import Database
from .understand.chapter_cache import make_chapter_cache_from_settings
from .understand.chapter_planner import plan_chapters
from .understand.ir import LectureIR
from .understand.ir_builder import LectureIRBuilder, lecture_ir_to_lecture_json
from .understand.lecturize import LecturizeContext, Lecturizer
from .understand.length_adapt import LengthBudget
from .understand.profile import select_profile
from .understand.schema import Frame as LectureFrame
from .understand.vlm import FrameDescriber, FrameDescription

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.settings = get_settings()
        self.bili = BilibiliIngest()
        self.covers = CoverCache()
        self.subtitles = SubtitleExtractor()
        self.keyframes = KeyframeExtractor()
        self.vlm = FrameDescriber()
        self.ir_builder = LectureIRBuilder()
        self.lecturizer = Lecturizer()
        self.renderer = Renderer()
        self._concurrency = asyncio.Semaphore(self.settings.max_concurrent_jobs)
        # Copilot RAG store — initialised lazily on first index/search call.
        self.rag = RAGStore(db.db_path)
        # Last-run telemetry, populated by ``_run_inner``. Verifier scripts
        # and tests can read these to surface fine-grained stage timings
        # without re-querying the DB. Empty dicts after a fresh init.
        self.last_run_timing: dict[str, Any] = {}
        self.last_run_stats: dict[str, Any] = {}
        self.last_run_rag_task: asyncio.Task[Any] | None = None
        self.last_run_record: RunRecord | None = None

    async def run(
        self,
        url: str,
        *,
        force_refresh: bool = False,
        progress_cb=None,
    ) -> Path:
        """Run end-to-end. Returns the report path on success."""
        async with self._concurrency:
            return await self._run_inner(url, force_refresh=force_refresh, progress_cb=progress_cb)

    async def _run_inner(self, url: str, *, force_refresh: bool, progress_cb) -> Path:
        async def _progress(p: int, msg: str = "") -> None:
            logger.info("[%d%%] %s", p, msg)
            if progress_cb:
                await progress_cb(p, msg)

        # Reset per-run telemetry. Stage timings are recorded throughout
        # ``_run_inner`` and exposed via ``self.last_run_timing`` so
        # callers (verifier, tests) can surface phase breakdowns without
        # parsing logs.
        timing: dict[str, Any] = {}
        self.last_run_timing = timing
        self.last_run_stats = {}
        self.last_run_rag_task = None
        self.last_run_record = None

        # --- 1. metadata ---
        await _progress(5, "fetching metadata")
        t_meta = time.perf_counter()
        meta: VideoMeta = await self.bili.fetch_meta(url)
        timing["metadata_sec"] = round(time.perf_counter() - t_meta, 3)
        record = create_run_record(
            run_type="lecture_compile",
            entrypoint="pipeline",
            target_ref=meta.bv_id,
            expected_artifacts=["lecture_ir", "report_html", "timing_record"],
            expected_profile=None,
            writeback_allowed=True,
            minimum_acceptance=["report_rendered", "summary_persisted"],
        )
        timing["trace_id"] = record.trace_id
        timing["run_contract"] = record.run_contract
        timing["policy_snapshot"] = record.policy_snapshot

        # cache check
        if not force_refresh:
            existing = await self.db.get_summary(meta.bv_id)
            if existing and existing.status == "done":
                report_path = self.settings.data_dir / existing.report_path
                if report_path.exists():
                    timing["cache_hit"] = True
                    emit_artifact(
                        record,
                        artifact_type="report_html",
                        path_or_ref=str(report_path),
                        producer="renderer",
                    )
                    record.verdict = issue_verdict(
                        status="accept",
                        reasons=["cache_hit"],
                        warnings=[],
                        next_actions=[],
                        evidence_refs=[str(report_path)],
                    )
                    record.status = "finished"
                    timing["verdict"] = record.verdict
                    self.last_run_record = record
                    write_run_record(record, self.settings.data_dir / "debug")
                    await _progress(100, "cache hit")
                    return report_path

        # --- 2. subtitle + keyframes + cover (parallel) ---
        # subtitle / keyframes / cover only depend on metadata, so they
        # can run concurrently. ``cover_task`` is awaited later, right
        # before render. We still surface per-stage timing so the
        # verifier can see how much each leg actually cost.
        await _progress(15, "extracting subtitle, keyframes, cover (parallel)")
        t_ingest = time.perf_counter()

        async def _timed_subtitle() -> tuple[Any, float]:
            t = time.perf_counter()
            res = await self.subtitles.extract(meta.bv_id, meta.aid, meta.primary_cid)
            return res, time.perf_counter() - t

        async def _timed_keyframes() -> tuple[Any, float]:
            t = time.perf_counter()
            res = await self.keyframes.extract(meta.bv_id, meta.duration)
            return res, time.perf_counter() - t

        async def _timed_cover() -> tuple[Any, float]:
            t = time.perf_counter()
            res = await self.covers.fetch(meta.bv_id, meta.cover_url)
            return res, time.perf_counter() - t

        (sub_result, subtitle_sec), (frames, keyframes_sec), (cover_path, cover_sec) = (
            await asyncio.gather(_timed_subtitle(), _timed_keyframes(), _timed_cover())
        )
        timing["subtitle_sec"] = round(subtitle_sec, 3)
        timing["keyframes_sec"] = round(keyframes_sec, 3)
        timing["cover_sec"] = round(cover_sec, 3)
        timing["ingest_parallel_sec"] = round(time.perf_counter() - t_ingest, 3)
        timing["subtitle_source"] = sub_result.source
        timing["keyframes_count"] = len(frames)

        # Persist assets sequentially (SQLite single-writer friendly).
        await self.db.add_asset(
            meta.bv_id, "subtitle", str(sub_result.raw_path),
            meta={"source": sub_result.source, "language": sub_result.language},
        )
        for f in frames:
            await self.db.add_asset(meta.bv_id, "keyframe", str(f.path), meta={"ts": f.timestamp})

        # --- 4. VLM frame descriptions ---
        await _progress(55, "describing frames with VLM")
        t_vlm = time.perf_counter()
        frame_descs: list[FrameDescription] = await self.vlm.describe_all(frames)
        timing["vlm_sec"] = round(time.perf_counter() - t_vlm, 3)
        timing["vlm_concurrency"] = getattr(self.vlm, "_concurrency", None)
        # M3 telemetry: tiering hits / cache hits / estimated savings.
        timing["vlm"] = self.vlm.last_telemetry

        # --- 5. structured lecture ---
        await _progress(75, "generating LectureIR")
        t_ir = time.perf_counter()
        ctx = LecturizeContext(
            bv_id=meta.bv_id,
            url=f"https://www.bilibili.com/video/{meta.bv_id}",
            title=meta.title,
            author=meta.author,
            duration=meta.duration,
            cover_url=meta.cover_url,
            segments=sub_result.segments,
            frame_descs=frame_descs,
        )

        # M2 P7: route by length profile. The profile selects which
        # downstream components get to run (chapter planner, chapter
        # cache, map-reduce IR, projected Critic, patch Reviser) so
        # each video class can spend its token / latency budget where
        # it actually matters.
        profile = select_profile(meta.duration, self.settings)
        timing["profile"] = {
            "name": profile.name,
            "duration_sec": float(profile.duration_sec),
            "use_chapter_planner": profile.use_chapter_planner,
            "use_map_reduce": profile.use_map_reduce,
            "use_chapter_cache": profile.use_chapter_cache,
            "critic_mode": profile.critic_mode,
            "reviser_mode": profile.reviser_mode,
            "study_question_mode": profile.study_question_mode,
            "chapter_planner_mode": profile.chapter_planner_mode,
        }
        record.run_contract.expected_profile = profile.name

        # Chapter planner — always runs (per spec) but the profile picks
        # `hint` vs `structural` which only differs in how the anchors
        # are surfaced to the LLM. Failures degrade to an empty plan
        # rather than fail the whole pipeline.
        chapter_plan: list[Any] = []
        chapter_plan_sec = 0.0
        if profile.use_chapter_planner:
            t_plan = time.perf_counter()
            try:
                ir_budget = LengthBudget.for_duration(
                    meta.duration,
                    keyframe_base_min=self.settings.keyframe_min,
                    keyframe_base_max=self.settings.keyframe_max,
                )
                chapter_plan = list(
                    plan_chapters(
                        profile=profile,
                        duration_sec=float(meta.duration),
                        segments=sub_result.segments,
                        frames=frame_descs,
                        chapters_min=ir_budget.chapters_min,
                        chapters_max=ir_budget.chapters_max,
                        settings=self.settings,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChapterPlanner failed (degrading to empty): %s", exc)
                chapter_plan = []
            chapter_plan_sec = time.perf_counter() - t_plan
        timing["chapter_plan"] = {
            "anchors": len(chapter_plan),
            "mode": profile.chapter_planner_mode if profile.use_chapter_planner else None,
            "sec": round(chapter_plan_sec, 3),
        }

        chapter_cache_obj = None
        if profile.use_chapter_cache:
            try:
                chapter_cache_obj = make_chapter_cache_from_settings(self.settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChapterCache init failed (skipping): %s", exc)
                chapter_cache_obj = None

        lecture_ir: LectureIR | None = None
        try:
            # Use the multi-agent path (study questions + critic-reviser
            # loop) when enabled; the inner method honours the per-flag
            # config (LECTURE_QUESTION_DRIVEN / LECTURE_CRITIC_ENABLED).
            use_agents = bool(
                getattr(self.settings, "lecture_question_driven", True)
                or getattr(self.settings, "lecture_critic_enabled", True)
            )
            if use_agents:
                lecture_ir, stats = await self.ir_builder.build_with_agents(
                    ctx,
                    profile=profile,
                    chapter_plan=chapter_plan,
                    chapter_cache=chapter_cache_obj,
                )
            else:
                lecture_ir, stats = await self.ir_builder.build(ctx)
            debug_ir_path = self._dump_ir(meta.bv_id, lecture_ir.model_dump(mode="json"))
            emit_artifact(
                record,
                artifact_type="lecture_ir",
                path_or_ref=str(debug_ir_path),
                producer="ir_builder",
            )
            lecture = lecture_ir_to_lecture_json(lecture_ir)
            transcript = " ".join(s.text for s in sub_result.segments)
            marked = lecture.mark_unverified_points(transcript)
            stats["marked_unverified_points"] = marked
        except Exception as exc:  # noqa: BLE001
            if getattr(self.settings, "lecture_strict_agents", False):
                logger.exception("LectureIR v2 failed for %s in strict mode", meta.bv_id)
                raise
            logger.warning("LectureIR v2 failed for %s; falling back to v1 LectureJSON: %s", meta.bv_id, exc)
            await _progress(78, "LectureIR failed, falling back to v1")
            lecture, stats = await self.lecturizer.lecturize(ctx)
            lecture.generation_mode = "v1_fallback"
            lecture.completeness.notes = "LectureIR 分析失败，已回退到 v1 讲义生成路径。"
            lecture_ir = None
        timing["lecture_ir_sec"] = round(time.perf_counter() - t_ir, 3)
        if isinstance(stats, dict) and stats.get("agent_timing"):
            timing["agent_timing"] = stats.get("agent_timing")
        timing["revise_rounds"] = stats.get("revise_rounds") if isinstance(stats, dict) else 0
        # M2 P7: surface map-reduce / reviser / chapter-cache telemetry
        # into ``timing`` so the verifier and matrix can inspect them
        # without re-deriving from stats.
        if isinstance(stats, dict):
            if "map_reduce" in stats:
                timing["map_reduce"] = stats["map_reduce"]
            if "reviser" in stats:
                timing["reviser"] = stats["reviser"]
            if isinstance(stats.get("critique"), dict):
                timing["critic"] = {
                    "verdict": stats["critique"].get("verdict"),
                    "issue_count": len(stats["critique"].get("issues") or []),
                    "metrics": stats["critique"].get("metrics") or {},
                }
        if chapter_cache_obj is not None:
            try:
                timing["chapter_cache"] = chapter_cache_obj.stats()
            except Exception:  # noqa: BLE001
                timing["chapter_cache"] = {}
        self.last_run_stats = stats if isinstance(stats, dict) else {}

        # Repair the LLM-supplied taxonomy: white-list the domain, strip
        # stage suffixes from direction, dedupe + cap tags.  When the LLM
        # produced no taxonomy at all (legacy / v1 fallback path) we leave
        # ``lecture.taxonomy`` as None so the index page surfaces it under
        # "待归类".
        lecture.taxonomy = normalize_taxonomy(lecture.taxonomy)
        if lecture.taxonomy and lecture.taxonomy.tags:
            lecture.domain_tags = list(lecture.taxonomy.tags)

        # Hydrate chapter.frames if the LLM omitted them: distribute frames
        # into the chapter whose [start,end] contains their timestamp.
        if frame_descs and not any(ch.frames for ch in lecture.chapters):
            for fd in frame_descs:
                target = next(
                    (ch for ch in lecture.chapters if ch.start <= fd.timestamp <= ch.end),
                    None,
                )
                if target is None:
                    continue
                target.frames.append(
                    LectureFrame(
                        ts=fd.timestamp,
                        path=str(fd.path),
                        caption=fd.caption or "",
                        ocr_text=fd.ocr_text or "",
                        insight=fd.caption or fd.ocr_text[:120],
                    )
                )

        # --- 6. render ---
        await _progress(90, "rendering HTML")
        t_render = time.perf_counter()
        # cover_path was fetched in parallel with subtitle/keyframes above.
        css_inline = self.renderer.load_inline_css()
        report_path = self.renderer.render_lecture(lecture, css_inline, cover_path=cover_path)
        timing["render_sec"] = round(time.perf_counter() - t_render, 3)
        emit_artifact(
            record,
            artifact_type="report_html",
            path_or_ref=str(report_path),
            producer="renderer",
        )

        # --- 7. persist ---
        t_persist = time.perf_counter()
        rel_report = report_path.relative_to(self.settings.data_dir).as_posix()
        domain = lecture.taxonomy.domain if lecture.taxonomy else None
        direction = lecture.taxonomy.direction if lecture.taxonomy else None
        await self.db.upsert_summary(
            bv_id=meta.bv_id,
            url=ctx.url,
            title=lecture.title,
            author=lecture.author,
            duration=int(lecture.duration),
            cover_url=lecture.cover_url,
            category=lecture.category,
            domain_tags=lecture.domain_tags,
            summary_json=lecture.model_dump(mode="json"),
            report_path=rel_report,
            model_used=stats.get("model"),
            token_cost=stats.get("tokens", {}).get("total_tokens", 0) or 0,
            status="done",
            domain=domain or None,
            direction=direction or None,
        )

        timing["persist_sec"] = round(time.perf_counter() - t_persist, 3)

        # --- 8. Copilot RAG indexing (best-effort) ---
        # When ``PIPELINE_WAIT_RAG`` is true (default) we keep the legacy
        # behaviour and await indexing. When false, indexing is launched
        # as a fire-and-forget background task so the report path is
        # returned sooner; failures still log a warning.
        wait_rag = bool(getattr(self.settings, "pipeline_wait_rag", True))
        timing["wait_rag"] = wait_rag

        async def _do_rag_index() -> int | None:
            try:
                n = await index_lecture(self.rag, meta.bv_id, lecture, lecture_ir)
                logger.info("Copilot RAG indexed %d chunks for %s", n, meta.bv_id)
                return n
            except Exception as exc:  # noqa: BLE001
                logger.warning("Copilot RAG hook failed for %s: %s", meta.bv_id, exc)
                return None

        if wait_rag:
            await _progress(95, "indexing for Copilot RAG")
            t_rag = time.perf_counter()
            await _do_rag_index()
            timing["rag_index_sec"] = round(time.perf_counter() - t_rag, 3)
        else:
            await _progress(95, "scheduling RAG indexing in background")
            timing["rag_index_sec"] = 0.0

            async def _bg_rag() -> None:
                t_rag = time.perf_counter()
                await _do_rag_index()
                timing["rag_index_sec_bg"] = round(time.perf_counter() - t_rag, 3)

            self.last_run_rag_task = asyncio.create_task(_bg_rag())

        await _progress(100, "done")
        record.verdict = issue_verdict(
            status="revise" if lecture.generation_mode == "v1_fallback" else "accept",
            reasons=["report_rendered", "summary_persisted"],
            warnings=["used_v1_fallback"] if lecture.generation_mode == "v1_fallback" else [],
            next_actions=[] if wait_rag else ["check_background_rag_index"],
            evidence_refs=[str(report_path)],
        )
        record.status = "finished"
        timing["verdict"] = record.verdict
        self.last_run_record = record
        write_run_record(record, self.settings.data_dir / "debug")
        return report_path

    def _dump_ir(self, bv_id: str, data: dict) -> Path:
        debug_dir = self.settings.data_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{bv_id}.lecture_ir.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
