"""End-to-end pipeline glue: BV link → LectureJSON → HTML.

Kept in its own module so api.py stays thin and the pipeline can be
exercised from a CLI / unit test without spinning up FastAPI.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .config import get_settings
from .copilot.indexer import index_lecture
from .copilot.rag import RAGStore
from .copilot.taxonomy import normalize as normalize_taxonomy
from .ingest.bilibili import BilibiliIngest, VideoMeta
from .ingest.cover import CoverCache
from .ingest.keyframe import KeyframeExtractor
from .ingest.subtitle import SubtitleExtractor
from .render.renderer import Renderer
from .storage.db import Database
from .understand.ir import LectureIR
from .understand.ir_builder import LectureIRBuilder, lecture_ir_to_lecture_json
from .understand.lecturize import LecturizeContext, Lecturizer
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

        # --- 1. metadata ---
        await _progress(5, "fetching metadata")
        meta: VideoMeta = await self.bili.fetch_meta(url)

        # cache check
        if not force_refresh:
            existing = await self.db.get_summary(meta.bv_id)
            if existing and existing.status == "done":
                report_path = self.settings.data_dir / existing.report_path
                if report_path.exists():
                    await _progress(100, "cache hit")
                    return report_path

        # --- 2. subtitle ---
        await _progress(15, "extracting subtitle")
        sub_result = await self.subtitles.extract(meta.bv_id, meta.aid, meta.primary_cid)
        await self.db.add_asset(
            meta.bv_id, "subtitle", str(sub_result.raw_path),
            meta={"source": sub_result.source, "language": sub_result.language},
        )

        # --- 3. keyframes ---
        await _progress(35, "extracting keyframes")
        frames = await self.keyframes.extract(meta.bv_id, meta.duration)
        for f in frames:
            await self.db.add_asset(meta.bv_id, "keyframe", str(f.path), meta={"ts": f.timestamp})

        # --- 4. VLM frame descriptions ---
        await _progress(55, "describing frames with VLM")
        frame_descs: list[FrameDescription] = await self.vlm.describe_all(frames)

        # --- 5. structured lecture ---
        await _progress(75, "generating LectureIR")
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
        lecture_ir: LectureIR | None = None
        try:
            lecture_ir, stats = await self.ir_builder.build(ctx)
            self._dump_ir(meta.bv_id, lecture_ir.model_dump(mode="json"))
            lecture = lecture_ir_to_lecture_json(lecture_ir)
            transcript = " ".join(s.text for s in sub_result.segments)
            marked = lecture.mark_unverified_points(transcript)
            stats["marked_unverified_points"] = marked
        except Exception as exc:  # noqa: BLE001
            logger.warning("LectureIR v2 failed for %s; falling back to v1 LectureJSON: %s", meta.bv_id, exc)
            await _progress(78, "LectureIR failed, falling back to v1")
            lecture, stats = await self.lecturizer.lecturize(ctx)
            lecture.generation_mode = "v1_fallback"
            lecture.completeness.notes = "LectureIR 分析失败，已回退到 v1 讲义生成路径。"
            lecture_ir = None

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
        cover_path = await self.covers.fetch(meta.bv_id, meta.cover_url)
        css_inline = self.renderer.load_inline_css()
        report_path = self.renderer.render_lecture(lecture, css_inline, cover_path=cover_path)

        # --- 7. persist ---
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

        # --- 8. Copilot RAG indexing (best-effort) ---
        await _progress(95, "indexing for Copilot RAG")
        try:
            n = await index_lecture(self.rag, meta.bv_id, lecture, lecture_ir)
            logger.info("Copilot RAG indexed %d chunks for %s", n, meta.bv_id)
        except Exception as exc:  # noqa: BLE001
            # index_lecture already swallows its own errors; this catch is a
            # belt-and-braces safety net so RAG never blocks the pipeline.
            logger.warning("Copilot RAG hook failed for %s: %s", meta.bv_id, exc)

        await _progress(100, "done")
        return report_path

    def _dump_ir(self, bv_id: str, data: dict) -> None:
        debug_dir = self.settings.data_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{bv_id}.lecture_ir.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
