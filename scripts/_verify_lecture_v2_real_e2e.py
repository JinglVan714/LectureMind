from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.copilot.diagnose import write_timing_json
from app.pipeline import Pipeline
from app.storage.db import Database
from app.understand.length_adapt import chapters_target


_DURATION_RANGES = {
    "short": (0, 600),
    "medium": (600, 1200),
    "long": (1200, 10**9),
}


def _count_summary(summary_json: dict[str, Any]) -> dict[str, Any]:
    chapters = summary_json.get("chapters") or []
    knowledge_units = summary_json.get("knowledge_units") or []
    visual_evidence = summary_json.get("visual_evidence") or []
    study_questions = summary_json.get("study_questions") or []
    return {
        "chapters": len(chapters),
        "points": sum(len(ch.get("points") or []) for ch in chapters),
        "teaching_notes": sum(len(ch.get("teaching_notes") or []) for ch in chapters),
        "chapter_frames": sum(len(ch.get("frames") or []) for ch in chapters),
        "visual_evidence": len(visual_evidence),
        "knowledge_units": len(knowledge_units),
        "study_questions": len(study_questions),
        "code_blocks": sum(len(ch.get("code_blocks") or []) for ch in chapters),
        "formula_blocks": sum(len(ch.get("formula_blocks") or []) for ch in chapters),
        "review_questions": len(summary_json.get("review_questions") or []),
        "learning_path": len(summary_json.get("learning_path") or []),
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quality_metrics(summary_json: dict[str, Any]) -> dict[str, Any]:
    duration = max(0.0, _num(summary_json.get("duration")))
    minutes = duration / 60.0 if duration > 0 else 0.0
    density_denominator = minutes if minutes > 0 else 1.0
    chapters = [x for x in (summary_json.get("chapters") or []) if isinstance(x, dict)]
    knowledge_units = [x for x in (summary_json.get("knowledge_units") or []) if isinstance(x, dict)]
    visual_evidence = [x for x in (summary_json.get("visual_evidence") or []) if isinstance(x, dict)]
    target_min, target_max = chapters_target(duration)
    second_half_start = duration / 2.0 if duration > 0 else 0.0

    chapter_ranges: list[tuple[float, float, dict[str, Any]]] = []
    for ch in chapters:
        start = max(0.0, _num(ch.get("start")))
        end = max(start, _num(ch.get("end"), start))
        chapter_ranges.append((start, end, ch))

    spans = [end - start for start, end, _ in chapter_ranges]
    max_end = max((end for _, end, _ in chapter_ranges), default=0.0)
    points = []
    code_blocks = []
    formula_blocks = []
    teaching_notes_count = 0
    points_ts_out_of_chapter = 0
    for start, end, ch in chapter_ranges:
        ch_points = [x for x in (ch.get("points") or []) if isinstance(x, dict)]
        points.extend(ch_points)
        teaching_notes_count += len(ch.get("teaching_notes") or [])
        code_blocks.extend([x for x in (ch.get("code_blocks") or []) if isinstance(x, dict)])
        formula_blocks.extend([x for x in (ch.get("formula_blocks") or []) if isinstance(x, dict)])
        for point in ch_points:
            ts = _num(point.get("ts"))
            if ts < start - 1.0 or ts > end + 1.0:
                points_ts_out_of_chapter += 1

    return {
        "chapters_target_min": target_min,
        "chapters_target_max": target_max,
        "chapters_in_budget": target_min <= len(chapters) <= target_max,
        "chapter_coverage_ratio": round(max_end / duration, 4) if duration > 0 else 0.0,
        "max_chapter_span_sec": round(max(spans), 3) if spans else 0.0,
        "avg_chapter_span_sec": round(sum(spans) / len(spans), 3) if spans else 0.0,
        "second_half_chapters": sum(1 for start, end, _ in chapter_ranges if end > second_half_start and start < duration),
        "second_half_points": sum(1 for pnt in points if _num(pnt.get("ts")) >= second_half_start),
        "second_half_knowledge_units": sum(1 for ku in knowledge_units if _num(ku.get("ts")) >= second_half_start),
        "second_half_visuals": sum(1 for frame in visual_evidence if _num(frame.get("ts")) >= second_half_start),
        "points_per_min": round(len(points) / density_denominator, 4),
        "teaching_notes_per_min": round(teaching_notes_count / density_denominator, 4),
        "knowledge_units_per_min": round(len(knowledge_units) / density_denominator, 4),
        "visual_evidence_per_min": round(len(visual_evidence) / density_denominator, 4),
        "code_blocks_per_min": round(len(code_blocks) / density_denominator, 4),
        "formula_blocks_per_min": round(len(formula_blocks) / density_denominator, 4),
        "points_ts_zero": sum(1 for pnt in points if _num(pnt.get("ts")) <= 0.0),
        "points_ts_out_of_chapter": points_ts_out_of_chapter,
        "ku_ts_zero": sum(1 for ku in knowledge_units if _num(ku.get("ts")) <= 0.0),
        "visual_ts_zero": sum(1 for frame in visual_evidence if _num(frame.get("ts")) <= 0.0),
    }


def _inspect_html(report_path: Path) -> dict[str, bool]:
    if not report_path.exists():
        return {
            "exists": False,
            "study_questions_section": False,
            "code_card": False,
            "formula_card": False,
            "highlight_loader": False,
        }
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "exists": True,
        "study_questions_section": "study-questions" in text,
        "code_card": "code-card" in text,
        "formula_card": "formula-card" in text,
        "highlight_loader": "highlightAll" in text or "highlight.min.js" in text,
    }


def _sqlite_counts(db_path: Path, bv_id: str) -> dict[str, Any]:
    if not db_path.exists():
        return {"chunks": {}, "assets": {}}
    conn = sqlite3.connect(db_path)
    try:
        chunks = Counter(
            dict(conn.execute(
                "SELECT kind, COUNT(*) FROM lecture_chunks WHERE bv_id=? GROUP BY kind",
                (bv_id,),
            ).fetchall())
        )
        assets = Counter(
            dict(conn.execute(
                "SELECT kind, COUNT(*) FROM assets WHERE bv_id=? GROUP BY kind",
                (bv_id,),
            ).fetchall())
        )
        return {"chunks": dict(chunks), "assets": dict(assets)}
    finally:
        conn.close()


def _validate(
    *,
    duration_label: str,
    expect_code: bool,
    summary: Any,
    summary_json: dict[str, Any],
    metrics: dict[str, Any],
    quality_metrics: dict[str, Any],
    render_plan: dict[str, Any],
    report_path: Path,
    ir_path: Path,
    html: dict[str, bool],
    db_counts: dict[str, Any],
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []
    chunks = db_counts.get("chunks") or {}
    assets = db_counts.get("assets") or {}
    duration = int(summary.duration or summary_json.get("duration") or 0)
    generation_mode = str(summary_json.get("generation_mode") or "")

    if summary.status != "done":
        violations.append(f"summary status is {summary.status!r}, expected 'done'")
    if summary.error_msg:
        violations.append(f"summary has error_msg: {summary.error_msg}")
    if not report_path.exists():
        violations.append(f"report html missing: {report_path}")
    if not ir_path.exists() and generation_mode != "v1_fallback":
        violations.append(f"LectureIR debug json missing: {ir_path}")
    if generation_mode == "v1_fallback":
        violations.append("pipeline fell back to v1_fallback")
    if metrics["chapters"] <= 0:
        violations.append("no chapters generated")
    if metrics["learning_path"] <= 0:
        violations.append("no learning_path generated")
    if metrics["knowledge_units"] <= 0:
        violations.append("no knowledge_units generated")
    if metrics["study_questions"] <= 0:
        violations.append("no study_questions generated")
    if expect_code and metrics["code_blocks"] <= 0:
        violations.append("expected a code lecture, but no code_blocks were generated")
    if metrics["code_blocks"] > 0 and not render_plan.get("code_section"):
        violations.append("code_blocks exist but render_plan.code_section is false")
    if metrics["formula_blocks"] > 0 and not render_plan.get("formula_section"):
        violations.append("formula_blocks exist but render_plan.formula_section is false")
    if metrics["code_blocks"] <= 0 and render_plan.get("code_section"):
        warnings.append("render_plan.code_section is true but no code_blocks exist")
    if metrics["formula_blocks"] <= 0 and render_plan.get("formula_section"):
        warnings.append("render_plan.formula_section is true but no formula_blocks exist")
    if metrics["study_questions"] > 0 and not html.get("study_questions_section"):
        violations.append("study_questions exist but HTML study-questions section is missing")
    if metrics["code_blocks"] > 0 and not html.get("code_card"):
        violations.append("code_blocks exist but HTML code-card is missing")
    if metrics["formula_blocks"] > 0 and not html.get("formula_card"):
        violations.append("formula_blocks exist but HTML formula-card is missing")
    if metrics["code_blocks"] > 0 and not html.get("highlight_loader"):
        violations.append("code_blocks exist but highlight.js loader is missing")
    if sum(chunks.values()) <= 0:
        violations.append("no RAG chunks indexed")
    if metrics["study_questions"] > 0 and chunks.get("study_question", 0) <= 0:
        violations.append("study_questions exist but no study_question RAG chunks were indexed")
    if metrics["code_blocks"] > 0 and chunks.get("code_block", 0) <= 0:
        violations.append("code_blocks exist but no code_block RAG chunks were indexed")
    if metrics["formula_blocks"] > 0 and chunks.get("formula_block", 0) <= 0:
        violations.append("formula_blocks exist but no formula_block RAG chunks were indexed")
    if assets.get("subtitle", 0) <= 0:
        violations.append("subtitle asset missing")
    if assets.get("keyframe", 0) <= 0:
        violations.append("keyframe assets missing")

    if quality_metrics.get("chapters_in_budget") is False:
        warnings.append(
            "chapter count is outside length_adapt budget "
            f"[{quality_metrics.get('chapters_target_min')}, {quality_metrics.get('chapters_target_max')}]"
        )
    if duration >= 1200 and quality_metrics.get("chapter_coverage_ratio", 0.0) < 0.9:
        warnings.append("long video chapter coverage ratio is below 0.9")
    if duration >= 1200 and quality_metrics.get("second_half_chapters", 0) <= 0:
        warnings.append("long video has no chapter coverage in the second half")
    if quality_metrics.get("points_ts_out_of_chapter", 0) > 0:
        warnings.append("some point timestamps are outside their chapter ranges")
    if duration_label in _DURATION_RANGES:
        lo, hi = _DURATION_RANGES[duration_label]
        if not (lo <= duration < hi):
            violations.append(f"duration {duration}s is outside expected {duration_label} range [{lo}, {hi})")
    return violations, warnings


async def _async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    started = time.perf_counter()
    progress_events: list[dict[str, Any]] = []
    if not settings.dashscope_api_key:
        timing = {
            "ok": False,
            "duration_label": args.duration,
            "expect_code": args.expect_code,
            "error": "DASHSCOPE_API_KEY is not set",
        }
        write_timing_json(args.bv, timing)
        print(json.dumps(timing, ensure_ascii=False, indent=2))
        return 2

    db = Database(settings.db_path)
    await db.init()
    pipeline = Pipeline(db)

    async def progress_cb(progress: int, message: str) -> None:
        event = {
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "progress": progress,
            "message": message,
        }
        progress_events.append(event)
        print(f"[{progress:3d}%] {message}", file=sys.stderr, flush=True)

    try:
        report_path = await pipeline.run(args.bv, force_refresh=args.force, progress_cb=progress_cb)
        pipeline_timing = dict(getattr(pipeline, "last_run_timing", {}) or {})
        pipeline_stats = dict(getattr(pipeline, "last_run_stats", {}) or {})
    except Exception as exc:
        timing = {
            "ok": False,
            "duration_label": args.duration,
            "expect_code": args.expect_code,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "progress_events": progress_events,
            "error": repr(exc),
        }
        write_timing_json(args.bv, timing)
        print(json.dumps(timing, ensure_ascii=False, indent=2))
        return 2

    bv_id = report_path.stem
    summary = await db.get_summary(bv_id)
    if summary is None:
        timing = {
            "ok": False,
            "bv_id": bv_id,
            "duration_label": args.duration,
            "expect_code": args.expect_code,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "progress_events": progress_events,
            "report_path": str(report_path),
            "error": "summary row missing after pipeline run",
        }
        write_timing_json(bv_id, timing)
        print(json.dumps(timing, ensure_ascii=False, indent=2))
        return 2

    summary_json = summary.summary_json or {}
    metrics = _count_summary(summary_json)
    quality_metrics = _quality_metrics(summary_json)
    render_plan = summary_json.get("render_plan") or {}
    ir_path = settings.data_dir / "debug" / f"{bv_id}.lecture_ir.json"
    html = _inspect_html(report_path)
    db_counts = _sqlite_counts(settings.db_path, bv_id)
    violations, warnings = _validate(
        duration_label=args.duration,
        expect_code=args.expect_code,
        summary=summary,
        summary_json=summary_json,
        metrics=metrics,
        quality_metrics=quality_metrics,
        render_plan=render_plan,
        report_path=report_path,
        ir_path=ir_path,
        html=html,
        db_counts=db_counts,
    )
    timing = {
        "ok": not violations,
        "bv_id": bv_id,
        "input": args.bv,
        "duration_label": args.duration,
        "expect_code": args.expect_code,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "progress_events": progress_events,
        "title": summary.title,
        "author": summary.author,
        "duration_sec": summary.duration,
        "report_path": str(report_path),
        "ir_path": str(ir_path),
        "model_used": summary.model_used,
        "token_cost": summary.token_cost,
        "generation_mode": summary_json.get("generation_mode"),
        "taxonomy": summary_json.get("taxonomy"),
        "render_plan": render_plan,
        "metrics": metrics,
        "quality_metrics": quality_metrics,
        "html": html,
        "db": db_counts,
        "validation": {
            "violations": violations,
            "warnings": warnings,
        },
        "pipeline_timing": pipeline_timing,
        "pipeline_stats": pipeline_stats,
    }
    timing_path = write_timing_json(bv_id, timing)
    timing["timing_path"] = str(timing_path)
    print(json.dumps(timing, ensure_ascii=False, indent=2))
    return 0 if not violations else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bv", required=True, help="Bilibili BV id or URL")
    parser.add_argument("--duration", choices=["short", "medium", "long", "code", "tiny", "standard", "epic", "code-heavy"], required=True)
    parser.add_argument("--expect-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
