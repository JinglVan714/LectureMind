from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.understand.length_adapt import chapters_target


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="Path to a timing JSON file. Can be repeated.")
    parser.add_argument("--bv", action="append", default=[], help="BV id whose timing JSON should be read from data/debug. Can be repeated.")
    parser.add_argument("--out", default="lecture_matrix_report", help="Output basename or path without extension.")
    return parser.parse_args()


def _timing_paths(args: argparse.Namespace) -> list[Path]:
    settings = get_settings()
    paths = [Path(x) for x in args.input]
    paths.extend(settings.data_dir / "debug" / f"{bv}.timing.json" for bv in args.bv)
    if not paths:
        paths = sorted((settings.data_dir / "debug").glob("*.timing.json"))
    return paths


def _load_timing(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    data.setdefault("timing_path", str(path))
    return data


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _load_summary_json(bv_id: str) -> dict[str, Any] | None:
    db_path = get_settings().db_path
    if not bv_id or not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT summary_json FROM summaries WHERE bv_id=?", (bv_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    data = json.loads(row[0] or "{}")
    return data if isinstance(data, dict) else None


def _enrich_timing(timing: dict[str, Any]) -> dict[str, Any]:
    if timing.get("quality_metrics") and timing.get("metrics"):
        return timing
    summary_json = _load_summary_json(str(timing.get("bv_id") or timing.get("input") or ""))
    if not summary_json:
        return timing
    if not timing.get("metrics"):
        timing["metrics"] = _count_summary(summary_json)
    if not timing.get("quality_metrics"):
        timing["quality_metrics"] = _quality_metrics(summary_json)
        timing["quality_metrics_source"] = "db_summary_json"
    return timing


def _acceptance_issues(timing: dict[str, Any], metrics: dict[str, Any], quality: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    duration = _num(timing.get("duration_sec"))
    if not timing.get("ok"):
        issues.append("verifier_failed")
    if not quality:
        issues.append("quality_metrics_missing")
        return issues
    if quality.get("chapters_in_budget") is False:
        issues.append("chapters_out_of_budget")
    if _num(quality.get("chapter_coverage_ratio")) < 0.9:
        issues.append("coverage_below_0.9")
    if duration >= 600 and _num(quality.get("second_half_chapters")) <= 0:
        issues.append("second_half_chapters_missing")
    if duration >= 600 and _num(quality.get("second_half_points")) <= 0:
        issues.append("second_half_points_missing")
    if _num(quality.get("points_ts_out_of_chapter")) > 0:
        issues.append("points_ts_out_of_chapter")
    if timing.get("expect_code") and _num(metrics.get("code_blocks")) <= 0:
        issues.append("expected_code_blocks_missing")
    return issues


def _row(timing: dict[str, Any]) -> dict[str, Any]:
    timing = _enrich_timing(timing)
    metrics = timing.get("metrics") or {}
    quality = timing.get("quality_metrics") or {}
    validation = timing.get("validation") or {}
    violations = validation.get("violations") or []
    warnings = validation.get("warnings") or []
    acceptance_issues = _acceptance_issues(timing, metrics, quality)
    return {
        "ok": bool(timing.get("ok")),
        "bv_id": timing.get("bv_id") or timing.get("input") or "",
        "duration_label": timing.get("duration_label") or "",
        "duration_sec": timing.get("duration_sec") or 0,
        "generation_mode": timing.get("generation_mode") or "",
        "chapters": metrics.get("chapters", 0),
        "chapter_budget": f"{quality.get('chapters_target_min', '')}-{quality.get('chapters_target_max', '')}",
        "chapters_in_budget": quality.get("chapters_in_budget"),
        "coverage": quality.get("chapter_coverage_ratio", 0),
        "max_chapter_span_sec": quality.get("max_chapter_span_sec", 0),
        "second_half_chapters": quality.get("second_half_chapters", 0),
        "second_half_points": quality.get("second_half_points", 0),
        "points_per_min": quality.get("points_per_min", 0),
        "ku_per_min": quality.get("knowledge_units_per_min", 0),
        "visuals_per_min": quality.get("visual_evidence_per_min", 0),
        "code_blocks": metrics.get("code_blocks", 0),
        "formula_blocks": metrics.get("formula_blocks", 0),
        "elapsed_sec": timing.get("elapsed_sec", 0),
        "violations": len(violations),
        "warnings": len(warnings),
        "acceptance_issues": acceptance_issues,
        "acceptance_issue_count": len(acceptance_issues),
        "timing_path": timing.get("timing_path") or "",
        "quality_metrics_source": timing.get("quality_metrics_source") or "timing_json",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    for row in rows:
        label = str(row.get("duration_label") or "unknown")
        labels[label] = labels.get(label, 0) + 1
    return {
        "total": len(rows),
        "ok": sum(1 for row in rows if row["ok"]),
        "failed": sum(1 for row in rows if not row["ok"]),
        "acceptance_failed": sum(1 for row in rows if row.get("acceptance_issue_count", 0) > 0),
        "by_duration_label": labels,
        "avg_points_per_min": round(sum(_num(row.get("points_per_min")) for row in rows) / len(rows), 4) if rows else 0.0,
        "avg_ku_per_min": round(sum(_num(row.get("ku_per_min")) for row in rows) / len(rows), 4) if rows else 0.0,
        "min_coverage": min((_num(row.get("coverage")) for row in rows), default=0.0),
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "# Lecture Validation Matrix",
        "",
        f"- Total: {report['summary']['total']}",
        f"- OK: {report['summary']['ok']}",
        f"- Failed: {report['summary']['failed']}",
        f"- Acceptance failed: {report['summary']['acceptance_failed']}",
        "",
        "| BV | Label | Duration | OK | Chapters | Budget | Coverage | Max Span | 2H Ch | 2H Points | Points/min | KU/min | Violations | Warnings | Acceptance issues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['bv_id']} | {row['duration_label']} | {row['duration_sec']} | {row['ok']} | "
            f"{row['chapters']} | {row['chapter_budget']} | {row['coverage']} | {row['max_chapter_span_sec']} | "
            f"{row['second_half_chapters']} | {row['second_half_points']} | {row['points_per_min']} | "
            f"{row['ku_per_min']} | {row['violations']} | {row['warnings']} | "
            f"{', '.join(row['acceptance_issues']) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _output_paths(out: str) -> tuple[Path, Path]:
    out_path = Path(out)
    if out_path.suffix:
        base = out_path.with_suffix("")
    else:
        base = out_path
    return base.with_suffix(".json"), base.with_suffix(".md")


def main() -> None:
    _configure_stdio()
    args = _parse_args()
    timings = []
    for path in _timing_paths(args):
        if not path.exists():
            print(f"missing timing file: {path}", file=sys.stderr)
            continue
        timings.append(_load_timing(path))
    rows = [_row(timing) for timing in timings]
    rows.sort(key=lambda row: (_num(row.get("duration_sec")), str(row.get("bv_id"))))
    report = {"summary": _summary(rows), "rows": rows}
    json_path, md_path = _output_paths(args.out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
