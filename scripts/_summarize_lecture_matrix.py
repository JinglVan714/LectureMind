from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings


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


def _row(timing: dict[str, Any]) -> dict[str, Any]:
    metrics = timing.get("metrics") or {}
    quality = timing.get("quality_metrics") or {}
    validation = timing.get("validation") or {}
    violations = validation.get("violations") or []
    warnings = validation.get("warnings") or []
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
        "timing_path": timing.get("timing_path") or "",
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
        "",
        "| BV | Label | Duration | OK | Chapters | Budget | Coverage | Max Span | 2H Ch | 2H Points | Points/min | KU/min | Violations | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['bv_id']} | {row['duration_label']} | {row['duration_sec']} | {row['ok']} | "
            f"{row['chapters']} | {row['chapter_budget']} | {row['coverage']} | {row['max_chapter_span_sec']} | "
            f"{row['second_half_chapters']} | {row['second_half_points']} | {row['points_per_min']} | "
            f"{row['ku_per_min']} | {row['violations']} | {row['warnings']} |"
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
