"""HTML-only rerender reports from existing local structured data.

Usage:
    python -m scripts.rerender_reports
    python -m scripts.rerender_reports --only BV1fyNKz5Egb BV1JV96B5EFJ
    python -m scripts.rerender_reports --fail-fast
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.copilot.taxonomy import normalize as normalize_taxonomy
from app.render.renderer import Renderer
from app.understand._latex_repair import repair_obj as _repair_latex_escapes
from app.understand.ir import LectureIR
from app.understand.ir_builder import lecture_ir_to_lecture_json
from app.understand.schema import LectureJSON


@dataclass(frozen=True)
class RerenderResult:
    bv_id: str
    status: str
    source: str
    report: Path | None = None
    error: str = ""


def _bv_id_from_ir_path(path: Path) -> str:
    suffix = ".lecture_ir.json"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _discover_ir_files(debug_dir: Path) -> dict[str, Path]:
    return {_bv_id_from_ir_path(path): path for path in sorted(debug_dir.glob("*.lecture_ir.json"))}


def _load_db_summaries(db_path: Path) -> dict[str, dict]:
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT bv_id, summary_json FROM summaries WHERE status = 'done' ORDER BY bv_id"
        ).fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        try:
            out[str(row["bv_id"])] = json.loads(row["summary_json"] or "{}")
        except json.JSONDecodeError:
            logging.warning("Invalid summary_json in SQLite for %s", row["bv_id"])
    return out


def _discover_report_bv_ids(reports_dir: Path) -> set[str]:
    return {path.stem for path in reports_dir.glob("BV*.html")}


def _discover_targets(
    *,
    ir_files: dict[str, Path],
    db_summaries: dict[str, dict],
    report_bv_ids: set[str],
    only: list[str],
) -> list[str]:
    available = set(ir_files) | set(db_summaries) | report_bv_ids
    if only:
        wanted = set(only)
        missing = sorted(wanted - available)
        if missing:
            raise SystemExit(f"Unknown BV id(s): {', '.join(missing)}")
        return sorted(wanted)
    return sorted(available)


def _lecture_from_source(bv_id: str, ir_files: dict[str, Path], db_summaries: dict[str, dict]) -> tuple[LectureJSON, str]:
    # ``_repair_latex_escapes`` fixes historical reports whose IR JSON or
    # SQLite ``summary_json`` was written before the LaTeX-escape repair
    # landed in ``ir_builder._extract_json``: the old data contains
    # JSON-eaten control characters in place of LaTeX backslashes.
    if bv_id in ir_files:
        path = ir_files[bv_id]
        data = _repair_latex_escapes(json.loads(path.read_text(encoding="utf-8")))
        lecture_ir = LectureIR.model_validate(data)
        return lecture_ir_to_lecture_json(lecture_ir), str(path)
    if bv_id in db_summaries:
        data = _repair_latex_escapes(db_summaries[bv_id])
        return LectureJSON.model_validate(data), "sqlite:summary_json"
    raise ValueError("No LectureIR debug JSON or SQLite summary_json available for HTML-only rerender")


def _merge_visuals_from_summary(lecture: LectureJSON, summary_data: dict | None) -> LectureJSON:
    """Backfill visuals from persisted LectureJSON when IR projection is sparse.

    ``lecture_ir_to_lecture_json()`` intentionally focuses on projection logic
    and does not reproduce pipeline-time hydration such as chapter frame
    injection. The SQLite ``summary_json`` already contains those hydrated
    chapter frames, so rerender should reuse them instead of dropping images.
    """
    if not summary_data:
        return lecture
    merged = lecture.model_dump(mode="json")
    chapters = merged.get("chapters") or []
    summary_chapters = (summary_data or {}).get("chapters") or []
    summary_by_index = {
        int(ch.get("index")): ch
        for ch in summary_chapters
        if isinstance(ch, dict) and ch.get("index") is not None
    }
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        idx = chapter.get("index")
        persisted = summary_by_index.get(int(idx)) if idx is not None else None
        if not isinstance(persisted, dict):
            continue
        if not (chapter.get("frames") or []):
            chapter["frames"] = deepcopy(persisted.get("frames") or [])
    if not (merged.get("visual_evidence") or []):
        merged["visual_evidence"] = deepcopy((summary_data or {}).get("visual_evidence") or [])
    if not merged.get("cover_url"):
        merged["cover_url"] = str((summary_data or {}).get("cover_url") or "")
    return LectureJSON.model_validate(merged)


def _discover_cover_path(data_dir: Path, bv_id: str) -> Path | None:
    covers_dir = data_dir / "covers"
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = covers_dir / f"{bv_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _render_one(
    renderer: Renderer,
    css_inline: str,
    data_dir: Path,
    bv_id: str,
    ir_files: dict[str, Path],
    db_summaries: dict[str, dict],
) -> RerenderResult:
    try:
        lecture, source = _lecture_from_source(bv_id, ir_files, db_summaries)
        lecture = _merge_visuals_from_summary(lecture, db_summaries.get(bv_id))
        lecture.taxonomy = normalize_taxonomy(lecture.taxonomy)
        if lecture.taxonomy and lecture.taxonomy.tags:
            lecture.domain_tags = list(lecture.taxonomy.tags)
        cover_path = _discover_cover_path(data_dir, bv_id)
        report = renderer.render_lecture(lecture, css_inline, cover_path=cover_path)
        return RerenderResult(bv_id=bv_id, status="ok", source=source, report=report)
    except Exception as exc:  # noqa: BLE001
        logging.exception("HTML-only rerender failed: %s", bv_id)
        return RerenderResult(bv_id=bv_id, status="failed", source="-", error=str(exc))


def _print_summary(results: list[RerenderResult]) -> None:
    ok = sum(1 for result in results if result.status == "ok")
    failed = len(results) - ok
    print(f"\nHTML-only rerender summary: {ok}/{len(results)} ok, {failed} failed")
    for result in results:
        if result.status == "ok":
            print(f"  OK     {result.bv_id} [{result.source}] -> {result.report}")
        else:
            print(f"  FAILED {result.bv_id} :: {result.error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerender LectureMind HTML reports from local LectureIR debug JSON or SQLite summary_json only."
    )
    parser.add_argument("--only", nargs="+", default=[], help="Only rerender selected BV ids.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory containing *.lecture_ir.json files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed report.")
    args = parser.parse_args()

    settings = get_settings()
    debug_dir = args.debug_dir or settings.data_dir / "debug"
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

    ir_files = _discover_ir_files(debug_dir)
    db_summaries = _load_db_summaries(settings.db_path)
    report_bv_ids = _discover_report_bv_ids(settings.reports_dir)
    targets = _discover_targets(
        ir_files=ir_files,
        db_summaries=db_summaries,
        report_bv_ids=report_bv_ids,
        only=args.only,
    )
    if not targets:
        raise SystemExit("No local report sources found.")

    renderer = Renderer()
    css_inline = renderer.load_inline_css()
    results: list[RerenderResult] = []
    for index, bv_id in enumerate(targets, start=1):
        source_label = "LectureIR" if bv_id in ir_files else "SQLite"
        print(f"[{index}/{len(targets)}] {bv_id} · HTML-only rerender · {source_label}", file=sys.stderr)
        result = _render_one(renderer, css_inline, settings.data_dir, bv_id, ir_files, db_summaries)
        results.append(result)
        if result.status != "ok" and args.fail_fast:
            break

    _print_summary(results)
    raise SystemExit(1 if any(result.status != "ok" for result in results) else 0)


if __name__ == "__main__":
    main()
