"""Jinja2 renderer: LectureJSON → self-contained HTML on disk."""
from __future__ import annotations

import base64
import difflib
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from ..config import get_settings
from ..understand.length_adapt import global_visual_limit as _adaptive_global_visual_limit
from ..understand.schema import LectureJSON

logger = logging.getLogger(__name__)
# Floor; the actual limit per render scales with ``lecture.duration``
# via :func:`_adaptive_global_visual_limit` so long lectures show more
# evidence frames without leaving short lectures bloated.
_GLOBAL_VISUAL_LIMIT = 5
_CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+\-.#]*)?[ \t]*\n(.*?)```", re.DOTALL)
_TIMELINE_TS_RE = re.compile(
    r"^\s*(?P<time>(?:\d{1,2}:)?\d{1,2}:\d{2}|\d+(?:\.\d+)?\s*(?:s|秒))\s*[:：\-—]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_KATEX_REQUIRED_ASSETS = (
    "katex.min.css",
    "katex.min.js",
    "contrib/auto-render.min.js",
    "contrib/mhchem.min.js",
)
_HIGHLIGHT_REQUIRED_ASSETS = (
    "highlight.min.js",
    "atom-one-light.min.css",
)
_LONG_VIDEO_THRESHOLD_SEC = 20 * 60
# Stable jsDelivr URLs used as fallback when no vendored bundle is present.
# Keep version pinned so a CDN-side breakage doesn't silently change behaviour.
_HIGHLIGHT_CDN_VERSION = "11.10.0"
_HIGHLIGHT_CDN_JS = (
    "https://cdn.jsdelivr.net/npm/highlight.js@"
    f"{_HIGHLIGHT_CDN_VERSION}/lib/index.min.js"
)
_HIGHLIGHT_CDN_CSS = (
    "https://cdn.jsdelivr.net/npm/highlight.js@"
    f"{_HIGHLIGHT_CDN_VERSION}/styles/atom-one-light.min.css"
)


_PENDING_DOMAIN = "待归类"


def _is_long_lecture(duration: float | int) -> bool:
    return float(duration or 0.0) >= _LONG_VIDEO_THRESHOLD_SEC


def _text_key(text: str) -> str:
    return re.sub(r"[\s锛屻€傦紒锛??銆侊紱;锛?,.銆娿€嬧€溾€漒\"'锛堬級()\[\]銆愩€慭-鈥擾]+", "", str(text or "")).lower()


def _texts_heavily_overlap(left: str, right: str, *, threshold: float = 0.82) -> bool:
    left_key = _text_key(left)
    right_key = _text_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter = min(len(left_key), len(right_key))
    longer = max(len(left_key), len(right_key))
    if shorter and (left_key in right_key or right_key in left_key):
        if shorter / max(longer, 1) >= 0.72:
            return True
    return difflib.SequenceMatcher(None, left_key, right_key).ratio() >= threshold


def group_summaries_by_domain(rows: list) -> list[dict]:
    """Group ``SummaryRow`` objects into a two-level tree for the index page.

    Output schema (kept as plain dicts so the Jinja template doesn't need
    to import any dataclass)::

        [
          {
            "domain": "AI 技术",
            "count": 5,
            "needs_review": False,
            "directions": [
              {"direction": "注意力机制", "count": 3, "cards": [SummaryRow, ...]},
              {"direction": "RLHF",         "count": 2, "cards": [...]},
            ],
          },
          {
            "domain": "待归类",
            "count": 2,
            "needs_review": True,
            "directions": [
              {"direction": "",  # flat bucket — no sub-grouping
               "count": 2,
               "cards": [...]},
            ],
          },
        ]

    Sort rules (kept deterministic to simplify snapshot tests):

    * ``"待归类"`` is always pinned after classified domains when present.
    * Remaining domains are sorted by ``count`` DESC then ``domain`` ASC.
    * Within a domain, directions are sorted by ``count`` DESC then name ASC.
    * Within a direction, items are sorted by ``created_at`` DESC (empty
      strings sink to the bottom).

    A row lands in ``"待归类"`` when its persisted ``domain`` is missing,
    is the ``"其他"`` fall-back, or when ``direction`` is empty. This
    mirrors :func:`app.copilot.taxonomy.needs_review` but operates on
    the flat ``summaries`` columns so we don't have to re-parse
    ``summary_json`` for every row at render time.
    """
    pending: list = []
    buckets: dict[tuple[str, str], list] = {}

    for row in rows:
        if getattr(row, "status", "done") != "done":
            # Skip in-flight / failed rows — they're surfaced by the
            # existing /api/jobs page, not by the lecture library.
            continue
        domain = (getattr(row, "domain", None) or "").strip()
        direction = (getattr(row, "direction", None) or "").strip()
        if not domain or domain == "其他" or not direction:
            pending.append(row)
            continue
        buckets.setdefault((domain, direction), []).append(row)

    # Sort items inside each bucket by created_at DESC.
    def _sort_key_item(r):
        return (getattr(r, "created_at", "") or "")

    domain_map: dict[str, list[dict]] = {}
    for (domain, direction), items in buckets.items():
        items.sort(key=_sort_key_item, reverse=True)
        domain_map.setdefault(domain, []).append(
            {"direction": direction, "count": len(items), "cards": items}
        )

    groups: list[dict] = []
    for domain, directions in domain_map.items():
        directions.sort(key=lambda d: (-d["count"], d["direction"]))
        groups.append(
            {
                "domain": domain,
                "count": sum(d["count"] for d in directions),
                "needs_review": False,
                "directions": directions,
            }
        )
    groups.sort(key=lambda g: (-g["count"], g["domain"]))

    if pending:
        pending.sort(key=_sort_key_item, reverse=True)
        groups.append(
            {
                "domain": _PENDING_DOMAIN,
                "count": len(pending),
                "needs_review": True,
                "directions": [
                    {"direction": "", "count": len(pending), "cards": pending}
                ],
            }
        )

    return groups


def _format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_duration(seconds: float | int) -> str:
    return _format_ts(seconds)


def _bili_link(bv_id: str, ts: float) -> str:
    return f"https://www.bilibili.com/video/{bv_id}/?t={int(ts)}"


def _rich_text(value: object) -> Markup:
    if value is None:
        return Markup("")
    text = str(value)
    chunks: list[str] = []
    pos = 0
    for match in _CODE_FENCE_RE.finditer(text):
        chunks.append(_rich_plain(text[pos : match.start()]))
        language = _sanitize_code_language(match.group(1) or "")
        code = escape(match.group(2).strip("\n"))
        class_attr = f' class="language-{language}"' if language else ""
        chunks.append(f'<pre class="code-block"><code{class_attr}>{code}</code></pre>')
        pos = match.end()
    chunks.append(_rich_plain(text[pos:]))
    return Markup("".join(chunks))


def _rich_plain(text: str) -> str:
    return str(escape(text)).replace("\n", "<br>")


def _sanitize_code_language(language: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+\-.#]", "", language.strip())[:32]


def _compact_inline_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        text = "".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def _frame_url(path: str, bv_id: str, data_dir: Path | None = None) -> str:
    if not path:
        return ""
    data_dir = data_dir or get_settings().data_dir
    resolved = _resolve_frame_path(path, bv_id, data_dir)
    if resolved and resolved.exists():
        try:
            encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except OSError as exc:
            logger.warning("Failed to inline frame %s: %s", resolved, exc)
    return _relative_frame_url(path, bv_id)


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _cover_url(lecture: LectureJSON, cover_path: Path | None) -> str:
    if cover_path and cover_path.exists():
        try:
            return _image_data_url(cover_path)
        except OSError as exc:
            logger.warning("Failed to inline cover %s: %s", cover_path, exc)
    frame = next((f for f in lecture.visual_evidence if f.path), None)
    if frame is None:
        for ch in lecture.chapters:
            frame = next((f for f in ch.frames if f.path), None)
            if frame is not None:
                break
    if frame is not None:
        return _frame_url(frame.path, lecture.bv_id)
    return lecture.cover_url


def _resolve_frame_path(path: str, bv_id: str, data_dir: Path) -> Path | None:
    p = Path(path)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                p,
                data_dir / p,
                data_dir / "keyframes" / bv_id / p.name,
            ]
        )
        s = str(path).replace("\\", "/")
        if s.startswith("keyframes/"):
            candidates.append(data_dir / s)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _relative_frame_url(path: str, bv_id: str) -> str:
    p = Path(path)
    s = str(path).replace("\\", "/")
    if s.startswith("keyframes/"):
        return "../" + s
    if f"keyframes/{bv_id}/" in s:
        tail = s.split("keyframes/", 1)[-1]
        return "../keyframes/" + tail
    return f"../keyframes/{bv_id}/{p.name}"


def _index_frame_url(path: str, bv_id: str) -> str:
    if not path:
        return ""
    s = str(path).replace("\\", "/")
    if s.startswith(("data:image/", "http://", "https://")):
        return s
    return f"/keyframes/{bv_id}/{Path(path).name}"


def _summary_cover_url(row: object) -> str:
    bv_id = str(getattr(row, "bv_id", "") or "")
    data = getattr(row, "summary_json", None) or {}
    if isinstance(data, dict):
        for frame in data.get("visual_evidence") or []:
            if isinstance(frame, dict) and frame.get("path"):
                return _index_frame_url(str(frame["path"]), bv_id)
        for chapter in data.get("chapters") or []:
            if not isinstance(chapter, dict):
                continue
            for frame in chapter.get("frames") or []:
                if isinstance(frame, dict) and frame.get("path"):
                    return _index_frame_url(str(frame["path"]), bv_id)
        for key in ("cover_url", "cover"):
            value = str(data.get(key) or "").strip()
            if value:
                return value
    return str(getattr(row, "cover_url", "") or "")


def _type_recap(lecture: LectureJSON) -> dict[str, object] | None:
    units = [u for u in lecture.knowledge_units if u.title]
    if _is_long_lecture(lecture.duration) and len(lecture.chapters) >= 4:
        return None
    primary = lecture.profile.primary_type
    if primary == "technical_formula":
        selected = [u for u in units if u.type in {"formula", "code", "mechanism", "pitfall"}]
        if selected:
            return {"title": "技术复盘", "units": selected, "style": "technical-section"}
    if primary == "procedural_tutorial":
        selected = [u for u in units if u.type == "procedure"]
        if len(selected) >= 3:
            return {"title": "操作路线与检查点", "units": selected, "style": "procedure-section"}
    if primary == "conceptual_talk":
        selected = [u for u in units if u.type in {"concept", "mechanism", "example", "boundary", "pitfall"}]
        if selected:
            return {"title": "观点与边界复盘", "units": selected, "style": "conceptual-section"}
    selected = [u for u in units if u.type in {"concept", "mechanism", "example", "boundary", "pitfall"}]
    if selected:
        return {"title": "关键知识点复盘", "units": selected, "style": "generic-section"}
    return None


def _mainline_items(lecture: LectureJSON) -> list[str]:
    items = lecture.learning_path or lecture.timeline.argument_path
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        key = re.sub(r"[\s，。！？!?、；;：:,.《》“”\"'（）()\[\]【】\-—_]+", "", text).lower()
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 8:
            break
    return out


def _timeline_detail_items(
    items: list[str],
    *,
    mainline_items: list[str] | None = None,
    long_mode: bool = False,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items:
        text = _compact_inline_text(item)
        if not text:
            continue
        time = ""
        body = text
        match = _TIMELINE_TS_RE.match(text)
        if match:
            time = re.sub(r"\s+", "", match.group("time"))
            body = match.group("body").strip() or text
        if mainline_items and _texts_heavily_overlap(body, " ".join(mainline_items), threshold=0.78):
            continue
        if any(_texts_heavily_overlap(body, existing["text"], threshold=0.82) for existing in out):
            continue
        out.append({"time": time, "text": body})
        if long_mode and len(out) >= 4:
            break
    return out


def _hero_one_liner(lecture: LectureJSON) -> str:
    if lecture.core_question and _texts_heavily_overlap(
        lecture.one_liner,
        lecture.core_question,
        threshold=0.76,
    ):
        return ""
    return lecture.one_liner


def _cover_one_liner(lecture: LectureJSON, note_projection: dict[str, object]) -> str:
    one_liner = ""
    if note_projection.get("has_units"):
        front_matter = note_projection.get("front_matter") or {}
        one_liner = str(front_matter.get("one_sentence_claim") or "").strip()
        if lecture.core_question and _texts_heavily_overlap(one_liner, lecture.core_question, threshold=0.76):
            one_liner = ""
    if not one_liner:
        one_liner = _hero_one_liner(lecture)
    return one_liner


def _review_questions_to_render(lecture: LectureJSON) -> list[str]:
    study = [str(item).strip() for item in lecture.study_questions if str(item).strip()]
    review = [str(item).strip() for item in lecture.review_questions if str(item).strip()]
    if not review:
        return []
    if len(study) == len(review) and all(
        _texts_heavily_overlap(left, right) for left, right in zip(study, review)
    ):
        return []
    return review


def _format_section_ref_label(refs: list[int], total_chapters: int) -> str:
    unique_refs = sorted({int(ref) for ref in refs if int(ref) > 0})
    if not unique_refs:
        return ""
    if total_chapters > 0 and unique_refs == list(range(1, total_chapters + 1)):
        return "综合全片"
    if unique_refs == list(range(unique_refs[0], unique_refs[-1] + 1)):
        if unique_refs[0] == 1 and len(unique_refs) >= 3:
            return f"对应原视频前 {len(unique_refs)} 章"
        if len(unique_refs) >= 3:
            return f"对应原视频第 {unique_refs[0]}-{unique_refs[-1]} 章"
    if len(unique_refs) == 1:
        return f"对应原视频第 {unique_refs[0]} 章"
    return "对应原视频第 " + "、".join(str(ref) for ref in unique_refs) + " 章"


def _composition_body_sections(lecture: LectureJSON) -> list[dict[str, object]]:
    if not lecture.composition or not lecture.composition.body_sections:
        return []
    total_chapters = len(lecture.chapters)
    rendered: list[dict[str, object]] = []
    for section_index, section in enumerate(lecture.composition.body_sections, start=1):
        block_payloads: list[dict[str, object]] = []
        for block in section.outline_blocks:
            block_payloads.append(
                {
                    "id": block.id,
                    "ordinal": f"{section_index}.{int(block.ordinal)}",
                    "title": block.title,
                    "lead": block.lead,
                    "paragraphs": list(block.paragraphs),
                    "source_timestamps": list(block.source_timestamps),
                }
            )
        rendered.append(
            {
                "id": section.id,
                "title": section.title,
                "section_role": section.section_role,
                "summary": section.summary,
                "section_ref_label": _format_section_ref_label(
                    list(section.source_chapter_refs),
                    total_chapters,
                ),
                "supporting_visuals": [
                    visual.model_dump(mode="json") for visual in section.supporting_visuals
                ],
                "outline_blocks": block_payloads,
            }
        )
    return rendered


def _compat_note_projection_from_composition(lecture: LectureJSON) -> dict[str, object]:
    sections = _composition_body_sections(lecture)
    takeaways = list(getattr(lecture.composition, "key_takeaways_top", []) or [])
    reading_map = [str(sec.get("title") or "").strip() for sec in sections if str(sec.get("title") or "").strip()]
    teaching_units: list[dict[str, object]] = []
    for index, sec in enumerate(sections, start=1):
        teaching_units.append(
            {
                "id": sec["id"],
                "ordinal": str(index),
                "title": sec["title"],
                "unit_role": sec["section_role"],
                "core_message": sec["summary"],
                "transition_from_previous": "",
                "primary_timestamp": 0.0,
                "section_ref_label": sec["section_ref_label"],
                "content_blocks": [
                    {
                        "id": block["id"],
                        "ordinal": block["ordinal"],
                        "show_ordinal": True,
                        "show_head": True,
                        "title": block["title"],
                        "lead": block["lead"],
                        "paragraphs": list(block["paragraphs"]),
                        "source_timestamps": list(block["source_timestamps"]),
                    }
                    for block in sec["outline_blocks"]
                ],
                "visual_slots": list(sec["supporting_visuals"]),
            }
        )
    return {
        "mode": "compat",
        "has_units": bool(teaching_units),
        "audience_fit": getattr(lecture.composition, "audience_fit", ""),
        "front_matter": {
            "one_sentence_claim": _hero_one_liner(lecture),
            "reader_orientation": lecture.core_question,
            "takeaways_top": takeaways,
            "reading_map": reading_map,
            "reader_prerequisites": [],
            "suitable_for": [],
            "not_suitable_for": [],
        },
        "teaching_units": teaching_units,
        "back_matter": {
            "boundary_and_risks": [],
            "term_quick_ref": [],
            "source_index_entrypoints": [],
            "appendices": [],
            "transfer_and_next_steps": list(getattr(lecture, "review_questions", [])[:4]),
        },
    }


def _lecture_note_projection(lecture: LectureJSON) -> dict[str, object]:
    note = lecture.lecture_note_ir
    if note is None:
        return _compat_note_projection_from_composition(lecture)
    total_chapters = len(lecture.chapters)
    evidence_refs_by_node: dict[str, list[str]] = {}
    evidence_objects_by_node: dict[str, list] = {}
    if lecture.evidence_index is not None:
        for obj in lecture.evidence_index.evidence_objects:
            if not obj.evidence_id:
                continue
            for node_id in obj.note_node_ids:
                if not node_id:
                    continue
                evidence_refs_by_node.setdefault(node_id, [])
                if obj.evidence_id not in evidence_refs_by_node[node_id]:
                    evidence_refs_by_node[node_id].append(obj.evidence_id)
                evidence_objects_by_node.setdefault(node_id, []).append(obj)
    teaching_units: list[dict[str, object]] = []
    for unit_index, unit in enumerate(note.body.teaching_units, start=1):
        block_count = len(unit.content_blocks)
        content_blocks: list[dict[str, object]] = []
        primary_timestamp = 0.0
        for block_index, block in enumerate(unit.content_blocks, start=1):
            block_evidence_refs = list(block.evidence_refs)
            for evidence_id in evidence_refs_by_node.get(block.block_id, []):
                if evidence_id not in block_evidence_refs:
                    block_evidence_refs.append(evidence_id)
            timestamps = list(block.source_timestamps)
            if not primary_timestamp and timestamps:
                primary_timestamp = float(timestamps[0])
            show_head = block_count > 1 and bool(block.title or timestamps or block_evidence_refs)
            content_blocks.append(
                {
                    "id": block.block_id,
                    "ordinal": f"{unit_index}.{block_index}",
                    "show_ordinal": block_count > 1,
                    "show_head": show_head,
                    "title": _fallback_block_title(
                        block.title,
                        block_role=block.block_role,
                        unit_role=unit.unit_role,
                        block_index=block_index,
                    ),
                    "lead": block.lead,
                    "paragraphs": list(block.paragraphs),
                    "source_timestamps": timestamps,
                    "evidence_refs": block_evidence_refs,
                }
            )
        unit_evidence_refs = list(unit.evidence_refs)
        for evidence_id in evidence_refs_by_node.get(unit.unit_id, []):
            if evidence_id not in unit_evidence_refs:
                unit_evidence_refs.append(evidence_id)
        visual_slots = [
            _project_visual_slot(
                {
                    "slot_id": slot.slot_id,
                    "visual_role": slot.visual_role,
                    "path": slot.source_paths[0] if slot.source_paths else "",
                    "caption": slot.caption,
                    "title": slot.title,
                    "ts": float(slot.ts or 0.0),
                    "evidence_refs": list(slot.evidence_refs),
                }
            )
            for slot in unit.visual_slots
            if slot.source_paths
        ]
        if not visual_slots:
            visual_slots = _fallback_visual_slots_for_unit(
                lecture,
                unit_id=unit.unit_id,
                source_chapter_refs=list(unit.source_chapter_refs),
                evidence_objects=evidence_objects_by_node.get(unit.unit_id, []),
            )
        teaching_units.append(
            {
                "id": unit.unit_id,
                "ordinal": unit.ordinal or str(unit_index),
                "title": unit.title,
                "unit_role": unit.unit_role,
                "core_message": unit.core_message,
                "transition_from_previous": unit.transition_from_previous,
                "primary_timestamp": primary_timestamp,
                "evidence_refs": unit_evidence_refs,
                "source_chapter_refs": list(unit.source_chapter_refs),
                "section_ref_label": _format_section_ref_label(list(unit.source_chapter_refs), total_chapters),
                "content_blocks": content_blocks,
                "visual_slots": visual_slots,
            }
        )
    audience_fit = ""
    if lecture.composition and lecture.composition.audience_fit:
        audience_fit = lecture.composition.audience_fit
    elif note.front_matter.suitable_for:
        audience_fit = "适合：" + "；".join(note.front_matter.suitable_for[:2])
    return {
        "mode": "lecture_note_ir",
        "has_units": bool(teaching_units),
        "audience_fit": audience_fit,
        "front_matter": {
            "one_sentence_claim": note.front_matter.one_sentence_claim,
            "reader_orientation": note.front_matter.reader_orientation,
            "takeaways_top": list(note.front_matter.takeaways_top),
            "reading_map": list(note.front_matter.reading_map),
            "reader_prerequisites": list(note.front_matter.reader_prerequisites),
            "suitable_for": list(note.front_matter.suitable_for),
            "not_suitable_for": list(note.front_matter.not_suitable_for),
        },
        "teaching_units": teaching_units,
        "back_matter": {
            "boundary_and_risks": list(note.back_matter.boundary_and_risks),
            "term_quick_ref": list(note.back_matter.term_quick_ref),
            "source_index_entrypoints": list(note.back_matter.source_index_entrypoints),
            "appendices": _reader_appendices_view(list(note.back_matter.appendices)),
            "transfer_and_next_steps": list(note.back_matter.transfer_and_next_steps),
        },
    }


def _fallback_visual_slots_for_unit(
    lecture: LectureJSON,
    *,
    unit_id: str,
    source_chapter_refs: list[int],
    evidence_objects: list,
) -> list[dict[str, object]]:
    visuals: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for obj in evidence_objects:
        path = str(getattr(obj, "path", "") or "").strip()
        if not path or path in seen_paths:
            continue
        kind = str(getattr(obj, "kind", "") or "").strip()
        if kind and kind != "frame_evidence":
            continue
        seen_paths.add(path)
        visuals.append(
            _project_visual_slot(
                {
                    "slot_id": f"{unit_id}-fallback-evidence-{len(visuals) + 1}",
                    "visual_role": "keyframe_explainer",
                    "path": path,
                    "caption": str(getattr(obj, "summary", "") or "").strip(),
                    "title": str(getattr(obj, "title", "") or "").strip(),
                    "ts": float(getattr(obj, "ts", 0.0) or 0.0),
                    "evidence_refs": [str(getattr(obj, "evidence_id", "") or "").strip()] if getattr(obj, "evidence_id", "") else [],
                }
            )
        )
    if visuals:
        visuals.sort(key=_visual_slot_rank, reverse=True)
        return visuals[:1]
    if not source_chapter_refs:
        return visuals
    chapter_frames: list[dict[str, object]] = []
    for chapter in lecture.chapters:
        if chapter.index not in source_chapter_refs:
            continue
        for frame_index, frame in enumerate(chapter.frames, start=1):
            path = str(frame.path or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            chapter_frames.append(
                _project_visual_slot(
                    {
                        "slot_id": f"{unit_id}-chapter-frame-{chapter.index}-{frame_index}",
                        "visual_role": "keyframe_explainer",
                        "path": path,
                        "caption": frame.insight or frame.selected_reason or "",
                        "title": frame.caption or "",
                        "ts": float(frame.ts or 0.0),
                        "evidence_refs": [],
                    }
                )
            )
    chapter_frames.sort(key=_visual_slot_rank, reverse=True)
    return chapter_frames[:1]


def _project_visual_slot(raw: dict[str, object]) -> dict[str, object]:
    title = _clean_visual_text(raw.get("title"))
    caption = _clean_visual_text(raw.get("caption"))
    if title and caption and _texts_heavily_overlap(title, caption, threshold=0.76):
        title = ""
    return {
        "slot_id": str(raw.get("slot_id") or "").strip(),
        "visual_role": str(raw.get("visual_role") or "").strip(),
        "path": str(raw.get("path") or "").strip(),
        "caption": caption,
        "title": title,
        "ts": float(raw.get("ts") or 0.0),
        "evidence_refs": [str(item).strip() for item in list(raw.get("evidence_refs") or []) if str(item).strip()],
    }


def _clean_visual_text(value: object, *, max_len: int = 88) -> str:
    text = _compact_inline_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if '"ocr_text"' in lowered or lowered.startswith("{") or lowered.startswith("["):
        return ""
    noisy_tokens = (
        "icloud",
        "macintosh",
        "downloads",
        "airdrop",
        "finder",
        "标签",
        "最近使用",
        "共享位置",
        "文稿",
        "收藏",
        "个人收藏",
        "隔空投送",
    )
    if any(token in lowered for token in noisy_tokens):
        return ""
    punctuation_count = sum(text.count(ch) for ch in '{}[]":')
    if punctuation_count >= 4:
        return ""
    if len(text) > 52 and text.count(" ") >= 6 and not any(ch in text for ch in "。！？；;：:"):
        return ""
    if len(text) > 30 and text.count(" ") >= 4 and not any(ch in text for ch in "。！？；;：:"):
        return ""
    if len(text) > max_len:
        sentence = re.split(r"[。！？；;]", text, maxsplit=1)[0].strip()
        if 8 <= len(sentence) <= max_len:
            text = sentence
        else:
            text = text[:max_len].rstrip("，,、；;:： ")
    return text + ("…" if len(text) == max_len else "")


def _visual_slot_rank(slot: dict[str, object]) -> tuple[float, float]:
    score = 0.0
    if slot.get("caption"):
        score += 2.0
    if slot.get("title"):
        score += 1.0
    if slot.get("evidence_refs"):
        score += 0.25
    return (score, -float(slot.get("ts") or 0.0))


def _source_index(lecture: LectureJSON) -> dict[str, object]:
    if not lecture.composition or not lecture.composition.source_index:
        return {}
    return dict(lecture.composition.source_index)


def _reader_appendices_view(items: list[object]) -> list[dict[str, str]]:
    views: list[dict[str, str]] = []
    kind_labels = {
        "boundary_review": "边界复盘",
        "related_material": "相关材料",
        "reference": "参考资料",
    }
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                views.append({"title": "补充说明", "summary": text, "kind_label": ""})
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("label") or "").strip()
        summary = str(
            item.get("summary")
            or item.get("explanation")
            or item.get("note")
            or item.get("reason")
            or item.get("text")
            or ""
        ).strip()
        if not summary:
            continue
        kind = str(item.get("kind") or "").strip()
        views.append(
            {
                "title": title,
                "summary": summary,
                "kind_label": kind_labels.get(kind, kind.replace("_", " ").strip()),
            }
        )
    return views


def _fallback_block_title(
    explicit_title: str,
    *,
    block_role: str = "",
    unit_role: str = "",
    block_index: int = 1,
) -> str:
    title = str(explicit_title or "").strip()
    if title:
        return title
    role = str(block_role or "").strip().lower()
    if not role:
        role = str(unit_role or "").strip().lower()
    labels = {
        "claim": "核心内容",
        "concept": "核心内容",
        "problem": "问题与判断",
        "mechanism": "原理说明",
        "process": "步骤与方法",
        "case": "案例展开",
        "example": "案例展开",
        "boundary": "边界与风险",
        "synthesis": "小结",
        "summary": "小结",
    }
    if role in labels:
        return labels[role]
    if block_index == 1:
        return "核心内容"
    return "展开说明"


def _compat_evidence_projection_from_source_index(lecture: LectureJSON) -> dict[str, object]:
    source_index = _source_index(lecture)
    return {
        "mode": "compat",
        "has_items": bool(source_index),
        "source_index_view": list(source_index.get("source_index_view", [])),
        "chapter_map": list(source_index.get("chapter_map", [])),
        "evidence_quotes": list(source_index.get("evidence_quotes", [])),
        "term_index": list(source_index.get("term_index", [])),
        "glossary_view": [],
        "appendix_view": list(source_index.get("tool_appendix", [])),
        "tool_appendix": list(source_index.get("tool_appendix", [])),
    }


def _evidence_projection(lecture: LectureJSON) -> dict[str, object]:
    evidence = lecture.evidence_index
    if evidence is None:
        return _compat_evidence_projection_from_source_index(lecture)
    views = dict(evidence.projection_views)
    has_items = any(bool(views.get(key)) for key in ("source_index_view", "chapter_map", "evidence_quotes", "term_index", "glossary_view", "appendix_view"))
    return {
        "mode": "evidence_index",
        "has_items": has_items,
        "source_index_view": list(views.get("source_index_view", [])),
        "chapter_map": list(views.get("chapter_map", [])),
        "evidence_quotes": list(views.get("evidence_quotes", [])),
        "term_index": list(views.get("term_index", [])),
        "glossary_view": list(views.get("glossary_view", [])),
        "appendix_view": list(views.get("appendix_view", [])),
        "tool_appendix": list(views.get("tool_appendix", [])),
    }


def _ensure_katex_assets(reports_dir: Path) -> bool:
    vendor_dir = Path(__file__).parent.parent / "static" / "vendor" / "katex"
    missing = [asset for asset in _KATEX_REQUIRED_ASSETS if not (vendor_dir / asset).exists()]
    if missing:
        logger.warning("KaTeX assets are missing: %s", ", ".join(missing))
        return False
    target_dir = reports_dir / "assets" / "katex"
    try:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendor_dir, target_dir, dirs_exist_ok=True)
    except OSError as exc:
        logger.warning("Failed to copy KaTeX assets to %s: %s", target_dir, exc)
        return False
    missing_target = [asset for asset in _KATEX_REQUIRED_ASSETS if not (target_dir / asset).exists()]
    if missing_target:
        logger.warning("Copied KaTeX assets are incomplete: %s", ", ".join(missing_target))
        return False
    return True


def _ensure_highlight_assets(reports_dir: Path) -> bool:
    """Copy a vendored highlight.js bundle if present.

    The renderer is happy to use either:

    * **Vendored** files in ``app/static/vendor/highlight/`` (e.g.
      ``highlight.min.js`` + ``atom-one-light.min.css``); these are
      copied to ``reports_dir/assets/highlight/`` and the template
      points at the local copy — best for offline reading.
    * **CDN fallback** (jsDelivr); used automatically when no vendored
      bundle is present.

    Returns ``True`` only when the local copy succeeded so the caller
    knows whether to enable the CDN fallback.
    """
    vendor_dir = Path(__file__).parent.parent / "static" / "vendor" / "highlight"
    if not vendor_dir.exists():
        # Quiet by design: missing vendor is the common case and the
        # template falls back to the CDN.
        return False
    missing = [asset for asset in _HIGHLIGHT_REQUIRED_ASSETS if not (vendor_dir / asset).exists()]
    if missing:
        logger.info("highlight.js vendored partially (%s); using CDN fallback", ", ".join(missing))
        return False
    target_dir = reports_dir / "assets" / "highlight"
    try:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendor_dir, target_dir, dirs_exist_ok=True)
    except OSError as exc:
        logger.warning("Failed to copy highlight.js assets to %s: %s", target_dir, exc)
        return False
    missing_target = [asset for asset in _HIGHLIGHT_REQUIRED_ASSETS if not (target_dir / asset).exists()]
    if missing_target:
        logger.warning("Copied highlight.js assets incomplete: %s", ", ".join(missing_target))
        return False
    return True


def _global_visual_evidence(lecture: LectureJSON) -> list:
    chapter_paths = {
        str(frame.path or "").strip()
        for chapter in lecture.chapters
        for frame in chapter.frames
        if str(frame.path or "").strip()
    }
    frames = []
    seen: set[str] = set()
    for frame in lecture.visual_evidence:
        if not frame.path or frame.path in seen or frame.path in chapter_paths:
            continue
        seen.add(frame.path)
        frames.append(frame)
    limit = max(_GLOBAL_VISUAL_LIMIT, _adaptive_global_visual_limit(lecture.duration))
    if len(frames) > limit:
        preferred = {"diagram", "formula", "code", "table", "slide_text"}
        frames = sorted(
            frames,
            key=lambda f: (
                f.importance_score,
                1 if f.visual_type in preferred else 0,
                1 if f.ocr_text or f.caption else 0,
            ),
            reverse=True,
        )[:limit]
    return sorted(frames, key=lambda f: f.ts)


class Renderer:
    def __init__(self) -> None:
        self._settings = get_settings()
        templates_dir = Path(__file__).parent / "templates"
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html", "j2"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["fmt_ts"] = _format_ts
        self._env.filters["fmt_dur"] = _format_duration
        self._env.filters["rich_text"] = _rich_text

    def render_lecture(
        self, lecture: LectureJSON, css_inline: str, cover_path: Path | None = None
    ) -> Path:
        tpl = self._env.get_template("lecture.html.j2")
        katex_assets_available = _ensure_katex_assets(self._settings.reports_dir)
        highlight_choice = (
            getattr(self._settings, "lecture_code_highlighter", "highlight.js") or ""
        ).strip().lower()
        highlight_assets_available = False
        highlight_cdn_url = ""
        highlight_cdn_css = ""
        if highlight_choice in {"highlight.js", "hljs", "highlightjs"}:
            highlight_assets_available = _ensure_highlight_assets(self._settings.reports_dir)
            if not highlight_assets_available:
                # Fall back to CDN so the lecture still gets coloured on
                # readers with internet access.
                highlight_cdn_url = _HIGHLIGHT_CDN_JS
                highlight_cdn_css = _HIGHLIGHT_CDN_CSS
        # Build a ``frame.path → global frame_id`` map that matches
        # Stage-3's ``_flat_frames`` so the Copilot panel's [F7]
        # anchors resolve to the exact DOM nodes (D28).
        from app.copilot.tools import _flat_frames  # local import avoids cycles

        frame_id_map = {
            fr.path: idx
            for idx, (fr, _chapter_idx) in enumerate(_flat_frames(lecture), start=1)
            if fr.path
        }
        mainline_items = _mainline_items(lecture)
        long_mode = _is_long_lecture(lecture.duration)
        note_projection = _lecture_note_projection(lecture)
        evidence_projection = _evidence_projection(lecture)
        html = tpl.render(
            lecture=lecture,
            hero_one_liner=_hero_one_liner(lecture),
            cover_one_liner=_cover_one_liner(lecture, note_projection),
            review_questions_to_render=_review_questions_to_render(lecture),
            composition_body_sections=_composition_body_sections(lecture),
            source_index=_source_index(lecture),
            note_projection=note_projection,
            evidence_projection=evidence_projection,
            summary_mode=lecture.composition.summary_mode,
            css_inline=css_inline,
            copilot_css_inline=self.load_copilot_css(),
            frame_id_map=frame_id_map,
            cover_url=_cover_url(lecture, cover_path),
            type_recap=_type_recap(lecture),
            mainline_items=mainline_items,
            timeline_turning_points=_timeline_detail_items(
                lecture.timeline.turning_points,
                mainline_items=mainline_items,
                long_mode=long_mode,
            ),
            timeline_state_evolution=_timeline_detail_items(
                lecture.timeline.state_evolution,
                mainline_items=mainline_items,
                long_mode=long_mode,
            ),
            global_visual_evidence=_global_visual_evidence(lecture),
            katex_assets_available=katex_assets_available,
            highlight_assets_available=highlight_assets_available,
            highlight_cdn_url=highlight_cdn_url,
            highlight_cdn_css=highlight_cdn_css,
            bili_link=_bili_link,
            frame_url=lambda p: _frame_url(p, lecture.bv_id, self._settings.data_dir),
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        out_path = self._settings.reports_dir / f"{lecture.bv_id}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Wrote lecture HTML: %s", out_path)
        return out_path

    def render_index(self, summaries: list, css_inline: str) -> str:
        from app.copilot.taxonomy import DOMAIN_WHITELIST

        tpl = self._env.get_template("index.html.j2")
        groups = group_summaries_by_domain(summaries)
        return tpl.render(
            summaries=summaries,
            groups=groups,
            domain_whitelist=list(DOMAIN_WHITELIST),
            css_inline=css_inline,
            index_cover_url=_summary_cover_url,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def load_copilot_css(self) -> str:
        css_path = Path(__file__).parent.parent / "static" / "copilot.css"
        if not css_path.exists():
            return ""
        try:
            return css_path.read_text(encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            logger.warning("Failed to read copilot.css: %s", exc)
            return ""

    def load_inline_css(self) -> str:
        css_path = Path(__file__).parent.parent / "static" / "lecture.css"
        if not css_path.exists():
            return ""
        return css_path.read_text(encoding="utf-8")
