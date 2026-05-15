from __future__ import annotations

import re
from typing import Any

from .evidence_index import build_evidence_index
from .ir import LectureIR
from .lecture_blueprint import build_lecture_blueprint
from .schema import (
    ContentBlock,
    LectureJSON,
    LectureNoteBackMatter,
    LectureNoteBody,
    LectureNoteFrontMatter,
    LectureNoteIR,
    TeachingUnit,
    VisualSlot,
)


def compile_lecture(lecture_ir: LectureIR) -> LectureJSON:
    from .composition import compose_from_lecture_json
    from .ir_builder import project_legacy_lecture_json

    materials = material_normalization(lecture_ir)
    reading = semantic_reading(materials)
    blueprint = build_lecture_blueprint(materials["legacy_lecture"], reading)
    note_ir = note_composition(materials["legacy_lecture"], blueprint, reading)
    evidence_index = build_evidence_index(materials["legacy_lecture"], blueprint, note_ir)

    lecture = project_legacy_lecture_json(lecture_ir)
    lecture.lecture_blueprint = blueprint
    lecture.lecture_note_ir = note_ir
    lecture.evidence_index = evidence_index
    lecture.composition = compose_from_lecture_json(lecture)
    return lecture


def material_normalization(lecture_ir: LectureIR) -> dict[str, Any]:
    from .ir_builder import project_legacy_lecture_json

    legacy_lecture = project_legacy_lecture_json(lecture_ir)
    return {
        "lecture_ir": lecture_ir,
        "legacy_lecture": legacy_lecture,
        "chapters": legacy_lecture.chapters,
        "glossary": legacy_lecture.glossary,
    }


def semantic_reading(materials: dict[str, Any]) -> dict[str, str]:
    lecture = materials["legacy_lecture"]
    primary_type = materials["lecture_ir"].profile.primary_type
    semantic_portrait_map = {
        "procedural_tutorial": "流程教学",
        "technical_formula": "技术讲解",
        "conceptual_talk": "概念讲解",
        "generic_lecture": "综合讲座",
    }
    semantic_portrait = semantic_portrait_map.get(primary_type, "综合讲座")
    if lecture.composition.summary_mode in {"handout", "segmented_handout"} or lecture.duration >= 1800:
        reading_burden = "high"
    elif lecture.duration >= 900 or len(lecture.chapters) >= 4:
        reading_burden = "medium"
    else:
        reading_burden = "low"
    if any(ch.process_steps for ch in lecture.chapters):
        center = "process"
    elif any(ch.key_takeaways for ch in lecture.chapters):
        center = "claim"
    else:
        center = "mechanism"
    return {
        "semantic_portrait": semantic_portrait,
        "reading_burden": reading_burden,
        "teaching_center_of_gravity": center,
    }


def note_composition(
    lecture: LectureJSON,
    blueprint: Any,
    semantic_reading: dict[str, str],
) -> LectureNoteIR:
    teaching_units: list[TeachingUnit] = []
    for chapter, plan in zip(lecture.chapters, blueprint.body_unit_plan, strict=False):
        content_blocks = _build_content_blocks(chapter, semantic_reading)
        teaching_units.append(
            TeachingUnit(
                unit_id=plan.unit_id,
                ordinal=str(chapter.index),
                title=plan.title or chapter.title,
                teaching_goal=plan.teaching_goal,
                unit_role=plan.unit_role,
                core_message=_dedupe_unit_core_message(plan.core_message, content_blocks),
                transition_from_previous=plan.transition_from_previous,
                content_blocks=content_blocks,
                visual_slots=_build_visual_slots(chapter),
                evidence_refs=_build_unit_evidence_refs(chapter),
                source_chapter_refs=[chapter.index],
            )
        )

    return LectureNoteIR(
        front_matter=LectureNoteFrontMatter(
            one_sentence_claim=blueprint.front_matter_plan.one_sentence_claim,
            reader_orientation=blueprint.front_matter_plan.reader_orientation or lecture.core_question,
            takeaways_top=list(blueprint.front_matter_plan.takeaways_top),
            reading_map=[unit.title for unit in teaching_units],
            reader_prerequisites=list(blueprint.front_matter_plan.reader_prerequisites),
            suitable_for=list(blueprint.front_matter_plan.suitable_for),
            not_suitable_for=list(blueprint.front_matter_plan.not_suitable_for),
        ),
        body=LectureNoteBody(teaching_units=teaching_units),
        back_matter=LectureNoteBackMatter(
            boundary_and_risks=list(blueprint.back_matter_plan.boundary_and_risks),
            term_quick_ref=[
                {"term": item.term, "explanation": item.explanation, "ts": item.ts}
                for item in lecture.glossary[:8]
            ],
            source_index_entrypoints=list(blueprint.back_matter_plan.source_index_entrypoints),
            appendices=list(blueprint.back_matter_plan.appendices),
            transfer_and_next_steps=list(blueprint.back_matter_plan.transfer_and_next_steps)
            or list(lecture.review_questions[:4] or lecture.study_questions[:4]),
        ),
    )


def _build_content_blocks(chapter: Any, semantic_reading: dict[str, str]) -> list[ContentBlock]:
    content_kind = _classify_content_kind(chapter, semantic_reading)
    if content_kind == "conceptual":
        block_specs = _build_conceptual_block_specs(chapter)
    elif content_kind == "procedural":
        block_specs = _build_procedural_block_specs(chapter)
    else:
        block_specs = _build_technical_block_specs(chapter)

    if not block_specs:
        block_specs = [
            {
                "title": "",
                "block_role": "claim",
                "paragraphs": [chapter.learning_goal or chapter.title],
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        ]

    if len(block_specs) == 1:
        block_specs[0]["title"] = ""

    return [
        ContentBlock(
            block_id=f"teaching-unit-{chapter.index}-block-{index}",
            title=spec.get("title", ""),
            block_role=spec.get("block_role", ""),
            paragraphs=list(spec.get("paragraphs", [])),
            source_chapter_refs=[chapter.index],
            source_timestamps=list(spec.get("source_timestamps", [float(chapter.start)])),
            evidence_refs=list(spec.get("evidence_refs", [])),
        )
        for index, spec in enumerate(block_specs, start=1)
        if spec.get("paragraphs")
    ]


def _classify_content_kind(chapter: Any, semantic_reading: dict[str, str]) -> str:
    semantic_portrait = semantic_reading.get("semantic_portrait")
    if semantic_portrait == "流程教学":
        return "procedural"
    if _looks_procedural_chapter(chapter):
        return "procedural"
    if semantic_portrait == "概念讲解":
        return "conceptual"
    if _looks_technical_chapter(chapter):
        return "technical"
    return "conceptual"


def _build_conceptual_block_specs(chapter: Any) -> list[dict[str, Any]]:
    main_paragraphs = _merge_distinct_paragraphs(
        [chapter.summary],
        chapter.teaching_notes,
        chapter.key_takeaways[:1],
    )
    inline_boundary = _build_inline_boundary_paragraph(chapter, main_paragraphs)
    if inline_boundary:
        main_paragraphs.append(inline_boundary)

    blocks: list[dict[str, Any]] = []
    if main_paragraphs:
        blocks.append(
            {
                "title": "",
                "block_role": "claim",
                "paragraphs": main_paragraphs,
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": _chapter_quote_evidence_refs(chapter) or [f"chapter-{chapter.index}"],
            }
        )
    if _should_surface_boundary_block(chapter, main_paragraphs):
        blocks.append(
            {
                "title": "边界与风险",
                "block_role": "boundary",
                "paragraphs": _merge_distinct_paragraphs(chapter.pitfalls),
                "source_timestamps": [float(chapter.end)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        )
    return blocks


def _build_procedural_block_specs(chapter: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    overview = _merge_distinct_paragraphs([chapter.summary], chapter.teaching_notes)
    inline_boundary = _build_inline_boundary_paragraph(chapter, overview)
    if inline_boundary:
        overview.append(inline_boundary)

    if overview:
        blocks.append(
            {
                "title": "",
                "block_role": "claim",
                "paragraphs": overview,
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": _chapter_quote_evidence_refs(chapter) or [f"chapter-{chapter.index}"],
            }
        )
    if _should_surface_process_block(chapter, "procedural", overview):
        blocks.append(
            {
                "title": "步骤",
                "block_role": "process",
                "paragraphs": _merge_distinct_paragraphs(chapter.process_steps),
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": [
                    f"procedure-{chapter.index}-{idx}"
                    for idx, _ in enumerate(chapter.process_steps, start=1)
                ],
            }
        )
    elif chapter.key_takeaways:
        blocks.append(
            {
                "title": "",
                "block_role": "synthesis",
                "paragraphs": _merge_distinct_paragraphs(chapter.key_takeaways),
                "source_timestamps": [float(chapter.end)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        )
    if _should_surface_boundary_block(chapter, overview, chapter.process_steps):
        blocks.append(
            {
                "title": "容易出错的地方",
                "block_role": "boundary",
                "paragraphs": _merge_distinct_paragraphs(chapter.pitfalls),
                "source_timestamps": [float(chapter.end)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        )
    return blocks


def _build_technical_block_specs(chapter: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    overview = _merge_distinct_paragraphs([chapter.summary])
    mechanism = _merge_distinct_paragraphs(chapter.teaching_notes)
    takeaways = _merge_distinct_paragraphs(chapter.key_takeaways)
    inline_boundary = _build_inline_boundary_paragraph(chapter, overview, mechanism, takeaways)
    if inline_boundary:
        mechanism = [*mechanism, inline_boundary]

    if overview:
        blocks.append(
            {
                "title": "",
                "block_role": "claim",
                "paragraphs": overview,
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        )
    if mechanism:
        blocks.append(
            {
                "title": "关键机制" if blocks else "",
                "block_role": "mechanism",
                "paragraphs": mechanism,
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": _chapter_quote_evidence_refs(chapter) or [f"chapter-{chapter.index}"],
            }
        )
    if _should_surface_process_block(chapter, "technical", overview, mechanism):
        blocks.append(
            {
                "title": "计算流程",
                "block_role": "process",
                "paragraphs": _merge_distinct_paragraphs(chapter.process_steps),
                "source_timestamps": [float(chapter.start)],
                "evidence_refs": [
                    f"procedure-{chapter.index}-{idx}"
                    for idx, _ in enumerate(chapter.process_steps, start=1)
                ],
            }
        )
    elif takeaways and not _is_redundant_against_group(takeaways, overview, mechanism):
        blocks.append(
            {
                "title": "结论",
                "block_role": "synthesis",
                "paragraphs": takeaways,
                "source_timestamps": [float(chapter.end)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        )
    if _should_surface_boundary_block(chapter, overview, mechanism, takeaways, chapter.process_steps):
        blocks.append(
            {
                "title": "实现边界",
                "block_role": "boundary",
                "paragraphs": _merge_distinct_paragraphs(chapter.pitfalls),
                "source_timestamps": [float(chapter.end)],
                "evidence_refs": [f"chapter-{chapter.index}"],
            }
        )
    return blocks


def _dedupe_unit_core_message(core_message: str, content_blocks: list[ContentBlock]) -> str:
    if not core_message or not content_blocks:
        return core_message
    first_block_text = " ".join(content_blocks[0].paragraphs[:2])
    if _texts_highly_overlap(core_message, first_block_text):
        return ""
    return core_message


def _looks_procedural_chapter(chapter: Any) -> bool:
    steps = _merge_distinct_paragraphs(chapter.process_steps)
    if len(steps) < 2:
        return False
    notes = _merge_distinct_paragraphs(chapter.teaching_notes)
    text = _normalize_text(_chapter_text_blob(chapter))
    strong_hint_hits = sum(
        1
        for hint in ("步骤", "流程", "操作", "实操", "演示", "执行", "搭建")
        if _normalize_text(hint) in text
    )
    if strong_hint_hits:
        return True
    if len(steps) >= 3 and len(notes) <= 1:
        return True
    weak_hint_hits = sum(
        1
        for hint in ("检索", "查找", "读取", "如何")
        if _normalize_text(hint) in text
    )
    return weak_hint_hits >= 2 and len(notes) <= 1


def _looks_technical_chapter(chapter: Any) -> bool:
    text = _normalize_text(_chapter_text_blob(chapter))
    return any(
        _normalize_text(hint) in text
        for hint in ("实现", "优化", "原理", "机制", "架构", "代码", "公式", "细节")
    )


def _chapter_text_blob(chapter: Any) -> str:
    return " ".join(
        part
        for part in [chapter.title, chapter.summary, chapter.learning_goal, *list(chapter.teaching_notes[:2])]
        if part
    )


def _should_surface_process_block(chapter: Any, content_kind: str, *existing_groups: list[str]) -> bool:
    steps = _merge_distinct_paragraphs(chapter.process_steps)
    if len(steps) < 2:
        return False
    if _is_redundant_against_group(steps, *existing_groups):
        return False
    if content_kind == "procedural":
        return True
    return _looks_procedural_chapter(chapter) and len(steps) >= 3


def _build_inline_boundary_paragraph(chapter: Any, *existing_groups: list[str]) -> str:
    pitfalls = _merge_distinct_paragraphs(chapter.pitfalls)
    if not pitfalls:
        return ""
    if _is_redundant_against_group(pitfalls, *existing_groups):
        return ""
    if _should_surface_boundary_block(chapter, *existing_groups):
        return ""
    return "注意：" + "；".join(pitfalls[:2])


def _should_surface_boundary_block(chapter: Any, *existing_groups: list[str]) -> bool:
    pitfalls = _merge_distinct_paragraphs(chapter.pitfalls)
    if not pitfalls:
        return False
    if _is_redundant_against_group(pitfalls, *existing_groups):
        return False
    text = _normalize_text(_chapter_text_blob(chapter))
    explicit_boundary_focus = any(
        _normalize_text(hint) in text
        for hint in ("风险", "边界", "误区", "坑", "陷阱", "局限", "注意事项")
    )
    return explicit_boundary_focus or len(pitfalls) >= 3


def _chapter_quote_evidence_refs(chapter: Any) -> list[str]:
    return [f"quote-{chapter.index}-{idx}" for idx, _ in enumerate(chapter.points, start=1)]


def _merge_distinct_paragraphs(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for paragraph in group:
            text = (paragraph or "").strip()
            if not text:
                continue
            if any(_texts_highly_overlap(text, existing) for existing in merged):
                continue
            merged.append(text)
    return merged


def _is_redundant_against_group(paragraphs: list[str], *groups: list[str]) -> bool:
    if not paragraphs:
        return False
    existing = _merge_distinct_paragraphs(*groups)
    if not existing:
        return False
    return all(any(_texts_highly_overlap(item, candidate) for candidate in existing) for item in paragraphs)


def _texts_highly_overlap(left: str, right: str) -> bool:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return False
    shorter, longer = sorted((left_norm, right_norm), key=len)
    if shorter == longer:
        return True
    if len(shorter) >= 8 and shorter in longer:
        return True
    shorter_chars = set(shorter)
    longer_chars = set(longer)
    return (len(shorter_chars & longer_chars) / max(len(shorter_chars), 1)) >= 0.8


def _normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _build_visual_slots(chapter: Any) -> list[VisualSlot]:
    slots: list[VisualSlot] = []
    for index, frame in enumerate(chapter.frames[:2], start=1):
        if not frame.path:
            continue
        slots.append(
            VisualSlot(
                slot_id=f"visual-slot-{chapter.index}-{index}",
                visual_role="process_figure" if frame.visual_type in {"code", "formula"} else "keyframe_explainer",
                title=frame.caption or f"Visual {index}",
                caption=frame.insight or frame.caption,
                source_paths=[frame.path],
                evidence_refs=[f"frame-{chapter.index}-{index}"],
                ts=float(frame.ts),
            )
        )
    return slots


def _build_unit_evidence_refs(chapter: Any) -> list[str]:
    refs = [f"chapter-{chapter.index}", f"compound-teaching-unit-{chapter.index}"]
    refs.extend(f"quote-{chapter.index}-{idx}" for idx, _ in enumerate(chapter.points, start=1))
    refs.extend(f"frame-{chapter.index}-{idx}" for idx, frame in enumerate(chapter.frames, start=1) if frame.path)
    refs.extend(f"procedure-{chapter.index}-{idx}" for idx, _ in enumerate(chapter.process_steps, start=1))
    return refs
