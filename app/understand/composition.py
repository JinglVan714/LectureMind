from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import get_settings
from .schema import (
    BodySectionView,
    CompositionView,
    LectureJSON,
    OutlineBlockView,
    SupportingVisualView,
)

_SKIM_THRESHOLD = 0.18
_HANDOUT_THRESHOLD = 0.45
_SEGMENTED_THRESHOLD = 0.72
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_ASR_TERM_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("可不搞", "口播稿"),
    ("开外计划", "开发大纲"),
    ("上下门管理", "上下文管理"),
    ("注意注意", "注意"),
)
_BODY_NOISE_MARKERS = (
    "语音识别错误",
    "字幕中为",
    "术语在字幕中",
    "实际应为",
)


def compose_from_lecture_json(lecture: LectureJSON) -> CompositionView:
    if lecture.lecture_note_ir is not None and lecture.evidence_index is not None:
        return compat_projection_from_note_and_evidence(lecture)
    signals = _burden_signals(lecture)
    burden_score = _burden_score(signals)
    summary_mode = _summary_mode(burden_score)
    composition_profile = _composition_profile(lecture)
    body_sections = _body_sections(lecture, composition_profile, summary_mode)
    body_sections, visual_plan = _attach_supporting_visuals(lecture, body_sections)
    return CompositionView(
        burden_score=burden_score,
        burden_signals=signals,
        summary_mode=summary_mode,
        semantic_profile=lecture.profile.primary_type or lecture.category,
        composition_profile=composition_profile,
        reorder_strength="light",
        reading_goal=(
            "standalone_dense_handout"
            if summary_mode in {"handout", "segmented_handout"}
            else "quick_independent_summary"
        ),
        hero_summary=_core_summary_text(lecture),
        key_takeaways_top=_top_takeaways(lecture),
        audience_fit=_audience_fit(summary_mode),
        body_sections=body_sections,
        source_index=_source_index(lecture),
        visual_plan={"items": visual_plan},
    )


def compat_projection_from_note_and_evidence(lecture: LectureJSON) -> CompositionView:
    legacy = lecture.composition if lecture.composition and lecture.composition.body_sections else None
    summary_mode = (
        legacy.summary_mode
        if legacy and legacy.summary_mode
        else ("handout" if lecture.duration >= 600 else "note")
    )
    body_sections = body_sections_from_teaching_units(lecture)
    source_index = source_index_from_evidence_projection(lecture)
    visual_plan = {"items": [visual for section in body_sections for visual in section.supporting_visuals]}
    return CompositionView(
        burden_score=legacy.burden_score if legacy else 0.0,
        burden_signals=legacy.burden_signals if legacy else {},
        summary_mode=summary_mode,
        semantic_profile=legacy.semantic_profile if legacy else lecture.category,
        composition_profile=legacy.composition_profile if legacy else "mixed",
        reorder_strength="light",
        reading_goal=legacy.reading_goal if legacy else "compatibility_projection",
        hero_summary=lecture.lecture_note_ir.front_matter.one_sentence_claim,
        key_takeaways_top=list(lecture.lecture_note_ir.front_matter.takeaways_top),
        audience_fit=legacy.audience_fit if legacy else "",
        body_sections=body_sections,
        source_index=source_index,
        visual_plan=visual_plan,
        meta={"source": "compat_projection"},
    )


def body_sections_from_teaching_units(lecture: LectureJSON) -> list[BodySectionView]:
    note = lecture.lecture_note_ir
    if note is None:
        return []
    sections: list[BodySectionView] = []
    for unit in note.body.teaching_units:
        sections.append(
            BodySectionView(
                id=unit.unit_id,
                title=unit.title,
                section_role=unit.unit_role,
                summary=unit.core_message,
                paragraphs=[
                    paragraph
                    for block in unit.content_blocks
                    for paragraph in ([block.lead] if block.lead else []) + list(block.paragraphs)
                ],
                outline_blocks=[
                    OutlineBlockView(
                        id=block.block_id,
                        ordinal=index,
                        title=block.title,
                        lead=block.lead,
                        paragraphs=list(block.paragraphs),
                        source_chapter_refs=list(block.source_chapter_refs),
                        source_timestamps=list(block.source_timestamps),
                        block_role=block.block_role,
                    )
                    for index, block in enumerate(unit.content_blocks, start=1)
                ],
                supporting_visuals=[
                    SupportingVisualView(
                        visual_role=slot.visual_role,
                        source_mode="lecture_note_ir",
                        path=slot.source_paths[0] if slot.source_paths else "",
                        caption=slot.caption or slot.title,
                        ts=float(slot.ts or 0.0),
                    )
                    for slot in unit.visual_slots
                ],
                source_chapter_refs=list(unit.source_chapter_refs),
            )
        )
    return sections


def source_index_from_evidence_projection(lecture: LectureJSON) -> dict[str, Any]:
    evidence = lecture.evidence_index
    if evidence is None:
        return {}
    views = evidence.projection_views
    return {
        "chapter_map": list(views.get("chapter_map", [])),
        "evidence_quotes": list(views.get("evidence_quotes", [])),
        "term_index": list(views.get("term_index", [])),
        "tool_appendix": list(views.get("tool_appendix", [])),
        "source_index_view": list(views.get("source_index_view", [])),
    }


def _burden_signals(lecture: LectureJSON) -> dict[str, Any]:
    total_chars = len(lecture.one_liner or "") + len(lecture.final_synthesis or "")
    teaching_note_count = 0
    point_count = 0
    key_takeaway_count = 0
    pitfall_count = 0
    code_or_formula_count = 0
    chapter_frame_count = 0
    for chapter in lecture.chapters:
        total_chars += len(chapter.title or "") + len(chapter.learning_goal or "") + len(chapter.summary or "")
        total_chars += sum(len(item or "") for item in chapter.teaching_notes)
        total_chars += sum(len(item or "") for item in chapter.process_steps)
        total_chars += sum(len(item or "") for item in chapter.key_takeaways)
        total_chars += sum(len(item or "") for item in chapter.pitfalls)
        total_chars += sum(len(point.text or "") + len(point.quote or "") for point in chapter.points)
        teaching_note_count += len(chapter.teaching_notes)
        point_count += len(chapter.points)
        key_takeaway_count += len(chapter.key_takeaways)
        pitfall_count += len(chapter.pitfalls)
        code_or_formula_count += len(chapter.code_blocks) + len(chapter.formula_blocks)
        chapter_frame_count += len(chapter.frames)
    total_chars += sum(len(item.term or "") + len(item.explanation or "") for item in lecture.glossary)
    total_chars += sum(len(item or "") for item in lecture.study_questions)
    total_chars += sum(len(item or "") for item in lecture.review_questions)
    duration_min = max(float(lecture.duration or 0.0) / 60.0, 1.0)
    return {
        "duration_min": round(duration_min, 2),
        "chapter_count": len(lecture.chapters),
        "chars_per_min": round(total_chars / duration_min, 2),
        "has_final_synthesis": 1 if (lecture.final_synthesis or lecture.one_liner) else 0,
        "study_question_count": len(lecture.study_questions),
        "review_question_count": len(lecture.review_questions),
        "glossary_count": len(lecture.glossary),
        "knowledge_unit_count": len(lecture.knowledge_units),
        "point_count": point_count,
        "key_takeaway_count": key_takeaway_count,
        "pitfall_count": pitfall_count,
        "teaching_note_count": teaching_note_count,
        "code_or_formula_count": code_or_formula_count,
        "visual_evidence_count": len(lecture.visual_evidence) + chapter_frame_count,
    }


def _burden_score(signals: dict[str, Any]) -> float:
    score = 0.0
    score += min(float(signals["chapter_count"]) / 4.0, 1.0) * 0.16
    score += min(float(signals["glossary_count"]) / 4.0, 1.0) * 0.12
    score += min(float(signals["study_question_count"]) / 3.0, 1.0) * 0.12
    structured_density = (
        float(signals["knowledge_unit_count"])
        + float(signals["point_count"])
        + float(signals["key_takeaway_count"])
        + float(signals["pitfall_count"])
    )
    score += min(structured_density / 8.0, 1.0) * 0.18
    score += min(float(signals["chars_per_min"]) / 90.0, 1.0) * 0.18
    score += min(float(signals["teaching_note_count"]) / 4.0, 1.0) * 0.08
    score += min(float(signals["visual_evidence_count"]) / 2.0, 1.0) * 0.08
    if float(signals["code_or_formula_count"]) > 0:
        score += 0.08
    if float(signals["has_final_synthesis"]) > 0:
        score += 0.08
    if float(signals["chapter_count"]) > 1:
        score += 0.04
    if float(signals["duration_min"]) >= 20.0:
        score += 0.04
    return round(min(score, 1.0), 4)


def _summary_mode(burden_score: float) -> str:
    if burden_score >= _SEGMENTED_THRESHOLD:
        return "segmented_handout"
    if burden_score >= _HANDOUT_THRESHOLD:
        return "handout"
    if burden_score <= _SKIM_THRESHOLD:
        return "skim"
    return "note"


def _composition_profile(lecture: LectureJSON) -> str:
    primary_type = (lecture.profile.primary_type or lecture.category or "").lower()
    haystack = f"{lecture.title or ''} {lecture.core_question or ''}".lower()
    if "procedural" in primary_type or any(
        token in haystack for token in ("教程", "步骤", "流程", "配置", "安装", "实操", "操作")
    ):
        return "procedural"
    if "interview" in primary_type or any(
        token in haystack for token in ("访谈", "观点", "争论", "评论", "辩论")
    ):
        return "argumentative"
    if any(token in haystack for token in ("案例", "复盘", "实战", "项目")):
        return "case_based"
    return "conceptual"


def _body_sections(
    lecture: LectureJSON,
    composition_profile: str,
    summary_mode: str,
) -> list[BodySectionView]:
    sections: list[BodySectionView] = []
    for group in _chapter_groups(lecture, summary_mode):
        section = _body_section_from_group(group, composition_profile)
        if section is not None:
            sections.append(section)
    boundary = _boundary_section(lecture)
    if boundary is not None:
        sections.append(boundary)
    return sections


def _mainline_section(lecture: LectureJSON, composition_profile: str) -> BodySectionView | None:
    seeds = [str(item).strip() for item in (lecture.learning_path or lecture.mainline) if str(item).strip()]
    if not seeds:
        return None
    refs = [chapter.index for chapter in lecture.chapters[: min(4, len(lecture.chapters))]]
    blocks = _section_outline_blocks(
        [
            _make_outline_block(
                ordinal=1,
                title=_flow_title(composition_profile),
                paragraphs=[f"{idx}. {_clean_text(item)}" for idx, item in enumerate(seeds[:4], start=1)],
                source_chapter_refs=refs,
                source_timestamps=[float(lecture.chapters[0].start)] if lecture.chapters else [],
                block_role="process" if composition_profile == "procedural" else "reason",
                source_hint="mainline",
            )
        ]
    )
    return _build_section(
        id="key-flow",
        title=_flow_title(composition_profile),
        section_role="process" if composition_profile == "procedural" else "reason",
        summary="先把视频真正推进的主线抓住，后面的细节才不会散。",
        outline_blocks=blocks,
        source_chapter_refs=refs,
    )


def _chapter_groups(lecture: LectureJSON, summary_mode: str) -> list[list[Any]]:
    chapters = list(lecture.chapters)
    if not chapters:
        return []
    return [[chapter] for chapter in chapters]


def _body_section_from_group(group: list[Any], composition_profile: str) -> BodySectionView | None:
    outline_blocks = _outline_blocks_for_group(group, composition_profile)
    if not outline_blocks:
        return None
    return _build_section(
        id=_group_id(group),
        title=_group_title(group),
        section_role=_group_role(group, composition_profile),
        summary=_group_summary(group),
        outline_blocks=outline_blocks,
        source_chapter_refs=[chapter.index for chapter in group],
    )


def _group_id(group: list[Any]) -> str:
    if len(group) == 1:
        return f"chapter-{group[0].index}"
    return "group-" + "-".join(str(chapter.index) for chapter in group)


def _group_title(group: list[Any]) -> str:
    if len(group) == 1:
        return _clean_text(group[0].title)
    return " · ".join(_clean_text(chapter.title) for chapter in group if _clean_text(chapter.title))


def _group_role(group: list[Any], composition_profile: str) -> str:
    haystack = " ".join(_clean_text(chapter.title) for chapter in group)
    if _contains_any(haystack, ("流程", "步骤", "工作流", "安装", "配置", "录制")):
        return "process"
    if _contains_any(haystack, ("为什么", "可控性", "设计", "维度")):
        return "reason"
    if _contains_any(haystack, ("实战", "案例", "演示", "项目")):
        return "case"
    if composition_profile == "procedural":
        return "process"
    if composition_profile == "argumentative":
        return "reason"
    if composition_profile == "case_based":
        return "case"
    return "concept"


def _group_summary(group: list[Any]) -> str:
    sentences: list[str] = []
    for chapter in group:
        summary = _first_sentence(_chapter_scoped_summary("", chapter.summary))
        if summary:
            sentences.append(summary)
    return _join_sentences(sentences[:2])


def _group_paragraphs(group: list[Any], composition_profile: str) -> list[str]:
    if len(group) > 1:
        return _dedupe_texts(
            [_chapter_paragraph(chapter, composition_profile, include_title=True) for chapter in group]
        )
    chapter = group[0]
    paragraphs: list[str] = []
    summary = _best_body_summary(chapter)
    if summary:
        paragraphs.append(summary)
    detail = _chapter_detail_sentence(chapter, composition_profile)
    if detail:
        paragraphs.append(detail)
    return _dedupe_texts(paragraphs)


def _chapter_paragraph(chapter: Any, composition_profile: str, *, include_title: bool) -> str:
    summary = _best_body_summary(chapter)
    detail = _chapter_detail_sentence(chapter, composition_profile)
    if include_title:
        summary = _chapter_scoped_summary(chapter.title, summary)
    if summary and detail:
        return _join_sentences([summary, detail])
    return summary or detail or ""


def _boundary_section(lecture: LectureJSON) -> BodySectionView | None:
    blocks: list[OutlineBlockView] = []
    pitfall_items = [
        (
            _clean_body_text(item),
            _clean_text(chapter.title),
            chapter.index,
        )
        for chapter in lecture.chapters[:4]
        for item in chapter.pitfalls[:1]
        if _clean_body_text(item)
    ]
    pitfalls = _dedupe_texts(
        [
            text
            for text, _, _ in pitfall_items
        ]
    )
    if pitfalls:
        blocks.append(
            _make_outline_block(
                ordinal=len(blocks) + 1,
                title=_topic_title_from_texts(
                    pitfalls,
                    fallback=_theme_from_titles([title for _, title, _ in pitfall_items]),
                ),
                paragraphs=pitfalls[:3],
                source_chapter_refs=[index for _, _, index in pitfall_items],
                source_timestamps=[float(lecture.chapters[-1].end)] if lecture.chapters else [],
                block_role="boundary",
                source_hint="pitfalls",
            )
        )
    questions = _dedupe_texts(
        [_clean_text(item) for item in (lecture.study_questions[:2] + lecture.review_questions[:2]) if item]
    )
    if questions:
        blocks.append(
            _make_outline_block(
                ordinal=len(blocks) + 1,
                title=_topic_title_from_texts(
                    questions,
                    fallback=_theme_from_titles([chapter.title for chapter in lecture.chapters[:2]]) or "复盘问题",
                ),
                paragraphs=questions[:3],
                source_chapter_refs=[chapter.index for chapter in lecture.chapters],
                source_timestamps=[float(lecture.chapters[-1].end)] if lecture.chapters else [],
                block_role="review",
                source_hint="review_questions",
            )
        )
    transfer = _transfer_sentence(lecture.final_synthesis)
    if transfer:
        clean_transfer = _clean_transfer_sentence(transfer)
        blocks.append(
            _make_outline_block(
                ordinal=len(blocks) + 1,
                title=_transfer_block_title(
                    clean_transfer,
                    fallback=_theme_from_titles([chapter.title for chapter in lecture.chapters[-2:]]) or "迁移场景",
                ),
                paragraphs=[clean_transfer],
                source_chapter_refs=[chapter.index for chapter in lecture.chapters],
                source_timestamps=[float(lecture.chapters[-1].end)] if lecture.chapters else [],
                block_role="transfer",
                source_hint="final_synthesis",
            )
        )
    if not blocks:
        return None
    return _build_section(
        id="boundary-review",
        title="收束与复盘",
        section_role="boundary",
        summary=_boundary_section_summary(blocks),
        outline_blocks=_section_outline_blocks(blocks),
        source_chapter_refs=[chapter.index for chapter in lecture.chapters],
    )


def _build_section(
    *,
    id: str,
    title: str,
    section_role: str,
    summary: str,
    outline_blocks: list[OutlineBlockView],
    source_chapter_refs: list[int],
) -> BodySectionView:
    return BodySectionView(
        id=id,
        title=title,
        section_role=section_role,
        summary=summary,
        paragraphs=_flatten_outline_block_paragraphs(outline_blocks),
        outline_blocks=outline_blocks,
        source_chapter_refs=source_chapter_refs,
    )


def _section_outline_blocks(blocks: list[OutlineBlockView]) -> list[OutlineBlockView]:
    normalized: list[OutlineBlockView] = []
    for ordinal, block in enumerate(blocks, start=1):
        normalized.append(block.model_copy(update={"ordinal": ordinal}))
    return normalized


def _flatten_outline_block_paragraphs(blocks: list[OutlineBlockView]) -> list[str]:
    paragraphs: list[str] = []
    for block in blocks:
        if block.lead:
            paragraphs.append(_clean_body_text(block.lead))
        paragraphs.extend(_clean_body_text(item) for item in block.paragraphs if _clean_body_text(item))
    return _dedupe_texts(paragraphs)


def _outline_blocks_for_group(group: list[Any], composition_profile: str) -> list[OutlineBlockView]:
    blocks: list[OutlineBlockView] = []
    for chapter in group:
        blocks.extend(_chapter_outline_blocks(chapter, composition_profile, include_title=len(group) > 1))
    return _section_outline_blocks(blocks)


def _chapter_outline_blocks(
    chapter: Any,
    composition_profile: str,
    *,
    include_title: bool,
) -> list[OutlineBlockView]:
    summary = _best_body_summary(chapter)
    detail = _chapter_detail_sentence(chapter, composition_profile)
    detail_title = _chapter_detail_block_title(chapter)
    refs = [chapter.index]
    base_timestamps = _chapter_source_timestamps(chapter)
    blocks: list[OutlineBlockView] = []
    primary_paragraphs = _dedupe_texts(
        [_chapter_scoped_summary(chapter.title if include_title else "", summary)]
    )
    if primary_paragraphs:
        blocks.append(
            _make_outline_block(
                ordinal=len(blocks) + 1,
                title=_clean_text(chapter.title),
                paragraphs=primary_paragraphs,
                source_chapter_refs=refs,
                source_timestamps=base_timestamps,
                block_role=_group_role([chapter], composition_profile),
                topic_hint=_clean_text(chapter.title),
                source_hint="chapter_summary",
            )
        )
    if detail and _should_split_chapter_into_detail_block(chapter) and _detail_block_adds_value(
        chapter, detail, detail_title
    ):
        blocks.append(
            _make_outline_block(
                ordinal=len(blocks) + 1,
                title=detail_title,
                paragraphs=[detail],
                source_chapter_refs=refs,
                source_timestamps=base_timestamps,
                block_role="detail",
                topic_hint=_clean_text(chapter.learning_goal or chapter.title),
                source_hint="chapter_detail",
            )
        )
    elif detail:
        if blocks:
            merged = list(blocks[0].paragraphs) + [detail]
            blocks[0] = blocks[0].model_copy(update={"paragraphs": _dedupe_texts(merged)})
        else:
            blocks.append(
                _make_outline_block(
                    ordinal=1,
                    title=_clean_text(chapter.title),
                    paragraphs=[detail],
                    source_chapter_refs=refs,
                    source_timestamps=base_timestamps,
                    block_role="detail",
                    topic_hint=_clean_text(chapter.learning_goal or chapter.title),
                    source_hint="chapter_detail",
                )
            )
    return blocks


def _should_split_chapter_into_detail_block(chapter: Any) -> bool:
    summary = _clean_body_text(chapter.summary)
    sentence_count = len(_split_sentences(summary))
    structured_items = (
        len(getattr(chapter, "key_takeaways", []) or [])
        + len(getattr(chapter, "points", []) or [])
        + len(getattr(chapter, "pitfalls", []) or [])
        + len(getattr(chapter, "process_steps", []) or [])
    )
    if sentence_count >= 3:
        return structured_items >= 2
    if sentence_count >= 2:
        return structured_items >= 3
    if len(summary) >= 120:
        return structured_items >= 3
    return structured_items >= 4 and len(summary) >= 80


def _chapter_source_timestamps(chapter: Any) -> list[float]:
    timestamps = [float(chapter.start), float(chapter.end)]
    if getattr(chapter, "points", None):
        timestamps.extend(float(point.ts) for point in chapter.points[:2])
    return list(dict.fromkeys(ts for ts in timestamps if ts >= 0))


def _chapter_detail_block_title(chapter: Any) -> str:
    chapter_title = _clean_text(getattr(chapter, "title", ""))
    summary_sentences = _split_sentences(_clean_body_text(getattr(chapter, "summary", "")))
    candidates = (
        summary_sentences[1:]
        + [_clean_body_text(getattr(chapter, "learning_goal", ""))]
        + list(getattr(chapter, "process_steps", [])[:2])
        + list(getattr(chapter, "key_takeaways", [])[:2])
        + [point.text for point in getattr(chapter, "points", [])[:2]]
        + list(getattr(chapter, "pitfalls", [])[:1])
    )
    return _topic_title_from_texts(candidates, fallback=chapter_title, avoid=(chapter_title,))


def _detail_block_adds_value(chapter: Any, detail: str, detail_title: str) -> bool:
    chapter_title = _clean_text(getattr(chapter, "title", ""))
    clean_title = _clean_text(detail_title)
    clean_detail = _clean_body_text(detail).rstrip("。；; ")
    summary = _clean_body_text(getattr(chapter, "summary", ""))
    if not clean_title or clean_title == chapter_title:
        return False
    if clean_title == clean_detail:
        return False
    if clean_title in clean_detail and len(clean_detail) - len(clean_title) <= 4:
        return False
    if clean_detail and clean_detail in summary:
        return False
    if _detail_restates_summary(summary, clean_detail):
        return False
    return True


def _make_outline_block(
    *,
    ordinal: int,
    id: str = "",
    title: str,
    paragraphs: list[str],
    source_chapter_refs: list[int],
    source_timestamps: list[float],
    lead: str = "",
    block_role: str = "",
    topic_hint: str = "",
    source_hint: str = "",
) -> OutlineBlockView:
    return OutlineBlockView(
        id=id,
        ordinal=ordinal,
        title=_clean_text(title),
        lead=_clean_body_text(lead),
        paragraphs=_dedupe_texts([_clean_body_text(item) for item in paragraphs if _clean_body_text(item)]),
        source_chapter_refs=list(dict.fromkeys(source_chapter_refs)),
        source_timestamps=list(dict.fromkeys(float(ts) for ts in source_timestamps if ts >= 0)),
        block_role=block_role,
        topic_hint=_clean_text(topic_hint),
        source_hint=_clean_text(source_hint),
    )


def _core_summary_text(lecture: LectureJSON) -> str:
    return _clean_text(lecture.one_liner or _first_sentence(lecture.final_synthesis))


def _core_paragraphs(lecture: LectureJSON) -> list[str]:
    paragraphs: list[str] = []
    if lecture.core_question:
        paragraphs.append(f"这份讲义要回答的问题是：{_clean_text(lecture.core_question)}")
    if lecture.final_synthesis:
        paragraphs.extend(_paragraph_chunks(lecture.final_synthesis, limit=150, max_chunks=2))
    elif lecture.one_liner:
        paragraphs.append(_clean_text(lecture.one_liner))
    return _dedupe_texts(paragraphs)


def _attach_supporting_visuals(
    lecture: LectureJSON,
    sections: list[BodySectionView],
) -> tuple[list[BodySectionView], list[dict[str, Any]]]:
    chapter_frame_map = {
        chapter.index: [frame for frame in chapter.frames if frame.path]
        for chapter in lecture.chapters
    }
    global_frames = [frame for frame in lecture.visual_evidence if frame.path]
    fallback_frames = _filesystem_keyframes_by_chapter(lecture)
    global_cursor = 0
    used_paths: set[str] = set()
    visual_plan: list[dict[str, Any]] = []
    chapter_titles = {chapter.index: _clean_text(chapter.title) for chapter in lecture.chapters}
    updated: list[BodySectionView] = []
    for section in sections:
        supporting_visuals: list[SupportingVisualView] = []
        if section.id in {"core-summary", "key-flow", "boundary-review"}:
            updated.append(section.model_copy(update={"supporting_visuals": supporting_visuals}))
            continue
        target_count = 2 if len(section.source_chapter_refs) > 1 and section.id.startswith("group-") else 1
        for chapter_index in section.source_chapter_refs:
            if len(supporting_visuals) >= target_count:
                break
            frames = chapter_frame_map.get(chapter_index) or []
            fallback = fallback_frames.get(chapter_index) or []
            candidate = _first_unused_frame(frames, used_paths) or _first_unused_frame(fallback, used_paths)
            if candidate is None:
                continue
            used_paths.add(str(candidate.path))
            supporting_visuals.append(
                SupportingVisualView(
                    visual_role="keyframe",
                    source_mode="video_frame",
                    path=str(candidate.path),
                    caption=_visual_caption(section.title, chapter_titles.get(chapter_index, ""), candidate),
                    ts=float(candidate.ts or 0.0),
                )
            )
            visual_plan.append(
                {
                    "visual_role": "keyframe",
                    "placement_section_id": section.id,
                    "source_mode": "video_frame",
                    "must_have": True,
                    "source_refs": [{"ts": float(candidate.ts or 0.0), "path": str(candidate.path)}],
                }
            )
        if not supporting_visuals and global_cursor < len(global_frames):
            while global_cursor < len(global_frames):
                probe = global_frames[global_cursor]
                global_cursor += 1
                if str(probe.path) in used_paths:
                    continue
                used_paths.add(str(probe.path))
                supporting_visuals.append(
                    SupportingVisualView(
                        visual_role="keyframe",
                        source_mode="video_frame",
                        path=str(probe.path),
                        caption=_visual_caption(section.title, "", probe),
                        ts=float(probe.ts or 0.0),
                    )
                )
                visual_plan.append(
                    {
                        "visual_role": "keyframe",
                        "placement_section_id": section.id,
                        "source_mode": "video_frame",
                        "must_have": True,
                        "source_refs": [{"ts": float(probe.ts or 0.0), "path": str(probe.path)}],
                    }
                )
                break
        updated.append(section.model_copy(update={"supporting_visuals": supporting_visuals}))
    return updated, visual_plan


def _visual_caption(section_title: str, chapter_title: str, frame: Any) -> str:
    raw_caption = _clean_text(getattr(frame, "caption", "") or getattr(frame, "insight", ""))
    if raw_caption and raw_caption not in {"关键画面", "关键截图"}:
        return raw_caption
    if chapter_title:
        return f"对应“{chapter_title}”的关键画面，用来辅助理解“{section_title}”。"
    return f"对应“{section_title}”的关键画面。"


def _first_unused_frame(frames: list[Any], used_paths: set[str]) -> Any | None:
    for frame in frames:
        if str(frame.path) not in used_paths:
            return frame
    return None


def _filesystem_keyframes_by_chapter(lecture: LectureJSON) -> dict[int, list[Any]]:
    keyframe_dir = get_settings().data_dir / "keyframes" / lecture.bv_id
    if not keyframe_dir.exists():
        return {}
    raw_frames: list[SupportingVisualView] = []
    for path in sorted(keyframe_dir.glob("*.jpg")):
        ts = _timestamp_from_keyframe_path(path)
        if ts is None:
            continue
        raw_frames.append(
            SupportingVisualView(
                visual_role="keyframe",
                source_mode="video_frame",
                path=f"keyframes/{lecture.bv_id}/{path.name}",
                caption="关键画面",
                ts=ts,
            )
        )
    if not raw_frames:
        return {}
    out: dict[int, list[Any]] = {}
    for chapter in lecture.chapters:
        midpoint = (float(chapter.start) + float(chapter.end)) / 2.0
        in_range = [frame for frame in raw_frames if float(chapter.start) <= frame.ts <= float(chapter.end)]
        pool = in_range or raw_frames
        ranked = sorted(pool, key=lambda frame: abs(frame.ts - midpoint))
        out[chapter.index] = ranked[:2]
    return out


def _timestamp_from_keyframe_path(path: Path) -> float | None:
    stem = path.stem.strip()
    if not stem.isdigit():
        return None
    return int(stem) / 1000.0


def _source_index(lecture: LectureJSON) -> dict[str, Any]:
    evidence_quotes = []
    for chapter in lecture.chapters:
        for point in chapter.points[:1]:
            label = _clean_text(point.text)
            quote = _clean_text(point.quote)
            if not label and not quote:
                continue
            evidence_quotes.append(
                {
                    "label": label,
                    "quote": quote,
                    "ts": float(point.ts),
                    "chapter_index": chapter.index,
                }
            )
    return {
        "chapter_map": [
            {
                "chapter_index": chapter.index,
                "title": _clean_text(chapter.title),
                "start": float(chapter.start),
                "end": float(chapter.end),
            }
            for chapter in lecture.chapters
        ],
        "evidence_quotes": evidence_quotes[:4],
        "term_index": [
            {
                "term": _clean_text(item.term),
                "ts": float(item.ts),
                "explanation": _clean_text(item.explanation),
            }
            for item in lecture.glossary[:6]
            if _clean_text(item.term)
        ],
        "tool_appendix": [],
    }


def _top_takeaways(lecture: LectureJSON) -> list[str]:
    items: list[str] = []
    for chapter in lecture.chapters:
        items.extend(_clean_body_text(item) for item in chapter.key_takeaways if _clean_body_text(item))
    if not items:
        items.extend(_clean_body_text(item) for item in lecture.learning_path[:4] if _clean_body_text(item))
    if not items and lecture.final_synthesis:
        items.extend(_paragraph_chunks(lecture.final_synthesis, limit=100, max_chunks=2))
    return _dedupe_texts(items)[:4]


def _audience_fit(summary_mode: str) -> str:
    if summary_mode in {"handout", "segmented_handout"}:
        return "适合想脱离视频直接理解方法论、流程和可迁移结论的读者。"
    if summary_mode == "skim":
        return "适合先快速判断这条视频值不值得继续投入时间。"
    return "适合先建立整体理解，再按需回看原视频。"


def _best_body_summary(chapter: Any) -> str:
    summary = _clean_body_text(chapter.summary)
    if summary:
        return summary
    if chapter.learning_goal:
        return _clean_body_text(chapter.learning_goal)
    return ""


def _chapter_detail_sentence(chapter: Any, composition_profile: str) -> str:
    summary = _clean_body_text(chapter.summary)
    topic_seed = " ".join(
        item
        for item in (
            _clean_body_text(chapter.title),
            summary,
            _clean_body_text(chapter.learning_goal),
        )
        if item
    )
    dense_summary = len(summary) >= 90 or len(_split_sentences(summary)) >= 2
    very_dense_summary = len(summary) >= 150 or len(_split_sentences(summary)) >= 3
    if very_dense_summary:
        return ""
    process_sentence = _procedural_detail_sentence(chapter) if composition_profile == "procedural" else ""
    takeaway = _best_topic_matched_text(chapter.key_takeaways, topic_seed, min_overlap=2 if dense_summary else 0)
    point = _best_topic_matched_text([point.text for point in chapter.points], topic_seed, min_overlap=2 if dense_summary else 0)
    pitfall = _best_topic_matched_text(chapter.pitfalls, topic_seed, min_overlap=2 if dense_summary else 0)
    note = _best_topic_matched_text(chapter.teaching_notes, topic_seed, min_overlap=1 if dense_summary else 0)
    if dense_summary:
        if takeaway:
            return _profile_takeaway_sentence(takeaway, composition_profile)
        if point:
            return _profile_point_sentence(point, composition_profile)
        if process_sentence:
            return process_sentence
        if note and _topic_overlap_score(note, topic_seed) >= 2:
            return _join_sentences([note])
        if pitfall:
            return _profile_boundary_sentence(pitfall, composition_profile)
        return ""
    parts: list[str] = []
    if process_sentence:
        parts.append(process_sentence.rstrip("。"))
    if takeaway:
        parts.append(_profile_takeaway_sentence(takeaway, composition_profile).rstrip("。"))
    elif point:
        parts.append(_profile_point_sentence(point, composition_profile).rstrip("。"))
    if pitfall:
        parts.append(_profile_boundary_sentence(pitfall, composition_profile).rstrip("。"))
    if not parts and note:
        parts.append(note.rstrip("。"))
    if not parts:
        return ""
    return "；".join(part for part in parts if part) + "。"


def _transfer_sentence(final_synthesis: str) -> str:
    sentences = _split_sentences(final_synthesis)
    for sentence in reversed(sentences):
        if _contains_any(sentence, ("适用于", "可以推广", "可迁移", "迁移到", "迁到", "延伸", "后续行动建议")):
            return sentence
    return ""


def _clean_transfer_sentence(text: str) -> str:
    value = _clean_body_text(text).rstrip("。！？；; ")
    if not value:
        return ""
    value = re.sub(
        r"^(?:后续行动建议|行动建议|迁移建议|可迁移的结论)(?:是|为)?[：:]\s*",
        "",
        value,
    )
    value = re.sub(
        r"^(?:后续行动建议|行动建议|迁移建议|可迁移的结论)(?:是|为)\s*",
        "",
        value,
    )
    return _join_sentences([value])


def _transfer_block_title(text: str, fallback: str) -> str:
    clean_text = _clean_transfer_sentence(text).rstrip("。！？；; ")
    for pattern in (
        r"迁(?:移)?到(.{2,18}?)(?:里|中|上|内|等|。|，|$)",
        r"适用于(.{2,18}?)(?:里|中|上|内|等|。|，|$)",
        r"推广到(.{2,18}?)(?:里|中|上|内|等|。|，|$)",
    ):
        match = re.search(pattern, clean_text)
        if match:
            candidate = _clean_topic_title(match.group(1))
            if _is_usable_topic_title(candidate, avoided=set()):
                return candidate
    return _topic_title_from_texts([clean_text], fallback=fallback, avoid=("后续行动建议是", "后续行动建议"))


def _procedural_detail_sentence(chapter: Any) -> str:
    steps = [_clean_body_text(step) for step in chapter.process_steps[:3] if _clean_body_text(step)]
    if not steps:
        return ""
    return "流程上可以拆成" + _join_clause_list(steps) + "。"


def _boundary_section_summary(blocks: list[OutlineBlockView]) -> str:
    titles = _dedupe_texts([_clean_text(block.title) for block in blocks if _clean_text(block.title)])
    if not titles:
        return ""
    return _join_sentences(["；".join(title.rstrip("。！？；; ") for title in titles[:3])])


def _detail_restates_summary(summary: str, detail: str) -> bool:
    clean_summary = _clean_body_text(summary)
    clean_detail = _clean_body_text(detail).rstrip("。！？；; ")
    if not clean_summary or not clean_detail:
        return False
    detail_ngrams = _char_ngrams(clean_detail)
    if not detail_ngrams:
        return False
    for sentence in _split_sentences(clean_summary):
        clean_sentence = _clean_body_text(sentence).rstrip("。！？；; ")
        if not clean_sentence:
            continue
        if clean_detail in clean_sentence or clean_sentence in clean_detail:
            return True
        if _boundary_restatement_match(clean_detail, clean_sentence):
            return True
        overlap = _topic_overlap_score(clean_detail, clean_sentence)
        if overlap >= max(4, int(len(detail_ngrams) * 0.45)) and len(clean_detail) >= int(len(clean_sentence) * 0.4):
            return True
    return False


def _boundary_restatement_match(detail: str, sentence: str) -> bool:
    match = re.search(r"(.{1,10}?)(?:不要|别)(?:替|帮)(.{1,10}?)(?:做决定|做判断|下判断)", detail)
    if not match:
        return False
    left = _clean_text(match.group(1))
    right = _clean_text(match.group(2))
    if not left or not right:
        return False
    if left not in sentence or right not in sentence:
        return False
    return _contains_any(sentence, ("不替", "不负责", "只负责", "提供候选", "给后面的"))


def _topic_title_from_texts(texts: list[str], fallback: str, avoid: tuple[str, ...] = ()) -> str:
    avoided = {_title_key(item) for item in avoid if _title_key(item)}
    for text in texts:
        candidate = _topic_title_from_text(text)
        if _is_usable_topic_title(candidate, avoided):
            return candidate
    fallback_title = _clean_text(fallback)
    if _is_usable_topic_title(fallback_title, avoided=set()):
        return fallback_title
    return fallback_title


def _topic_title_from_text(text: str) -> str:
    full_text = _clean_body_text(text).rstrip("。！？；;：: ")
    if not full_text:
        return ""
    for pattern, formatter in (
        (r"(.{2,24}?)不要替(.{2,16}?)做决定$", lambda m: f"{m.group(1)}与{m.group(2)}分工"),
        (r"什么时候不要再扩大(.{2,16}?)$", lambda m: f"{m.group(1)}边界"),
        (r"不要.*把所有.*压给(.{2,16}?)$", lambda m: f"{m.group(1)}的边界"),
        (r"(.{2,28}?)(?:也?可以(?:迁移到|迁移|迁到|推广).*)$", lambda m: m.group(1)),
    ):
        match = re.search(pattern, full_text)
        if match:
            return _clean_topic_title(formatter(match))
    clause = re.split(r"[，。；：:！？]", full_text, maxsplit=1)[0]
    clause = re.sub(r"^(?:这一节|这部分|这一段|本章|本节|后半段|前半段|接下来|最后|这里)\s*", "", clause)
    clause = re.sub(r"^(?:先|再|继续|重点|核心|主要|至少)?(?:解释|说明|拆解|讨论|复盘|回到|看|看看)\s*", "", clause)
    clause = re.sub(r"^(?:为什么|如何|怎么|什么时候|哪些|什么是|是否)\s*", "", clause)
    clause = re.sub(r"^(?:这套|这个|这些)\s*", "", clause)
    for pattern, formatter in (
        (r"(.{2,28}?)(?:应该怎么配合|怎么配合)$", lambda m: m.group(1)),
        (r"(.{2,28}?)(?:要一起看|一起看|要一起调|一起调)$", lambda m: m.group(1)),
        (r"(.{2,28}?)(?:该负责什么|各自负责(?:的判断)?|负责什么)$", lambda m: f"{m.group(1)}分工"),
        (r"(.{2,28}?)(?:到底在解决什么问题|在解决什么问题)$", lambda m: m.group(1)),
    ):
        match = re.search(pattern, clause)
        if match:
            return _clean_topic_title(formatter(match))
    return _clean_topic_title(clause)


def _clean_topic_title(text: str) -> str:
    value = _clean_text(text).rstrip("。！？；;：: ")
    value = re.sub(r"^(?:先|再|继续|重点|核心|主要)\s*", "", value)
    value = value.strip("，、 ")
    parts = [part.strip("，、 ") for part in re.split(r"[、和与]", value) if part.strip("，、 ")]
    if len(parts) >= 3 and len(value) > 12:
        value = f"{parts[0]}与{parts[-1]}"
    return value


def _is_usable_topic_title(title: str, avoided: set[str]) -> bool:
    clean_title = _clean_topic_title(title)
    if not clean_title:
        return False
    if len(clean_title) < 4 or len(clean_title) > 28:
        return False
    if _title_key(clean_title) in avoided:
        return False
    if re.match(r"^(?:不要|别|避免|为什么|如何|怎么|什么时候)", clean_title):
        return False
    if clean_title in {"容易误判的地方", "复盘时先回答这几个问题", "可以迁移的结论", "后续行动建议"}:
        return False
    return True


def _title_key(text: str) -> str:
    return _clean_topic_title(text).replace(" ", "")


def _theme_from_titles(titles: list[str]) -> str:
    unique_titles = _dedupe_texts([_clean_text(title) for title in titles if _clean_text(title)])
    if not unique_titles:
        return ""
    if len(unique_titles) == 1:
        return unique_titles[0]
    return f"{unique_titles[0]}与{unique_titles[1]}"


def _best_topic_matched_text(items: list[Any], topic_seed: str, *, min_overlap: int) -> str:
    topic = _clean_body_text(topic_seed)
    best = ""
    best_score = -1
    for raw in items:
        text = _clean_body_text(raw)
        if not text:
            continue
        score = _topic_overlap_score(text, topic)
        if score < min_overlap:
            continue
        if score > best_score:
            best = text
            best_score = score
    return best


def _topic_overlap_score(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if left in right or right in left:
        return max(2, min(len(left), len(right)))
    return len(_char_ngrams(left) & _char_ngrams(right))


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    value = re.sub(r"\s+", "", _clean_body_text(text))
    if len(value) < size:
        return {value} if value else set()
    return {
        value[idx : idx + size]
        for idx in range(len(value) - size + 1)
        if re.search(r"[\u4e00-\u9fffA-Za-z]", value[idx : idx + size])
    }


def _lead_sentence(prefix: str, text: str) -> str:
    value = _clean_body_text(text).rstrip("。；; ")
    if not value:
        return ""
    if not prefix:
        return value
    return prefix + value


def _profile_takeaway_sentence(text: str, composition_profile: str) -> str:
    return _join_sentences([_clean_body_text(text)])


def _profile_point_sentence(text: str, composition_profile: str) -> str:
    return _join_sentences([_clean_body_text(text)])


def _profile_boundary_sentence(text: str, composition_profile: str) -> str:
    return _join_sentences([_clean_body_text(text)])


def _chapter_scoped_summary(title: str, summary: str) -> str:
    clean_title = _clean_text(title)
    clean_summary = _clean_body_text(summary)
    if not clean_summary:
        return clean_title
    stripped = re.sub(r"^(本章节?|这一[章节部分段]|这一节)", "", clean_summary).strip("，。 ")
    stripped = re.sub(r"^(详细介绍了|详细拆解了|介绍了|拆解了|解释了|说明了)", "", stripped).strip("，。 ")
    if not stripped:
        return clean_summary
    return stripped


def _paragraph_chunks(text: str, limit: int, max_chunks: int) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) > limit:
            chunks.append(_join_sentences(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
        if len(chunks) >= max_chunks:
            break
    if current and len(chunks) < max_chunks:
        chunks.append(_join_sentences(current))
    return _dedupe_texts(chunks[:max_chunks])


def _first_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    return sentences[0] if sentences else ""


def _split_sentences(text: str) -> list[str]:
    value = _clean_text(text)
    if not value:
        return []
    parts = _SENTENCE_SPLIT_RE.split(value)
    return [part.strip() for part in parts if part and part.strip()]


def _join_sentences(sentences: list[str]) -> str:
    cleaned = [part.strip() for part in sentences if part and part.strip()]
    if not cleaned:
        return ""
    out = "".join(
        sentence if sentence[-1:] in {"。", "！", "？", "!", "?", "；", ";"} else sentence + "。"
        for sentence in cleaned
    )
    return _clean_text(out)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = _clean_text(text).lower()
    return any(str(needle).lower() in haystack for needle in needles)


def _dedupe_texts(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _clean_text(item)
        key = text.replace(" ", "")
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _join_clause_list(items: list[str]) -> str:
    return "；".join(item.rstrip("。；; ") for item in items if item and item.strip())


def _clean_body_text(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    if any(marker in value for marker in _BODY_NOISE_MARKERS):
        return ""
    return value


def _clean_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    for raw, replacement in _ASR_TERM_REPLACEMENTS:
        value = value.replace(raw, replacement)
    value = value.replace("……", "。").replace("...", "。").replace("…", "。")
    value = re.sub(r"[ ]*([，。！？；:])", r"\1", value)
    return value.strip(" /")
