from __future__ import annotations

from typing import Any

from .schema import (
    BodyUnitPlan,
    LectureBackMatterPlan,
    LectureBlueprint,
    LectureFrontMatterPlan,
    LectureJSON,
)


def build_lecture_blueprint(
    lecture: LectureJSON,
    semantic_reading: dict[str, str],
) -> LectureBlueprint:
    body_unit_plan: list[BodyUnitPlan] = []
    visual_needs: list[dict[str, Any]] = []
    appendix_candidates: list[dict[str, Any]] = []
    chapter_count = len(lecture.chapters)
    previous_title = ""

    for chapter in lecture.chapters:
        unit_id = f"teaching-unit-{chapter.index}"
        unit_role = _infer_unit_role(chapter, index=chapter.index, chapter_count=chapter_count)
        core_message = _chapter_core_message(chapter)
        transition = _transition_from_previous(previous_title, chapter.title)
        chapter_visuals = [
            {
                "unit_id": unit_id,
                "chapter_index": chapter.index,
                "visual_role": _infer_visual_role(frame),
                "path": frame.path,
                "caption": frame.caption,
                "ts": frame.ts,
            }
            for frame in chapter.frames[:2]
            if frame.path
        ]
        if chapter.pitfalls:
            appendix_candidates.append(
                {
                    "kind": "boundary_review",
                    "unit_id": unit_id,
                    "chapter_index": chapter.index,
                    "title": chapter.title,
                }
            )
        body_unit_plan.append(
            BodyUnitPlan(
                unit_id=unit_id,
                title=chapter.title,
                teaching_goal=chapter.learning_goal or chapter.summary or chapter.title,
                unit_role=unit_role,
                core_message=core_message,
                transition_from_previous=transition,
                source_chapter_refs=[chapter.index],
                visual_needs=chapter_visuals,
                appendix_candidates=[item["kind"] for item in appendix_candidates if item["unit_id"] == unit_id],
            )
        )
        visual_needs.extend(chapter_visuals)
        previous_title = chapter.title

    return LectureBlueprint(
        front_matter_plan=LectureFrontMatterPlan(
            one_sentence_claim=lecture.one_liner or lecture.final_synthesis or lecture.core_question,
            reader_orientation=lecture.core_question or lecture.title,
            takeaways_top=_takeaways_top(lecture),
            reading_map=[item.title for item in body_unit_plan],
            reader_prerequisites=_reader_prerequisites(lecture),
            suitable_for=_suitable_for(semantic_reading),
            not_suitable_for=_not_suitable_for(semantic_reading),
        ),
        body_unit_plan=body_unit_plan,
        back_matter_plan=LectureBackMatterPlan(
            boundary_and_risks=[pitfall for chapter in lecture.chapters for pitfall in chapter.pitfalls][:6],
            source_index_entrypoints=[plan.title for plan in body_unit_plan[:4]],
            appendices=appendix_candidates,
            transfer_and_next_steps=list(lecture.review_questions[:4] or lecture.study_questions[:4]),
        ),
        reorder_strength="light",
        visual_needs=visual_needs,
        appendix_candidates=appendix_candidates,
        semantic_portrait=semantic_reading.get("semantic_portrait", ""),
        reading_burden=semantic_reading.get("reading_burden", ""),
        teaching_center_of_gravity=semantic_reading.get("teaching_center_of_gravity", ""),
    )


def _chapter_core_message(chapter: Any) -> str:
    for candidate in (chapter.summary, chapter.learning_goal, *chapter.teaching_notes, *chapter.key_takeaways):
        text = str(candidate or "").strip()
        if text:
            return text
    return chapter.title


def _transition_from_previous(previous_title: str, current_title: str) -> str:
    previous_title = str(previous_title or "").strip()
    current_title = str(current_title or "").strip()
    if not previous_title:
        return "先建立全片主线和当前单元的入口。"
    if not current_title:
        return ""
    return f"从「{previous_title}」继续推进到「{current_title}」。"


def _infer_unit_role(chapter: Any, *, index: int, chapter_count: int) -> str:
    title = f"{chapter.title} {chapter.learning_goal} {chapter.summary}".lower()
    if index == chapter_count and (chapter.pitfalls or "总结" in title or "synthesis" in title):
        return "synthesis"
    if chapter.pitfalls or "边界" in title or "风险" in title:
        return "boundary"
    if chapter.process_steps or "步骤" in title or "流程" in title or "过程" in title:
        return "process"
    if "案例" in title or "case" in title:
        return "case"
    if "为什么" in title or "why" in title or "问题" in title:
        return "problem"
    if "机制" in title or "原理" in title or "mechanism" in title:
        return "mechanism"
    return "claim"


def _infer_visual_role(frame: Any) -> str:
    visual_type = str(getattr(frame, "visual_type", "") or "").strip().lower()
    if visual_type in {"diagram", "table"}:
        return "concept_map"
    if visual_type in {"code", "formula"}:
        return "process_figure"
    return "keyframe_explainer"


def _takeaways_top(lecture: LectureJSON) -> list[str]:
    items: list[str] = []
    for chapter in lecture.chapters:
        items.extend(text for text in chapter.key_takeaways if text)
    if not items:
        items.extend(text for text in lecture.learning_path if text)
    if not items and lecture.final_synthesis:
        items.append(lecture.final_synthesis)
    return items[:4]


def _reader_prerequisites(lecture: LectureJSON) -> list[str]:
    items = [item.term for item in lecture.glossary[:4] if item.term]
    if items:
        return items
    if lecture.domain_tags:
        return lecture.domain_tags[:4]
    return [lecture.category] if lecture.category else []


def _suitable_for(semantic_reading: dict[str, str]) -> list[str]:
    portrait = semantic_reading.get("semantic_portrait", "")
    burden = semantic_reading.get("reading_burden", "")
    items = ["想先建立主线再决定是否回看原视频的读者"]
    if portrait:
        items.append(f"希望快速吸收{portrait}内容的读者")
    if burden in {"high", "very_high"}:
        items.append("愿意阅读结构化长讲义的读者")
    return items[:3]


def _not_suitable_for(semantic_reading: dict[str, str]) -> list[str]:
    if semantic_reading.get("reading_burden", "") in {"high", "very_high"}:
        return ["只想看一句话结论、不打算进入细节的读者"]
    return ["只想直接复制代码或命令、不要背景解释的读者"]
