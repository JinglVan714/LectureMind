from __future__ import annotations

from typing import Any

from .schema import EvidenceIndex, EvidenceObject, EvidenceRelation, LectureBlueprint, LectureJSON, LectureNoteIR, RagChunk


def build_evidence_index(
    lecture: LectureJSON,
    blueprint: LectureBlueprint,
    note_ir: LectureNoteIR,
) -> EvidenceIndex:
    evidence_objects: list[EvidenceObject] = []
    evidence_relations: list[EvidenceRelation] = []
    source_index_view: list[dict[str, Any]] = []
    glossary_view: list[dict[str, Any]] = []
    chapter_map: list[dict[str, Any]] = []
    evidence_quotes: list[dict[str, Any]] = []
    term_index: list[dict[str, Any]] = []
    video_anchor: dict[str, Any] = {}
    page_anchor: dict[str, Any] = {}
    evidence_anchor: dict[str, Any] = {}

    unit_by_chapter = {
        chapter_ref: unit
        for chapter_ref, unit in zip(
            [chapter.index for chapter in lecture.chapters],
            note_ir.body.teaching_units,
            strict=False,
        )
    }

    for unit in note_ir.body.teaching_units:
        page_anchor[unit.unit_id] = {"unit_id": unit.unit_id}
        for block in unit.content_blocks:
            page_anchor[block.block_id] = {"block_id": block.block_id, "unit_id": unit.unit_id}

    for chapter in lecture.chapters:
        unit = unit_by_chapter.get(chapter.index)
        unit_id = unit.unit_id if unit is not None else f"teaching-unit-{chapter.index}"
        chapter_id = f"chapter-{chapter.index}"
        chapter_map.append(
            {
                "chapter_index": chapter.index,
                "title": chapter.title,
                "start": float(chapter.start),
                "end": float(chapter.end),
            }
        )
        _register_anchor(
            evidence_anchor,
            video_anchor,
            page_anchor,
            evidence_id=chapter_id,
            note_node_id=unit_id,
            t_start=float(chapter.start),
            t_end=float(chapter.end),
        )
        evidence_objects.append(
            EvidenceObject(
                evidence_id=chapter_id,
                kind="chapter_evidence",
                title=chapter.title,
                summary=chapter.summary,
                chapter_index=chapter.index,
                text=chapter.summary,
                ts=float(chapter.start),
                t_end=float(chapter.end),
                note_node_ids=[unit_id],
                anchors=evidence_anchor[chapter_id],
                rag_chunks=[
                    RagChunk(
                        kind="teaching_note",
                        chapter_idx=chapter.index,
                        t_start=float(chapter.start),
                        t_end=float(chapter.end),
                        text=chapter.summary or chapter.learning_goal or chapter.title,
                        note_node_id=unit_id,
                    )
                ],
                source_payload={"chapter_title": chapter.title},
            )
        )
        evidence_relations.append(
            EvidenceRelation(relation="supports", from_id=chapter_id, to_id=unit_id)
        )
        source_index_view.append(
            {
                "label": _time_label(chapter.start),
                "supports_text": f"支撑讲义单元：{unit.title if unit is not None else chapter.title}",
                "evidence_id": chapter_id,
                "page_anchor": unit_id,
                "video_anchor": evidence_anchor[chapter_id]["video_anchor"],
                "kind": "chapter_evidence",
            }
        )

        for point_index, point in enumerate(chapter.points, start=1):
            evidence_id = f"quote-{chapter.index}-{point_index}"
            block_id = _best_block_id(unit, point_index)
            _register_anchor(
                evidence_anchor,
                video_anchor,
                page_anchor,
                evidence_id=evidence_id,
                note_node_id=block_id,
                t_start=float(point.ts),
            )
            evidence_objects.append(
                EvidenceObject(
                    evidence_id=evidence_id,
                    kind="quote_evidence",
                    title=point.text,
                    summary=point.text,
                    chapter_index=chapter.index,
                    quote=point.quote,
                    text=point.text,
                    ts=float(point.ts),
                    note_node_ids=[unit_id, block_id],
                    anchors=evidence_anchor[evidence_id],
                    rag_chunks=[
                        RagChunk(
                            kind="quote",
                            chapter_idx=chapter.index,
                            t_start=float(point.ts),
                            t_end=float(point.ts),
                            text=point.quote or point.text,
                            note_node_id=block_id,
                        )
                    ],
                    source_payload={"quote": point.quote, "point_text": point.text},
                )
            )
            evidence_relations.append(
                EvidenceRelation(relation="supports", from_id=evidence_id, to_id=block_id)
            )
            evidence_quotes.append(
                {
                    "label": point.text,
                    "quote": point.quote,
                    "ts": float(point.ts),
                    "chapter_index": chapter.index,
                }
            )
            source_index_view.append(
                {
                    "label": _time_label(point.ts),
                    "supports_text": f"支撑正文块：{block_id}",
                    "evidence_id": evidence_id,
                    "page_anchor": block_id,
                    "video_anchor": evidence_anchor[evidence_id]["video_anchor"],
                    "kind": "quote_evidence",
                }
            )

        for frame_index, frame in enumerate(chapter.frames, start=1):
            if not frame.path:
                continue
            evidence_id = f"frame-{chapter.index}-{frame_index}"
            _register_anchor(
                evidence_anchor,
                video_anchor,
                page_anchor,
                evidence_id=evidence_id,
                note_node_id=unit_id,
                t_start=float(frame.ts),
            )
            evidence_objects.append(
                EvidenceObject(
                    evidence_id=evidence_id,
                    kind="frame_evidence",
                    title=frame.caption or f"Chapter {chapter.index} frame {frame_index}",
                    summary=frame.insight or frame.caption,
                    chapter_index=chapter.index,
                    text=frame.caption,
                    ts=float(frame.ts),
                    path=frame.path,
                    note_node_ids=[unit_id],
                    anchors=evidence_anchor[evidence_id],
                    rag_chunks=[
                        RagChunk(
                            kind="frame_ocr",
                            chapter_idx=chapter.index,
                            t_start=float(frame.ts),
                            t_end=float(frame.ts),
                            text=frame.ocr_text or frame.caption,
                            note_node_id=unit_id,
                        )
                    ],
                    source_payload={"caption": frame.caption, "ocr_text": frame.ocr_text},
                )
            )
            evidence_relations.append(
                EvidenceRelation(relation="supports", from_id=evidence_id, to_id=unit_id)
            )

        for step_index, step in enumerate(chapter.process_steps, start=1):
            evidence_id = f"procedure-{chapter.index}-{step_index}"
            _register_anchor(
                evidence_anchor,
                video_anchor,
                page_anchor,
                evidence_id=evidence_id,
                note_node_id=unit_id,
                t_start=float(chapter.start),
                t_end=float(chapter.end),
            )
            evidence_objects.append(
                EvidenceObject(
                    evidence_id=evidence_id,
                    kind="procedure_evidence",
                    title=step,
                    summary=step,
                    chapter_index=chapter.index,
                    text=step,
                    ts=float(chapter.start),
                    t_end=float(chapter.end),
                    note_node_ids=[unit_id],
                    anchors=evidence_anchor[evidence_id],
                    rag_chunks=[
                        RagChunk(
                            kind="teaching_note",
                            chapter_idx=chapter.index,
                            t_start=float(chapter.start),
                            t_end=float(chapter.end),
                            text=step,
                            note_node_id=unit_id,
                        )
                    ],
                    source_payload={"step_index": step_index},
                )
            )
            evidence_relations.append(
                EvidenceRelation(relation="supports", from_id=evidence_id, to_id=unit_id)
            )

    for glossary_index, item in enumerate(lecture.glossary, start=1):
        evidence_id = f"concept-{glossary_index}"
        _register_anchor(
            evidence_anchor,
            video_anchor,
            page_anchor,
            evidence_id=evidence_id,
            note_node_id=note_ir.body.teaching_units[0].unit_id if note_ir.body.teaching_units else "",
            t_start=float(item.ts),
        )
        evidence_objects.append(
            EvidenceObject(
                evidence_id=evidence_id,
                kind="concept_evidence",
                title=item.term,
                summary=item.explanation,
                text=item.explanation,
                ts=float(item.ts),
                note_node_ids=[note_ir.body.teaching_units[0].unit_id] if note_ir.body.teaching_units else [],
                anchors=evidence_anchor[evidence_id],
                rag_chunks=[
                    RagChunk(
                        kind="knowledge_unit",
                        chapter_idx=1,
                        t_start=float(item.ts),
                        t_end=float(item.ts),
                        text=f"{item.term}：{item.explanation}",
                        note_node_id=note_ir.body.teaching_units[0].unit_id if note_ir.body.teaching_units else "",
                    )
                ],
                source_payload={"term": item.term},
            )
        )
        term_index.append(
            {"term": item.term, "ts": float(item.ts), "explanation": item.explanation}
        )
        glossary_view.append(
            {"term": item.term, "explanation": item.explanation, "evidence_id": evidence_id}
        )

    for plan in blueprint.body_unit_plan:
        compound_id = f"compound-{plan.unit_id}"
        _register_anchor(
            evidence_anchor,
            video_anchor,
            page_anchor,
            evidence_id=compound_id,
            note_node_id=plan.unit_id,
        )
        evidence_objects.append(
            EvidenceObject(
                evidence_id=compound_id,
                kind="compound_evidence",
                title=plan.title,
                summary=plan.core_message,
                text=plan.core_message,
                note_node_ids=[plan.unit_id],
                anchors=evidence_anchor[compound_id],
                source_payload={"source_chapter_refs": plan.source_chapter_refs},
            )
        )
        evidence_relations.append(
            EvidenceRelation(relation="supports", from_id=compound_id, to_id=plan.unit_id)
        )
        for chapter_index in plan.source_chapter_refs:
            evidence_relations.append(
                EvidenceRelation(
                    relation="derived_from",
                    from_id=plan.unit_id,
                    to_id=f"chapter-{chapter_index}",
                )
            )

    return EvidenceIndex(
        evidence_objects=evidence_objects,
        evidence_relations=evidence_relations,
        anchor_map={
            "video_anchor": video_anchor,
            "page_anchor": page_anchor,
            "evidence_anchor": evidence_anchor,
        },
        projection_views={
            "source_index_view": source_index_view,
            "glossary_view": glossary_view,
            "appendix_view": blueprint.appendix_candidates,
            "chapter_map": chapter_map,
            "evidence_quotes": evidence_quotes[:6],
            "term_index": term_index[:8],
            "tool_appendix": [],
        },
    )


def _register_anchor(
    evidence_anchor: dict[str, Any],
    video_anchor: dict[str, Any],
    page_anchor: dict[str, Any],
    *,
    evidence_id: str,
    note_node_id: str,
    t_start: float | None = None,
    t_end: float | None = None,
) -> None:
    if t_start is not None:
        video_key = f"t={int(round(t_start))}"
        video_anchor[video_key] = {"t_start": float(t_start), "t_end": float(t_end if t_end is not None else t_start)}
    else:
        video_key = ""
    evidence_anchor[evidence_id] = {
        "video_anchor": video_key,
        "page_anchor": note_node_id,
        "evidence_anchor": evidence_id,
    }
    if note_node_id and note_node_id not in page_anchor:
        page_anchor[note_node_id] = {"node_id": note_node_id}


def _best_block_id(unit: Any, point_index: int) -> str:
    if unit is None or not unit.content_blocks:
        return ""
    block = unit.content_blocks[min(point_index - 1, len(unit.content_blocks) - 1)]
    return block.block_id


def _time_label(ts: float | int) -> str:
    total = max(int(ts or 0), 0)
    minutes, seconds = divmod(total, 60)
    return f"[t={minutes:02d}:{seconds:02d}]"
