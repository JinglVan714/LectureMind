from __future__ import annotations

from app.understand.ir import IRChapter, IRFrame, IRPoint, KnowledgeUnit, LectureIR, LectureProfile
from app.understand.lecture_compiler import compile_lecture
from scripts._verify_note_evidence_real_sample import validate_rag_backlink_chain


def _sample_ir() -> LectureIR:
    return LectureIR(
        bv_id="BVevidence",
        title="Evidence sample",
        duration=600,
        profile=LectureProfile(primary_type="procedural_tutorial"),
        core_question="这个流程为什么成立？",
        mainline=["先定义输入", "再执行流程"],
        final_synthesis="这条视频先定义输入，再解释流程和边界。",
        knowledge_units=[
            KnowledgeUnit(
                id="ku-1",
                type="concept",
                title="Mask",
                term_short="mask",
                explanation="用于屏蔽不该参与计算的位置。",
                ts=80,
                chapter_index=1,
            )
        ],
        chapters=[
            IRChapter(
                index=1,
                title="输入和匹配分数",
                start=0,
                end=300,
                summary="先准备输入，再计算匹配分数。",
                learning_goal="理解输入和分数",
                teaching_notes=["输入阶段决定了后续能否正确比较。"],
                process_steps=["准备输入", "计算匹配分数"],
                points=[IRPoint(text="先比较再归一化", ts=45, quote="先比较再归一化")],
                frames=[
                    IRFrame(
                        ts=50,
                        path="keyframes/BVevidence/00050.jpg",
                        caption="输入流程图",
                        ocr_text="input score",
                        insight="图里把输入和分数计算连在一起。",
                        visual_type="diagram",
                    )
                ],
            ),
            IRChapter(
                index=2,
                title="归一化和边界",
                start=300,
                end=600,
                summary="再做归一化，并处理边界。",
                learning_goal="理解归一化和边界",
                teaching_notes=["归一化之前和之后的边界处理不同。"],
                points=[IRPoint(text="mask 位置会改变结果", ts=420, quote="mask 位置会改变结果")],
                pitfalls=["不要把 mask 放在 softmax 之后。"],
            ),
        ],
    )


def test_evidence_index_contains_anchors_relations_and_rag_backlinks():
    lecture = compile_lecture(_sample_ir())
    evidence = lecture.evidence_index

    assert evidence is not None
    assert evidence.evidence_objects
    assert any(obj.kind == "quote_evidence" for obj in evidence.evidence_objects)
    assert any(obj.kind == "frame_evidence" for obj in evidence.evidence_objects)
    assert any(obj.kind == "compound_evidence" for obj in evidence.evidence_objects)
    assert any(rel.relation == "supports" for rel in evidence.evidence_relations)
    assert "evidence_anchor" in evidence.anchor_map
    quote_objects = [obj for obj in evidence.evidence_objects if obj.kind == "quote_evidence"]
    assert quote_objects
    assert all(obj.rag_chunks for obj in quote_objects)
    assert all(chunk.note_node_id for obj in quote_objects for chunk in obj.rag_chunks)


def test_evidence_projection_preserves_legacy_source_index_compatibility_keys():
    lecture = compile_lecture(_sample_ir())
    evidence = lecture.evidence_index

    assert evidence is not None
    assert "source_index_view" in evidence.projection_views
    assert "chapter_map" in evidence.projection_views
    assert "evidence_quotes" in evidence.projection_views
    assert "term_index" in evidence.projection_views
    assert lecture.composition.source_index["chapter_map"]
    assert lecture.composition.source_index["source_index_view"]


def test_rag_backlink_chain_validator_accepts_evidence_to_note_roundtrip():
    lecture = compile_lecture(_sample_ir())
    evidence = lecture.evidence_index

    assert evidence is not None
    indexed_rows = []
    lecture_hits = []
    evidence_hits = []
    for idx, obj in enumerate(evidence.evidence_objects, start=1):
        for rag_chunk in obj.rag_chunks:
            note_node_id = rag_chunk.note_node_id or (obj.note_node_ids[0] if obj.note_node_ids else "")
            meta = {
                "evidence_id": obj.evidence_id,
                "note_node_id": note_node_id,
                "evidence_kind": obj.kind,
            }
            indexed_rows.append(
                {
                    "chunk_id": idx,
                    "kind": rag_chunk.kind,
                    "text": rag_chunk.text,
                    "meta": meta,
                }
            )
            lecture_hits.append(
                {
                    "evidence_id": obj.evidence_id,
                    "note_node_id": note_node_id,
                    "text": rag_chunk.text,
                }
            )
        evidence_hits.append(
            {
                "evidence_id": obj.evidence_id,
                "note_node_ids": list(obj.note_node_ids),
            }
        )

    result = validate_rag_backlink_chain(
        lecture=lecture,
        indexed_chunk_rows=indexed_rows,
        lecture_hits=lecture_hits,
        evidence_hits=evidence_hits,
    )

    assert result.ok is True
    assert result.source_rag_chunk_count >= 1
    assert result.indexed_backlinked_chunk_count >= 1
    assert result.searched_backlinked_hit_count >= 1
    assert result.searched_evidence_hit_count >= 1
    assert result.failures == []


def test_rag_backlink_chain_validator_rejects_missing_note_node_backlink():
    lecture = compile_lecture(_sample_ir())
    evidence = lecture.evidence_index

    assert evidence is not None
    quote_obj = next(obj for obj in evidence.evidence_objects if obj.rag_chunks)
    rag_chunk = quote_obj.rag_chunks[0]
    indexed_rows = [
        {
            "chunk_id": 1,
            "kind": rag_chunk.kind,
            "text": rag_chunk.text,
            "meta": {
                "evidence_id": quote_obj.evidence_id,
                "evidence_kind": quote_obj.kind,
            },
        }
    ]

    result = validate_rag_backlink_chain(
        lecture=lecture,
        indexed_chunk_rows=indexed_rows,
        lecture_hits=[
            {
                "evidence_id": quote_obj.evidence_id,
                "text": rag_chunk.text,
            }
        ],
        evidence_hits=[
            {
                "evidence_id": quote_obj.evidence_id,
                "note_node_ids": [],
            }
        ],
    )

    assert result.ok is False
    assert any("missing `note_node_id`" in item for item in result.failures)
    assert any("returned no `note_node_ids` backlinks" in item for item in result.failures)
