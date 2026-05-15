from __future__ import annotations

import json
from pathlib import Path

from app.understand.ir import IRChapter, IRFrame, IRPoint, KnowledgeUnit, LectureIR, LectureProfile
from app.understand.lecture_compiler import compile_lecture
from app.understand.schema import LectureJSON


def _sample_ir() -> LectureIR:
    return LectureIR(
        bv_id="BVcompiler",
        title="QKV 讲义编译样本",
        duration=840,
        profile=LectureProfile(primary_type="conceptual_talk", domain_tags=["Transformer"]),
        core_question="为什么 QKV 是理解注意力机制的入口？",
        mainline=["先理解 QKV 分工", "再理解权重计算", "最后收束实现边界"],
        final_synthesis="这条视频把注意力机制拆成主线、计算过程和实现边界三层。",
        review_questions=["QKV 如何改变信息路由？"],
        study_questions=["为什么 softmax 位置会改变结果？"],
        knowledge_units=[
            KnowledgeUnit(
                id="ku-1",
                type="concept",
                title="Query Key Value",
                term_short="QKV",
                explanation="注意力机制中的查询、键和值三元组。",
                ts=12,
                chapter_index=1,
            )
        ],
        chapters=[
            IRChapter(
                index=1,
                title="先把 QKV 说明白",
                start=0,
                end=360,
                summary="先定义 Query、Key、Value 各自承担的角色，再说明为什么不能混用。",
                learning_goal="理解 QKV 的角色分工。",
                teaching_notes=["Query 负责发问，Key 负责匹配，Value 负责携带被聚合的信息。"],
                points=[IRPoint(text="QKV 分工不同", ts=30, quote="QKV 分工不同")],
                key_takeaways=["QKV 决定了信息检索和聚合的路径。"],
                frames=[
                    IRFrame(
                        ts=48,
                        path="keyframes/BVcompiler/00048.jpg",
                        caption="QKV 关系图",
                        ocr_text="Query Key Value",
                        insight="这张图把三者的分工放在一张关系图里。",
                        visual_type="diagram",
                    )
                ],
            ),
            IRChapter(
                index=2,
                title="再看权重和边界",
                start=360,
                end=840,
                summary="继续解释注意力权重如何计算，以及 softmax 和 mask 放错位置会带来的偏差。",
                learning_goal="理解权重计算与实现边界。",
                teaching_notes=["权重计算是把匹配分数变成可比较的分布。"],
                process_steps=["计算匹配分数", "做 mask", "归一化为权重"],
                points=[IRPoint(text="softmax 位置很关键", ts=540, quote="softmax 位置很关键")],
                pitfalls=["不要把 mask 放在 softmax 之后。"],
                key_takeaways=["边界处理错误会直接改变最终权重。"],
            ),
        ],
    )


def _generic_talk_with_redundant_steps() -> LectureIR:
    return LectureIR(
        bv_id="BVgeneric",
        title="Agent Skills 讲义样本",
        duration=600,
        profile=LectureProfile(primary_type="generic_lecture", domain_tags=["Agent"]),
        core_question="Agent Skills 到底解决了什么问题？",
        mainline=["介绍概念", "解释工作方式"],
        chapters=[
            IRChapter(
                index=1,
                title="Agent Skills 简介",
                start=0,
                end=180,
                summary="本章先说明 Agent Skills 是什么，再解释它如何按需加载说明与脚本。",
                learning_goal="理解 Agent Skills 的基本工作方式。",
                teaching_notes=[
                    "Agent 启动时只加载 Skill 的简短描述，命中后才继续读取具体说明与脚本。",
                    "这种按需展开的方式降低了上下文负担，也避免把所有材料一次性塞进正文。",
                ],
                process_steps=[
                    "启动时只加载 Skill 描述",
                    "命中后再读取具体说明",
                    "必要时再执行脚本",
                ],
                pitfalls=["Skill 描述不清楚时，Agent 容易命中错误工具。"],
                key_takeaways=["核心不是堆更多文件，而是按需展开上下文。"],
                points=[IRPoint(text="按需加载", ts=24, quote="按需加载")],
            )
        ],
    )


def test_lecture_json_carries_parallel_note_and_evidence_truths():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVparallel",
            "title": "Compiler sample",
            "duration": 1200,
            "one_liner": "核心主张",
            "chapters": [{"index": 1, "title": "章节", "start": 0, "end": 1200, "summary": "章节摘要"}],
            "lecture_blueprint": {
                "front_matter_plan": {"one_sentence_claim": "核心主张"},
                "body_unit_plan": [{"unit_id": "u1", "unit_role": "claim"}],
                "back_matter_plan": {"appendices": []},
                "reorder_strength": "light",
                "visual_needs": [],
                "appendix_candidates": [],
            },
            "lecture_note_ir": {
                "front_matter": {"one_sentence_claim": "核心主张"},
                "body": {"teaching_units": [{"unit_id": "u1", "title": "为什么先讲判断"}]},
                "back_matter": {"appendices": []},
            },
            "evidence_index": {
                "evidence_objects": [{"evidence_id": "ev-1", "kind": "quote_evidence"}],
                "evidence_relations": [{"relation": "supports", "from_id": "ev-1", "to_id": "u1"}],
                "anchor_map": {"evidence_anchor": {"ev-1": {"video_anchor": "t=120"}}},
                "projection_views": {"source_index_view": []},
            },
        }
    )

    assert lecture.lecture_blueprint is not None
    assert lecture.lecture_note_ir is not None
    assert lecture.evidence_index is not None
    assert lecture.composition is not None


def test_compile_lecture_emits_blueprint_note_and_evidence_in_order():
    lecture_ir = LectureIR.model_validate(
        json.loads(Path("data/debug/BV1ypdgBCE9B.lecture_ir.json").read_text(encoding="utf-8"))
    )

    lecture = compile_lecture(lecture_ir)

    assert lecture.lecture_blueprint is not None
    assert lecture.lecture_blueprint.body_unit_plan
    assert lecture.lecture_note_ir is not None
    assert lecture.lecture_note_ir.body["teaching_units"]
    assert lecture.evidence_index is not None
    assert lecture.evidence_index.evidence_objects
    assert lecture.composition.summary_mode in {"skim", "note", "handout", "segmented_handout"}


def test_composition_becomes_compatibility_projection_after_compile():
    lecture = compile_lecture(_sample_ir())

    assert lecture.lecture_note_ir is not None
    assert lecture.evidence_index is not None
    assert lecture.composition.meta["source"] == "compat_projection"
    assert [section.id for section in lecture.composition.body_sections] == [
        unit.unit_id for unit in lecture.lecture_note_ir.body.teaching_units
    ]
    assert lecture.chapters


def test_conceptual_unit_uses_single_natural_block_and_dedupes_summary():
    lecture = compile_lecture(_sample_ir())

    conceptual_unit = lecture.lecture_note_ir.body.teaching_units[0]

    assert conceptual_unit.core_message == ""
    assert len(conceptual_unit.content_blocks) == 1
    assert conceptual_unit.content_blocks[0].title == ""
    assert conceptual_unit.content_blocks[0].block_role == "claim"


def test_procedural_unit_keeps_real_process_expansion():
    lecture = compile_lecture(_sample_ir())

    procedural_unit = lecture.lecture_note_ir.body.teaching_units[1]
    process_blocks = [block for block in procedural_unit.content_blocks if block.block_role == "process"]

    assert process_blocks
    assert process_blocks[0].title == "步骤"
    assert process_blocks[0].paragraphs == ["计算匹配分数", "做 mask", "归一化为权重"]
    assert all(
        block.title not in {"核心内容", "展开讲解", "讲义结论"}
        for block in procedural_unit.content_blocks
    )


def test_generic_talk_does_not_force_redundant_process_or_boundary_blocks():
    lecture = compile_lecture(_generic_talk_with_redundant_steps())

    unit = lecture.lecture_note_ir.body.teaching_units[0]

    assert [block.block_role for block in unit.content_blocks] == ["claim"]
    assert any("注意：" in paragraph for paragraph in unit.content_blocks[0].paragraphs)
    assert all(block.title == "" for block in unit.content_blocks)
