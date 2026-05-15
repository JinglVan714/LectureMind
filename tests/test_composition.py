from __future__ import annotations

from app.understand.composition import compose_from_lecture_json
from app.understand.ir import IRChapter, IRPoint, LectureIR, LectureProfile
from app.understand.ir_builder import lecture_ir_to_lecture_json
from app.understand.schema import LectureJSON


def _all_outline_paragraphs(lecture_json: LectureJSON) -> list[str]:
    return [
        paragraph
        for section in lecture_json.composition.body_sections
        for block in section.outline_blocks
        for paragraph in ([block.lead] if block.lead else []) + block.paragraphs
    ]


def test_short_but_dense_lecture_upgrades_to_handout():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVdense",
            "title": "5 分钟高密技术讲解",
            "duration": 300,
            "one_liner": "讲清 QKV 的职责分工、mask 的边界和实现中的常见坑。",
            "chapters": [
                {
                    "index": 1,
                    "title": "QKV 结构",
                    "start": 0,
                    "end": 150,
                    "summary": "解释 QKV 各自处理什么信息，以及它们为什么不能混用。",
                    "points": [{"text": "QKV 分工不同", "ts": 12, "quote": "QKV 分工不同"}],
                    "key_takeaways": ["QKV 决定信息路由", "mask 会改变注意力边界"],
                },
                {
                    "index": 2,
                    "title": "实现边界",
                    "start": 150,
                    "end": 300,
                    "summary": "解释 broadcasting、shape 对齐和 softmax 位置这些实现边界。",
                    "points": [{"text": "广播规则常出错", "ts": 220, "quote": "广播规则常出错"}],
                    "key_takeaways": ["shape 对不上就会炸", "softmax 位置很关键"],
                    "pitfalls": ["不要把 mask 放在 softmax 之后"],
                },
            ],
            "study_questions": ["QKV 为什么不能混用？", "mask 为什么会改变结果？"],
            "review_questions": ["QKV 为什么不能混用？", "mask 为什么会改变结果？"],
            "glossary": [{"term": "QKV", "ts": 12, "explanation": "查询、键和值"}],
            "final_synthesis": "这条视频解释了 QKV 的作用、边界和落地实现时最容易踩坑的地方。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    assert composed.summary_mode == "handout"
    assert composed.reading_goal == "standalone_dense_handout"
    assert len(composed.body_sections) >= 3


def test_long_but_sparse_interview_stays_note():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVsparse",
            "title": "50 分钟轻访谈",
            "duration": 3000,
            "one_liner": "聊聊近期工作体会。",
            "chapters": [
                {
                    "index": 1,
                    "title": "聊天",
                    "start": 0,
                    "end": 3000,
                    "summary": "整体是松散交流，没有高密度教学结构。",
                    "points": [{"text": "以经验分享为主", "ts": 30, "quote": "以经验分享为主"}],
                }
            ],
            "final_synthesis": "主要是经验交流，不是高密教学。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    assert composed.summary_mode == "note"
    assert composed.burden_score < 0.45


def test_note_mode_keeps_full_video_chapter_order():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVnoteorder",
            "title": "按顺序讲完四个观察",
            "duration": 1800,
            "one_liner": "按顺序解释四个观察点。",
            "chapters": [
                {"index": 1, "title": "观察一", "start": 0, "end": 300, "summary": "先立住第一个观察。"},
                {"index": 2, "title": "观察二", "start": 300, "end": 700, "summary": "再解释第二个观察。"},
                {"index": 3, "title": "观察三", "start": 700, "end": 1200, "summary": "随后进入第三个观察。"},
                {"index": 4, "title": "观察四", "start": 1200, "end": 1800, "summary": "最后收在第四个观察。"},
            ],
            "final_synthesis": "整体是按顺序一段段往下展开，不需要打乱。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    chapter_sections = [section.id for section in composed.body_sections if section.id.startswith("chapter-")]
    assert chapter_sections == ["chapter-1", "chapter-2", "chapter-3", "chapter-4"]


def test_projection_attaches_composition_to_lecture_json():
    ir = LectureIR(
        bv_id="BVprojection",
        title="Harness 实战",
        duration=1401.0,
        profile=LectureProfile(primary_type="conceptual_talk"),
        final_synthesis="Harness 解决复杂 Agent 任务的可控性问题。",
        chapters=[
            IRChapter(
                index=1,
                title="为什么选网页",
                start=0.0,
                end=120.0,
                summary="作者强调网页方案的可控性更强。",
                learning_goal="理解为什么网页比视频模型更可控。",
                points=[IRPoint(text="网页方案可控", ts=34.0, quote="答案是两个字，可控")],
                key_takeaways=["网页方案更可控"],
            )
        ],
        study_questions=["为什么网页方案更可控？"],
    )

    lecture = lecture_ir_to_lecture_json(ir)

    assert lecture.composition.summary_mode in {"skim", "note", "handout", "segmented_handout"}
    assert lecture.composition.body_sections
    assert lecture.study_questions
    assert lecture.review_questions


def test_composition_emits_keyframe_visual_plan_for_ui_heavy_section():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVui",
            "title": "UI 工作流演示",
            "duration": 480,
            "one_liner": "展示一个多步骤网页工作流。",
            "chapters": [
                {
                    "index": 1,
                    "title": "工作流",
                    "start": 0,
                    "end": 480,
                    "summary": "重点在界面结构和操作顺序。",
                }
            ],
            "visual_evidence": [
                {
                    "ts": 45,
                    "path": "keyframes/BVui/00045.jpg",
                    "caption": "工作流总览",
                    "selected_reason": "界面信息密度高",
                    "importance_score": 0.9,
                }
            ],
            "final_synthesis": "界面结构本身就是主要信息载体。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    assert composed.visual_plan["items"]
    assert composed.visual_plan["items"][0]["visual_role"] in {"keyframe", "flowchart", "concept_map"}


def test_composition_rewrites_grouped_sections_as_prose_without_field_labels():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVgrouped",
            "title": "Harness 实战：从文章到视频",
            "duration": 1500,
            "one_liner": "用 Harness 把文章稳定变成知识讲解视频。",
            "chapters": [
                {
                    "index": 1,
                    "title": "开场与问题引入",
                    "start": 0,
                    "end": 200,
                    "summary": "本章通过评论区提问把问题立住，并说明后文会完整展示视频生成流程。",
                    "key_takeaways": ["这期视频要解决的是视频效果如何稳定做出来。"],
                },
                {
                    "index": 2,
                    "title": "视频制作思路与可控性设计",
                    "start": 200,
                    "end": 400,
                    "summary": "本章解释为什么作者放弃视频生成模型，转而使用网页方案来追求可控性。",
                    "key_takeaways": ["网页方案比视频生成模型更可控。"],
                    "pitfalls": ["不要把风格一致性交给黑盒视频模型。"],
                },
                {
                    "index": 3,
                    "title": "视频制作流程的四个关键步骤",
                    "start": 400,
                    "end": 650,
                    "summary": "本章把文章转视频拆成口播稿、开发大纲、视觉演示和节奏对齐四步。",
                    "key_takeaways": ["真正要控制的是从稿到画再到节奏的四步链路。"],
                },
                {
                    "index": 4,
                    "title": "Skill 工作流详解：从文章到视频的四阶段与人工检查点",
                    "start": 650,
                    "end": 950,
                    "summary": "本章解释 Skill 如何把四步链路落实成四阶段工作流，并插入两个检查点。",
                    "key_takeaways": ["Skill 工作流负责把 Agent 产出接成稳定产线。"],
                },
                {
                    "index": 5,
                    "title": "Skill 设计拆解与实战环境搭建",
                    "start": 950,
                    "end": 1200,
                    "summary": "本章拆解 Skill 的设计维度，并给出环境搭建前提。",
                    "key_takeaways": ["上下文、记忆、工具、约束、观测共同组成 Harness。"],
                },
                {
                    "index": 6,
                    "title": "实战演示：从文章到视频的完整流程",
                    "start": 1200,
                    "end": 1500,
                    "summary": "本章完整演示从文章到视频的自动化跑通流程。",
                    "key_takeaways": ["Harness 的价值在于把模型、工具和人的判断编排成稳定流程。"],
                },
            ],
            "study_questions": ["为什么网页方案比视频模型更可控？"],
            "final_synthesis": (
                "视频先立住问题，再解释为什么要选网页方案。"
                "之后把生产链路拆成四步，再用 Skill 工作流和检查点把它们接成稳定产线。"
                "最后给出设计维度、实战跑通路径，以及可以迁移到其他内容生产任务的经验。"
            ),
        }
    )

    composed = compose_from_lecture_json(lecture)

    assert composed.summary_mode in {"handout", "segmented_handout"}
    assert all(" / " not in section.title for section in composed.body_sections)
    assert all(not section.summary.startswith("本章") for section in composed.body_sections if section.summary)
    assert all(not section.summary.startswith("本章节") for section in composed.body_sections if section.summary)
    assert all(not section.summary.startswith("介绍了") for section in composed.body_sections if section.summary)
    assert all(not section.summary.startswith("详细介绍了") for section in composed.body_sections if section.summary)
    grouped_section = next(
        section
        for section in composed.body_sections
        if len(section.outline_blocks) >= 2
    )
    assert len(grouped_section.outline_blocks) >= 2
    assert [block.ordinal for block in grouped_section.outline_blocks] == list(
        range(1, len(grouped_section.outline_blocks) + 1)
    )
    assert all(block.title for block in grouped_section.outline_blocks)
    assert all(block.source_chapter_refs for block in grouped_section.outline_blocks)
    assert all(block.source_timestamps for block in grouped_section.outline_blocks)
    all_paragraphs = _all_outline_paragraphs(lecture.model_copy(update={"composition": composed}))
    assert all("目标：" not in paragraph for paragraph in all_paragraphs)
    assert all("带走：" not in paragraph for paragraph in all_paragraphs)
    assert all("证据：" not in paragraph for paragraph in all_paragraphs)
    assert all("这一段的关键结论是" not in paragraph for paragraph in all_paragraphs)
    assert all("这里的核心结论是" not in paragraph for paragraph in all_paragraphs)
    assert all("这里真正要抓的是" not in paragraph for paragraph in all_paragraphs)
    assert all("关键在于" not in paragraph for paragraph in all_paragraphs)
    assert all("在“" not in paragraph for paragraph in all_paragraphs)
    assert all("…" not in paragraph and "..." not in paragraph for paragraph in all_paragraphs)


def test_composition_normalizes_noisy_asr_terms_and_drops_editorial_repair_notes():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVnoise",
            "title": "Skill 工作流与 Harness",
            "duration": 1320,
            "one_liner": "解释如何把文章稳定变成知识讲解视频。",
            "chapters": [
                {
                    "index": 1,
                    "title": "Skill 工作流",
                    "start": 0,
                    "end": 660,
                    "summary": "本章解释 Skill 如何把可不搞、开外计划和检查点接成稳定流程。",
                    "key_takeaways": [
                        "可不搞决定信息的描述方式和整体节奏。",
                        "开外计划决定每张做几步，每步屏幕上放什么内容。",
                    ],
                    "pitfalls": [
                        "注意注意：outline 和脚本的术语在字幕中为“可不搞”和“开外计划”，可能是语音识别错误。",
                    ],
                },
                {
                    "index": 2,
                    "title": "设计维度",
                    "start": 660,
                    "end": 1320,
                    "summary": "本章拆解上下门管理、状态和记忆、工具系统等设计维度。",
                    "key_takeaways": ["上下门管理的重点是分阶段读取信息，避免注意力被稀释。"],
                },
            ],
            "final_synthesis": "先用 Skill 工作流接住内容生产，再用 Harness 保证过程稳定可控。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    all_paragraphs = [
        paragraph
        for section in composed.body_sections
        for block in section.outline_blocks
        for paragraph in ([block.lead] if block.lead else []) + block.paragraphs
    ]
    assert all("可不搞" not in paragraph for paragraph in all_paragraphs)
    assert all("开外计划" not in paragraph for paragraph in all_paragraphs)
    assert all("上下门管理" not in paragraph for paragraph in all_paragraphs)
    assert all("语音识别错误" not in paragraph for paragraph in all_paragraphs)
    assert any("口播稿" in paragraph for paragraph in all_paragraphs)
    assert any("开发大纲" in paragraph for paragraph in all_paragraphs)
    assert any("上下文管理" in paragraph for paragraph in all_paragraphs)


def test_composition_skips_tooling_side_detail_when_summary_is_already_dense():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVdensechapter",
            "title": "完整流程演示",
            "duration": 1200,
            "one_liner": "演示一条完整的文章转视频流程。",
            "chapters": [
                {
                    "index": 1,
                    "title": "完整流程演示",
                    "start": 0,
                    "end": 1200,
                    "summary": (
                        "本章完整演示了从文章到知识讲解视频的自动化生成流程，先配置环境，再生成口播稿和开发大纲，"
                        "接着进入多 Agent 开发、验收、合成音频和录制收尾。"
                        "最后回到 Harness 的价值，强调真正重要的是把模型、工具和人的判断组织成稳定流程。"
                    ),
                    "key_takeaways": [
                        "Cline 的 Slidescape 模式可全自动执行，但需注意安全。",
                        "Harness 的价值在于把模型、工具和人的判断编排成稳定流程。",
                    ],
                }
            ],
            "final_synthesis": "关键不在单个工具，而在整条流程是否稳定可控。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    paragraphs = next(section.paragraphs for section in composed.body_sections if section.id == "chapter-1")
    assert all("Slidescape" not in paragraph for paragraph in paragraphs)
    assert any("Harness" in paragraph for paragraph in paragraphs)


def test_composition_splits_dense_single_chapter_into_multiple_outline_blocks():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVoutline",
            "title": "Embedding 检索拆解",
            "duration": 900,
            "one_liner": "把 embedding 检索的目标、召回边界和排序信号拆开讲清楚。",
            "profile": {"primary_type": "procedural_explainer"},
            "chapters": [
                {
                    "index": 1,
                    "title": "Embedding 检索拆解",
                    "start": 0,
                    "end": 900,
                    "summary": (
                        "这一节先解释 embedding 检索到底在解决什么问题，再把召回、粗排和精排各自负责的判断拆开。"
                        "后半段继续说明查询改写、相似度阈值和排序特征应该怎么配合，避免把所有问题都压给向量距离。"
                    ),
                    "learning_goal": "知道检索链路里每一段该负责什么。",
                    "points": [{"text": "召回不要替排序做决定", "ts": 240, "quote": "召回不要替排序做决定"}],
                    "key_takeaways": ["召回不要替排序做决定", "阈值和特征要一起调"],
                    "pitfalls": ["不要把所有质量判断都压给 embedding 相似度"],
                    "process_steps": ["先定召回目标", "再看排序信号怎么接上"],
                }
            ],
        }
    )

    composed = compose_from_lecture_json(lecture)

    chapter_section = next(section for section in composed.body_sections if section.id == "chapter-1")
    assert len(chapter_section.outline_blocks) >= 2
    assert chapter_section.outline_blocks[0].title == "Embedding 检索拆解"
    assert chapter_section.outline_blocks[1].title not in {"召回不要替排序做决定", "阈值和特征要一起调"}
    assert any(
        token in chapter_section.outline_blocks[1].title
        for token in ("查询改写", "阈值", "排序", "召回")
    )
    assert len(chapter_section.outline_blocks[1].title) < len(chapter_section.outline_blocks[1].paragraphs[0])
    assert chapter_section.outline_blocks[1].lead == ""
    assert chapter_section.outline_blocks[1].source_chapter_refs == [1]
    assert 240.0 in chapter_section.outline_blocks[1].source_timestamps


def test_composition_boundary_review_avoids_action_template_titles_and_bodies():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVboundary",
            "title": "召回和排序边界复盘",
            "duration": 780,
            "one_liner": "把召回、阈值和排序特征之间的分工重新捋顺。",
            "chapters": [
                {
                    "index": 1,
                    "title": "召回目标",
                    "start": 0,
                    "end": 360,
                    "summary": "先定召回到底是补覆盖，还是给后面的排序提供候选。",
                    "pitfalls": ["不要为了追求覆盖率，把所有质量判断都压给召回集合。"],
                },
                {
                    "index": 2,
                    "title": "排序信号",
                    "start": 360,
                    "end": 780,
                    "summary": "再看阈值、特征和排序信号怎么接上，避免链路职责打架。",
                    "pitfalls": ["阈值和排序特征要一起看，不要把它们拆开单独调。"],
                },
            ],
            "study_questions": ["为什么召回目标和排序特征要一起看？"],
            "review_questions": ["什么时候不要再扩大召回集合？"],
            "final_synthesis": "这套召回和排序分工也可以迁到别的检索任务里。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    boundary_section = next(section for section in composed.body_sections if section.id == "boundary-review")
    titles = [block.title for block in boundary_section.outline_blocks]
    paragraphs = [
        paragraph
        for block in boundary_section.outline_blocks
        for paragraph in ([block.lead] if block.lead else []) + block.paragraphs
    ]

    assert boundary_section.summary != "最后集中看容易误判的地方、复盘问题，以及哪些经验值得迁移。"
    assert not boundary_section.summary.startswith("最后集中看")
    assert "容易误判的地方" not in titles
    assert "复盘时先回答这几个问题" not in titles
    assert "可以迁移的结论" not in titles
    assert any("召回" in title or "排序" in title or "阈值" in title for title in titles)
    assert all("容易误判的地方主要有这些" not in paragraph for paragraph in paragraphs)
    assert all("复盘时，至少应能回答这些问题" not in paragraph for paragraph in paragraphs)
    assert all("后续行动建议" not in paragraph for paragraph in paragraphs)


def test_composition_cleans_transfer_prefix_in_boundary_review():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVtransferprefix",
            "title": "检索链路复盘",
            "duration": 840,
            "one_liner": "把召回、阈值和排序链路重新对齐。",
            "chapters": [
                {
                    "index": 1,
                    "title": "召回目标",
                    "start": 0,
                    "end": 420,
                    "summary": "先定召回阶段到底给后面的排序提供什么候选。",
                    "pitfalls": ["不要为了追求覆盖率，把所有质量判断都压给召回集合。"],
                },
                {
                    "index": 2,
                    "title": "排序特征",
                    "start": 420,
                    "end": 840,
                    "summary": "再看阈值和排序特征怎么接上，避免职责打架。",
                },
            ],
            "final_synthesis": "后续行动建议是：用户可以根据视频中的实战演示，把这套召回和排序分工迁到新的检索任务里。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    boundary_section = next(section for section in composed.body_sections if section.id == "boundary-review")
    transfer_block = boundary_section.outline_blocks[-1]

    assert transfer_block.title != "后续行动建议是"
    assert any(token in transfer_block.title for token in ("检索任务", "实战演示", "召回", "排序"))
    assert transfer_block.paragraphs[0].startswith("用户可以根据视频中的实战演示")
    assert "后续行动建议是" not in transfer_block.paragraphs[0]
    assert "后续行动建议是" not in boundary_section.summary


def test_composition_keeps_transfer_sentence_without_invented_subject():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVtransferplain",
            "title": "查询改写复盘",
            "duration": 600,
            "one_liner": "把查询改写和排序联动重新讲清楚。",
            "chapters": [
                {
                    "index": 1,
                    "title": "查询改写",
                    "start": 0,
                    "end": 300,
                    "summary": "先看查询改写怎么给召回补上下文。",
                },
                {
                    "index": 2,
                    "title": "排序联动",
                    "start": 300,
                    "end": 600,
                    "summary": "再看排序特征怎么接上查询改写后的候选。",
                },
            ],
            "final_synthesis": "可以迁移到别的检索任务里，尤其是需要先扩召回再做排序的场景。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    boundary_section = next(section for section in composed.body_sections if section.id == "boundary-review")
    transfer_block = boundary_section.outline_blocks[-1]

    assert transfer_block.paragraphs[0].startswith("可以迁移到别的检索任务里")
    assert not transfer_block.paragraphs[0].startswith("这套思路")


def test_composition_merges_detail_block_when_it_only_restates_summary():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVrestate",
            "title": "召回和排序分工",
            "duration": 720,
            "one_liner": "把召回和排序的职责边界拆开。",
            "profile": {"primary_type": "procedural_explainer"},
            "chapters": [
                {
                    "index": 1,
                    "title": "召回和排序分工",
                    "start": 0,
                    "end": 720,
                    "summary": (
                        "这一节先把召回和排序的分工拆开。"
                        "重点强调召回只负责给后面的排序提供候选，不替排序直接做决定。"
                    ),
                    "learning_goal": "知道召回阶段和排序阶段各自负责什么。",
                    "key_takeaways": ["召回不要替排序做决定", "先定召回目标"],
                    "points": [{"text": "召回不要替排序做决定", "ts": 180, "quote": "召回不要替排序做决定"}],
                }
            ],
        }
    )

    composed = compose_from_lecture_json(lecture)

    chapter_section = next(section for section in composed.body_sections if section.id == "chapter-1")
    assert len(chapter_section.outline_blocks) == 1
    assert any("召回不要替排序做决定" in paragraph for paragraph in chapter_section.outline_blocks[0].paragraphs)


def test_composition_does_not_use_question_template_when_only_learning_goal_exists():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVgoalonly",
            "title": "上下文裁剪",
            "duration": 480,
            "one_liner": "解释上下文裁剪该按什么顺序做。",
            "chapters": [
                {
                    "index": 1,
                    "title": "上下文裁剪",
                    "start": 0,
                    "end": 480,
                    "summary": "",
                    "learning_goal": "知道什么时候先裁剪历史消息，什么时候先压缩工具输出。",
                }
            ],
        }
    )

    composed = compose_from_lecture_json(lecture)

    chapter_section = next(section for section in composed.body_sections if section.id == "chapter-1")
    assert all("这一节主要在回答：" not in paragraph for paragraph in chapter_section.paragraphs)
    assert any("知道什么时候先裁剪历史消息" in paragraph for paragraph in chapter_section.paragraphs)


def test_composition_uses_contextual_visual_caption_instead_of_generic_keyframe():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVcaption",
            "title": "界面流程讲解",
            "duration": 720,
            "one_liner": "解释一个 UI 驱动的工作流。",
            "chapters": [
                {
                    "index": 1,
                    "title": "工作流总览",
                    "start": 0,
                    "end": 360,
                    "summary": "这一章先建立整体流程。",
                    "frames": [{"ts": 30, "path": "frames/flow.jpg"}],
                    "key_takeaways": ["先把整条链路看完整。"],
                },
                {
                    "index": 2,
                    "title": "逐步执行",
                    "start": 360,
                    "end": 720,
                    "summary": "这一章再进入逐步执行。",
                    "frames": [{"ts": 420, "path": "frames/step.jpg"}],
                    "key_takeaways": ["每一步都要和上游状态对齐。"],
                },
            ],
            "study_questions": ["这条链路为什么不能跳步？"],
            "final_synthesis": "先建立整体流程，再逐步执行。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    captions = [
        visual.caption
        for section in composed.body_sections
        for visual in section.supporting_visuals
        if visual.caption
    ]
    assert captions
    assert all(caption != "关键画面" for caption in captions)
    assert any("工作流总览" in caption or "逐步执行" in caption for caption in captions)
