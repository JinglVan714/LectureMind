"""Offline smoke tests that exercise the parts that don't need network or
LLM API keys: BV parsing, schema validation, quote verification, renderer.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Use a temp data dir for tests so we never touch the real ./data
_tmp = tempfile.mkdtemp(prefix="lecturemind-test-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

from app.ingest.bilibili import BilibiliIngest  # noqa: E402
from app.ingest.subtitle import SubtitleSegment, _local_faster_whisper_base_snapshot  # noqa: E402
from app.render.renderer import Renderer, _global_visual_evidence  # noqa: E402
from app.understand.ir import LectureIR, hydrate_lecture_ir_data, infer_primary_type  # noqa: E402
from app.understand.ir_builder import lecture_ir_to_lecture_json  # noqa: E402
from app.understand.lecturize import LecturizeContext  # noqa: E402
from app.understand.schema import (  # noqa: E402
    Chapter,
    Frame,
    Highlight,
    KnowledgeUnitView,
    LectureJSON,
    Point,
)
from app.understand.vlm import FrameDescription  # noqa: E402


class TestWhisperModelDiscovery:
    def test_uses_newest_local_snapshot(self, tmp_path: Path):
        snapshots_dir = (
            tmp_path
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Systran--faster-whisper-base"
            / "snapshots"
        )
        older = snapshots_dir / "older"
        newer = snapshots_dir / "newer"
        older.mkdir(parents=True)
        newer.mkdir()
        os.utime(older, (1, 1))
        os.utime(newer, (2, 2))

        snapshot = _local_faster_whisper_base_snapshot(tmp_path)

        assert snapshot is not None
        assert Path(snapshot).name == "newer"

    def test_missing_local_snapshot(self, tmp_path: Path):
        assert _local_faster_whisper_base_snapshot(tmp_path) is None


# ---------------- BV parsing ----------------

class TestBVParsing:
    def test_full_url(self):
        assert BilibiliIngest.parse_bv("https://www.bilibili.com/video/BV1xx411c7mD") == "BV1xx411c7mD"

    def test_url_with_query(self):
        assert (
            BilibiliIngest.parse_bv("https://www.bilibili.com/video/BV1xx411c7mD?t=120")
            == "BV1xx411c7mD"
        )

    def test_raw_bv(self):
        assert BilibiliIngest.parse_bv("BV1xx411c7mD") == "BV1xx411c7mD"

    def test_invalid(self):
        with pytest.raises(ValueError):
            BilibiliIngest.parse_bv("https://example.com/not-bili")


# ---------------- Schema + quote verification ----------------

def _sample_lecture() -> LectureJSON:
    return LectureJSON(
        bv_id="BV1xx411c7mD",
        url="https://www.bilibili.com/video/BV1xx411c7mD",
        title="测试讲义",
        author="测试 UP",
        duration=600,
        cover_url="",
        one_liner="一个测试视频。",
        learning_path=["理解问题背景", "学习核心机制", "完成复习检查"],
        profile={"primary_type": "technical_formula", "density": "dense"},
        chapters=[
            Chapter(
                index=1,
                title="第一章",
                start=0,
                end=300,
                summary="这是一段对第一章的完整描述。",
                learning_goal="理解第一章解决的问题。",
                teaching_notes=[
                    "第一章先解释背景，并保留公式 $Q(s,a)$ 与化学式 \\ce{H2O}。",
                    "随后说明具体机制。\n```python\nprint('hello')\n```",
                ],
                process_steps=["先看输入", "再看处理过程"],
                points=[
                    Point(text="P1", ts=10, quote="这是字幕里的一段原话"),
                    Point(text="P2", ts=200, quote="这是另一段不存在的话语"),
                ],
                frames=[
                    Frame(
                        ts=50,
                        path="keyframes/BV1xx411c7mD/00050.jpg",
                        caption="板书",
                        ocr_text="板书文字",
                        insight="这张板书补充了核心结构。",
                    )
                ],
                pitfalls=["不要把示例当成通用结论。"],
                key_takeaways=["第一章的关键结论。"],
            ),
            Chapter(
                index=2, title="第二章", start=300, end=600,
                summary="第二章的内容。",
                points=[Point(text="P3", ts=400, quote="第二章的关键句")],
            ),
        ],
        final_synthesis="最终综合会把两章串成完整闭环，并保留 $E=mc^2$。",
        highlights=[Highlight(text="金句", ts=120)],
        glossary=[
            {"term": "测试机制", "ts": 10, "explanation": "用于阅读正文前理解核心机制。"},
        ],
        review_questions=["复习问题是什么？"],
        core_question="这个测试视频要解决什么？",
        mainline=["提出问题", "解释机制", "完成复盘"],
        timeline={
            "argument_path": ["提出问题", "解释机制", "完成复盘"],
            "turning_points": ["0s:\n进入\n「背景与痛点」"],
            "state_evolution": ["从输入\n到输出"],
            "chapter_boundaries": ["第一章建立输入理解", "第二章把前面的机制收束成输出"],
        },
        completeness={
            "mainline_closed": True,
            "visual_coverage": "已覆盖板书",
            "topic_reusability": True,
            "reader_can_understand_without_video": True,
            "notes": "测试完整性卡。",
        },
        render_plan={"formula_section": True, "code_section": True},
        knowledge_units=[
            {
                "id": "ku-1",
                "type": "mechanism",
                "title": "测试机制",
                "explanation": "解释测试机制。",
                "ts": 10,
                "chapter_index": 1,
            }
        ],
        visual_evidence=[
            Frame(
                ts=50,
                path="keyframes/BV1xx411c7mD/00050.jpg",
                caption="板书",
                ocr_text="板书文字",
                insight="全局关键图。",
                visual_type="diagram",
                selected_reason="帮助理解结构。",
                importance_score=0.9,
            )
        ],
        generation_mode="lecture_ir_v2",
    )


def _sample_ir() -> LectureIR:
    ctx = LecturizeContext(
        bv_id="BV1xx411c7mD",
        url="https://www.bilibili.com/video/BV1xx411c7mD",
        title="手写多头注意力机制",
        author="测试 UP",
        duration=600,
        cover_url="",
        segments=[
            SubtitleSegment(start=0, end=2, text="今天我们来讲注意力机制"),
            SubtitleSegment(start=50, end=55, text="这里是 Q K V 的维度变化"),
            SubtitleSegment(start=300, end=305, text="最后把多个头拼接起来"),
        ],
        frame_descs=[
            FrameDescription(
                timestamp=50,
                path=Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD" / "00050.jpg",
                caption="Q K V 维度变化公式",
                ocr_text="Q K V softmax mask",
                visual_type="formula",
                importance_score=0.92,
                ocr_density=0.4,
                novelty_score=0.8,
                why_useful="解释注意力计算的结构。",
            )
        ],
    )
    data = {
        "core_question": "如何理解多头注意力？",
        "mainline": ["定义 QKV", "计算注意力权重", "拼接多个头"],
        "chapters": [
            {
                "title": "注意力输入",
                "start": 0,
                "end": 300,
                "summary": "本章解释 QKV 的来源与作用。",
                "points": [{"text": "QKV 是注意力计算入口", "ts": 50, "quote": "这里是 Q K V 的维度变化"}],
            },
            {
                "title": "多头拼接",
                "start": 300,
                "end": 600,
                "summary": "本章解释多头结果如何拼接。",
                "points": [{"text": "多头结果需要拼接", "ts": 300, "quote": "最后把多个头拼接起来"}],
            },
        ],
    }
    hydrate_lecture_ir_data(data, ctx)
    return LectureIR.model_validate(data)


class TestSchema:
    def test_validates(self):
        lec = _sample_lecture()
        assert lec.bv_id == "BV1xx411c7mD"
        assert len(lec.chapters) == 2

    def test_quote_verification_finds_missing(self):
        lec = _sample_lecture()
        transcript = "这是字幕里的一段原话，第二章的关键句出现了。"
        errors = lec.validate_quotes(transcript)
        # P1 quote is in transcript, P2 is not, P3 is in
        assert len(errors) == 1
        assert "200s" in errors[0] or "200" in errors[0]

    def test_mark_unverified_adds_warning(self):
        lec = _sample_lecture()
        transcript = "这是字幕里的一段原话，第二章的关键句出现了。"
        marked = lec.mark_unverified_points(transcript)
        assert marked == 1
        assert lec.chapters[0].points[1].text.startswith("⚠️")
        # Re-running should not double-mark
        marked2 = lec.mark_unverified_points(transcript)
        assert marked2 == 1
        assert not lec.chapters[0].points[1].text.startswith("⚠️ ⚠️")

    def test_chapter_end_must_not_exceed_duration(self):
        with pytest.raises(Exception):
            LectureJSON(
                bv_id="BVx", title="t", author="a", duration=10, one_liner="x",
                chapters=[Chapter(index=1, title="c", start=0, end=99,
                                   summary="s", points=[])],
            )


class TestLectureIR:
    def test_infers_technical_profile(self):
        assert infer_primary_type("手写多头注意力机制") == "technical_formula"

    def test_ir_hydration_defaults(self):
        ir = _sample_ir()
        assert ir.profile.primary_type == "technical_formula"
        assert ir.render_plan.formula_section is True
        assert ir.completeness.reader_can_understand_without_video is True
        assert ir.visual_evidence[0].selected is True

    def test_ir_to_lecture_json(self):
        lecture = lecture_ir_to_lecture_json(_sample_ir())
        assert lecture.generation_mode == "lecture_ir_v2"
        assert lecture.profile.primary_type == "technical_formula"
        assert lecture.timeline.argument_path
        assert lecture.completeness.mainline_closed is True
        assert lecture.visual_evidence[0].visual_type == "formula"

    def test_one_liner_uses_semantic_sentence(self):
        ir = _sample_ir()
        ir.core_question = (
            "现代AI编程代理（如Codex）如何在底层通过“上下文工程”与“合法提示词注入”"
            "机制管理会话历史，并将技能、环境信息与系统指令动态组装发送给大模型？"
        )

        lecture = lecture_ir_to_lecture_json(ir)

        assert lecture.one_liner.startswith("本视频讲解现代AI编程代理")
        assert lecture.one_liner.endswith("。")
        assert "管理会话历史" in lecture.one_liner
        assert not lecture.one_liner.endswith("上下文工程”")


    def test_mainline_is_cleaned_before_rendering(self):
        ir = _sample_ir()
        ir.duration = 2400
        ir.mainline = [
            "介绍相关背景",
            "定义 QKV",
            "定义QKV",
            "讲解核心知识",
            "计算注意力权重",
            "拼接多个头",
            "说明输出投影",
            "讨论掩码作用",
            "迁移到交叉注意力",
            "对比自注意力边界",
            "比较推理开销",
        ]

        lecture = lecture_ir_to_lecture_json(ir)

        assert lecture.learning_path == [
            "定义 QKV",
            "计算注意力权重",
            "拼接多个头",
            "说明输出投影",
            "讨论掩码作用",
            "迁移到交叉注意力",
            "对比自注意力边界",
            "比较推理开销",
        ]
        assert lecture.mainline == lecture.learning_path

    def test_short_teaching_notes_fall_back_to_existing_chapter_fields(self):
        ir = _sample_ir()
        ir.chapters[0].teaching_notes = ["太短"]
        ir.chapters[0].learning_goal = "理解 QKV 为什么是注意力入口"
        ir.chapters[0].key_takeaways = ["QKV 决定后续权重计算的输入。"]

        lecture = lecture_ir_to_lecture_json(ir)

        note = lecture.chapters[0].teaching_notes[0]
        assert "理解 QKV 为什么是注意力入口" in note
        assert "太短" in note
        assert "本章解释 QKV 的来源与作用" in note
        assert "QKV 决定后续权重计算的输入" in note


# ---------------- Renderer ----------------

class TestRenderer:
    def test_render_writes_html(self):
        lec = _sample_lecture()
        frame_path = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD" / "00050.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake-jpeg")
        renderer = Renderer()
        css = renderer.load_inline_css()
        path = renderer.render_lecture(lec, css)
        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "测试讲义" in html
        assert "术语速查" in html
        assert "主线推进路径" in html
        assert "mainline-steps" in html
        assert "mainline-step" in html
        assert "<ol>\n        <article class=\"mainline-step\"" not in html
        assert "技术复盘" in html
        assert "全局关键图解" in html
        assert "timeline-detail-item" in html
        assert "timeline-time" in html
        assert "0s" in html
        assert "进入「背景与痛点」" in html
        assert "从输入到输出" in html
        assert "进入<br>" not in html
        assert "本章问题" in html
        assert "展开证据与引文" in html
        assert "最终综合" in html
        assert "复习问题" in html
        assert "学习闭环检查" in html
        assert "生成信息" in html
        assert "学习地图" not in html
        assert "递进关系" not in html
        assert "视觉证据" not in html
        assert "为什么重要" not in html
        assert "可验证论点" not in html
        assert "金句墙" not in html
        assert "完整性检查" not in html
        assert "技术/公式/代码重点" not in html
        assert "操作路线与检查点" not in html
        assert html.index("技术复盘") < html.index("术语速查") < html.index("最终综合")
        assert html.index("最终综合") < html.index("复习问题") < html.index("全局关键图解")
        assert html.count("lecture_ir_v2") == 1
        assert html.index("生成信息") < html.index("lecture_ir_v2")
        assert 'assets/katex/katex.min.css' in html
        assert 'assets/katex/katex.min.js' in html
        assert 'assets/katex/contrib/mhchem.min.js' in html
        assert 'assets/katex/contrib/auto-render.min.js' in html
        assert '<pre class="code-block"><code class="language-python">' in html
        assert "$Q(s,a)$" in html
        assert "\\ce{H2O}" in html
        assert "$E=mc^2$" in html
        assert "data:image/jpeg;base64," in html
        assert "https://www.bilibili.com/video/BV1xx411c7mD" in html
        # Anchor URL contains a t= timestamp
        assert "?t=10" in html  # P1 ts
        # CSS got inlined
        assert ":root" in html or "{" in html

    def test_sparse_procedure_recap_is_hidden(self):
        lec = _sample_lecture()
        lec.profile.primary_type = "procedural_tutorial"
        lec.knowledge_units = [
            KnowledgeUnitView(id="p1", type="procedure", title="第一步", explanation="先做第一步。", ts=10),
            KnowledgeUnitView(id="p2", type="procedure", title="第二步", explanation="再做第二步。", ts=20),
        ]
        frame_path = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD" / "00050.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake-jpeg")
        renderer = Renderer()
        css = renderer.load_inline_css()
        path = renderer.render_lecture(lec, css)
        html = path.read_text(encoding="utf-8")
        assert "操作路线与检查点" not in html
        assert "关键知识点复盘" not in html

    def test_global_visual_evidence_is_ranked_limited_and_time_sorted(self):
        lec = _sample_lecture()
        lec.visual_evidence = [
            Frame(ts=70, path="low.jpg", caption="人物画面", visual_type="person", importance_score=0.2),
            Frame(ts=50, path="formula.jpg", caption="公式", visual_type="formula", importance_score=0.95),
            Frame(ts=10, path="diagram.jpg", caption="结构图", visual_type="diagram", importance_score=0.9),
            Frame(ts=30, path="code.jpg", caption="代码", visual_type="code", importance_score=0.8),
            Frame(ts=20, path="slide.jpg", caption="文字页", visual_type="slide_text", importance_score=0.85),
            Frame(ts=40, path="table.jpg", caption="表格", visual_type="table", importance_score=0.75),
            Frame(ts=60, path="other.jpg", caption="其他", visual_type="other", importance_score=0.7),
        ]
        frames = _global_visual_evidence(lec)
        assert len(frames) == 5
        assert [f.ts for f in frames] == [10, 20, 30, 40, 50]
        assert "low.jpg" not in {f.path for f in frames}

    def test_render_index(self):
        renderer = Renderer()
        css = renderer.load_inline_css()
        html = renderer.render_index([], css)
        assert "LectureMind" in html
        assert "粘贴 B 站链接" in html
