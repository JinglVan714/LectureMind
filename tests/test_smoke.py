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
from app.understand._latex_repair import repair_obj, repair_string  # noqa: E402
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


# ---------------- LaTeX escape repair ----------------

class TestLatexRepair:
    """Reverses ``json.loads`` silently eating LaTeX backslashes inside
    LLM-emitted JSON.  See ``app/understand/_latex_repair.py``."""

    def test_repairs_beta_tau_rightarrow_frac(self):
        # Each leading control char is the JSON escape for one of:
        #   \b → \beta  · \t → \tau / \text  · \r → \rightarrow / \rho
        #   \f → \frac  · etc.
        sample = (
            "$Total = \x08eta_1 A + \x08eta_2 E + \x08eta_3 P$"
            " · $r_{base}\x0dightarrow \x09ext{Correct}$"
            " · $\x0crac{a}{b}$"
        )
        out = repair_string(sample)
        assert "\\beta_1" in out
        assert "\\beta_2" in out
        assert "\\rightarrow" in out
        assert "\\text{Correct}" in out
        assert "\\frac{a}{b}" in out
        # Control chars must be gone
        for ch in ("\x08", "\x09", "\x0c", "\x0d"):
            assert ch not in out

    def test_keeps_real_newlines_intact(self):
        # Real ``\n`` paragraph separators must survive — over-correcting
        # would inject ``\nu`` into normal Chinese / English prose.
        sample = "第一段。\n第二段：beta vs eta\n第三段。"
        assert repair_string(sample) == sample

    def test_idempotent_on_clean_latex(self):
        clean = "$\\beta_1 + \\tau \\rightarrow \\text{Correct}$"
        assert repair_string(clean) == clean

    def test_walks_nested_dict_and_list(self):
        data = {
            "title": "Reward Modeling",
            "chapters": [
                {"summary": "$Total = \x08eta_1 A$"},
                {"points": [{"text": "门槛 $r_{base}\x0dightarrow 1$"}]},
            ],
        }
        out = repair_obj(data)
        assert out["chapters"][0]["summary"] == "$Total = \\beta_1 A$"
        assert out["chapters"][1]["points"][0]["text"] == "门槛 $r_{base}\\rightarrow 1$"

    def test_passes_through_non_strings(self):
        assert repair_obj(42) == 42
        assert repair_obj(None) is None
        assert repair_obj(True) is True
        assert repair_obj([1, 2.5, None]) == [1, 2.5, None]


# ---------------- v2: length-adaptive ceilings ----------------


class TestLengthAdapt:
    """Verify the duration-aware caps that replace the legacy
    fixed-everywhere ceilings (chapters: 3-8, mainline: 6-8, …)."""

    def test_keyframe_window_widens_for_long_videos(self):
        from app.understand.length_adapt import keyframe_window

        # 11 min: stays at the legacy floor
        kf_min_short, kf_max_short = keyframe_window(660, base_min=8, base_max=20)
        assert kf_min_short >= 8
        assert kf_max_short >= 20

        # 44 min: widens significantly so the long-video info-loss bug
        # described in the design doc no longer compresses 44 min into
        # the same envelope as 11 min.
        kf_min_long, kf_max_long = keyframe_window(2640, base_min=8, base_max=20)
        assert kf_max_long >= kf_max_short * 2
        assert kf_max_long <= 80  # hard ceiling to bound HTML size

    def test_chapters_target_grows_with_duration(self):
        from app.understand.length_adapt import chapters_target

        assert chapters_target(300) == (3, 4)        # 5 min
        assert chapters_target(1200) == (4, 6)       # 20 min
        ch_min, ch_max = chapters_target(2640)       # 44 min
        assert ch_min >= 5 and ch_max >= 7
        # Multi-hour videos still cap below an unreadable level.
        assert chapters_target(10800)[1] <= 12       # 3 h

    def test_mainline_max_is_continuous(self):
        from app.understand.length_adapt import mainline_max

        # No more 6→8 step at exactly 30 min; should grow smoothly.
        assert mainline_max(600) <= mainline_max(1800) <= mainline_max(3600)
        assert mainline_max(60) >= 4
        assert mainline_max(7200) <= 14

    def test_global_visual_limit_holds_at_5_for_short_videos(self):
        from app.understand.length_adapt import global_visual_limit

        # The renderer test suite assumes 5 frames for the 10 min fixture.
        assert global_visual_limit(600) == 5
        assert global_visual_limit(60) == 5
        # And expands for long videos.
        assert global_visual_limit(2640) > 5

    def test_length_budget_packs_all_caps(self):
        from app.understand.length_adapt import LengthBudget

        budget = LengthBudget.for_duration(2640, keyframe_base_min=8, keyframe_base_max=20)
        assert budget.duration_sec == 2640
        assert budget.chapters_min >= 5
        assert budget.mainline_max >= 8
        assert budget.keyframe_max >= 40
        assert budget.density_hint  # non-empty


# ---------------- v2: code/formula schema + render ----------------


class TestCodeFormulaSchema:
    """The schema must round-trip code_blocks / formula_blocks through
    Pydantic and the IR → LectureJSON conversion."""

    def test_chapter_carries_code_blocks(self):
        from app.understand.schema import Chapter, CodeBlock

        cb = CodeBlock(
            language="python",
            code="def f(x):\n    return x + 1",
            ts=10,
            chapter_index=1,
            source="ocr",
            explanation="The smallest possible function.",
        )
        ch = Chapter(
            index=1,
            title="第一章",
            start=0,
            end=120,
            summary="测试代码块",
            code_blocks=[cb],
        )
        assert ch.code_blocks[0].code.startswith("def f")
        assert "\n" in ch.code_blocks[0].code  # newlines survived

    def test_ir_to_lecture_passes_code_blocks(self):
        from app.understand.ir import IRCodeBlock, IRFormulaBlock

        ir = _sample_ir()
        ir.chapters[0].code_blocks = [
            IRCodeBlock(
                language="python",
                code="x = 1\ny = 2",
                ts=10,
                chapter_index=1,
                explanation="两行代码",
            )
        ]
        ir.chapters[0].formula_blocks = [
            IRFormulaBlock(
                latex="E = mc^2",
                ts=10,
                chapter_index=1,
                explanation="质能等价",
            )
        ]
        lecture = lecture_ir_to_lecture_json(ir)
        assert lecture.chapters[0].code_blocks
        assert lecture.chapters[0].code_blocks[0].language == "python"
        assert "\n" in lecture.chapters[0].code_blocks[0].code
        # Render-plan flags auto-flipped when content is present.
        assert lecture.render_plan.code_section is True
        assert lecture.render_plan.formula_section is True

    def test_renderer_emits_code_card_for_chapter_code_blocks(self):
        from app.understand.schema import CodeBlock

        lec = _sample_lecture()
        lec.chapters[0].code_blocks = [
            CodeBlock(
                language="python",
                code="def hello():\n    print('hi')",
                ts=10,
                chapter_index=1,
                explanation="hello world",
            )
        ]
        lec.render_plan.code_section = True
        frame_path = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD" / "00050.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake-jpeg")
        renderer = Renderer()
        css = renderer.load_inline_css()
        path = renderer.render_lecture(lec, css)
        html = path.read_text(encoding="utf-8")
        assert "代码片段" in html
        assert "code-card" in html
        assert "def hello()" in html
        assert 'class="language-python"' in html
        # Either a vendored bundle or the CDN fallback should be wired in.
        assert "highlight" in html.lower()


# ---------------- v2: study questions block ----------------


class TestStudyQuestionsRender:
    def test_template_renders_study_questions(self):
        lec = _sample_lecture()
        lec.study_questions = [
            "测试问题一？",
            "测试问题二？",
        ]
        frame_path = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD" / "00050.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake-jpeg")
        renderer = Renderer()
        css = renderer.load_inline_css()
        path = renderer.render_lecture(lec, css)
        html = path.read_text(encoding="utf-8")
        assert "读完后你应当能回答" in html
        assert "测试问题一？" in html
        assert "测试问题二？" in html


# ---------------- v2: fuzzy n-gram quote verification ----------------


class TestFuzzyQuoteVerification:
    """The legacy exact-substring matcher flagged any LLM paraphrase as
    ⚠️.  The new fuzzy matcher only flags actual hallucinations."""

    def _build(self, quote: str):
        from app.understand.schema import Chapter, LectureJSON, Point

        return LectureJSON(
            bv_id="BVfuzz",
            title="t",
            duration=600,
            one_liner="x",
            chapters=[
                Chapter(
                    index=1,
                    title="c",
                    start=0,
                    end=600,
                    summary="s",
                    points=[Point(text="P", ts=10, quote=quote)],
                )
            ],
        )

    def test_fuzzy_accepts_minor_paraphrase(self):
        # Transcript contains the canonical phrase; the LLM dropped one
        # filler particle ("我们") but kept enough characters that the
        # 4-gram overlap is well above 0.6.
        transcript = "我们今天来讲注意力机制的核心思想"
        lec = self._build("今天来讲注意力机制的核心思想")
        marked = lec.mark_unverified_points(transcript)
        assert marked == 0

    def test_fuzzy_rejects_unrelated_quote(self):
        transcript = "我们今天来讲注意力机制的核心思想"
        lec = self._build("猫吃了昨天剩下的鱼")
        marked = lec.mark_unverified_points(transcript)
        assert marked == 1
        assert lec.chapters[0].points[0].text.startswith("⚠️")

    def test_fuzzy_short_quote_still_uses_exact_match(self):
        # Quotes shorter than 6 normalised chars must match exactly so
        # we don't accept random substring noise.
        transcript = "abcdef"
        lec = self._build("xyz")
        assert lec.mark_unverified_points(transcript) == 1


# ---------------- v2: RAG chunk_lecture new kinds ----------------


class TestRagChunkV2:
    def test_chunk_lecture_emits_code_formula_and_study_question(self):
        from app.copilot.rag import chunk_lecture
        from app.understand.schema import CodeBlock, FormulaBlock

        lec = _sample_lecture()
        lec.chapters[0].code_blocks = [
            CodeBlock(
                language="python",
                code="x = 1\ny = 2",
                ts=10,
                chapter_index=1,
                explanation="两行代码",
            )
        ]
        lec.chapters[0].formula_blocks = [
            FormulaBlock(latex="E = mc^2", ts=10, chapter_index=1, explanation="经典公式")
        ]
        lec.study_questions = ["问题1？", "问题2？"]

        chunks = chunk_lecture(lec)
        kinds = {c.kind for c in chunks}
        assert "code_block" in kinds
        assert "formula_block" in kinds
        assert "study_question" in kinds
        # The code chunk preserves indentation/newlines for downstream RAG.
        code_chunks = [c for c in chunks if c.kind == "code_block"]
        assert len(code_chunks) == 1
        assert "x = 1\ny = 2" in code_chunks[0].text
        assert code_chunks[0].meta["language"] == "python"


# ---------------- v2: ts integrity regression (P2) ----------------


class TestTsIntegrity:
    """Regression coverage for the May-2026 ts hallucination bug class.

    The IR builder used to trust LLM-supplied ``ts`` values verbatim. In
    practice the LLM frequently:

      * extracted only the minute portion of the prompt's ``[MM:SS]`` prefix
        (so a frame at 12:51.0 became ``ts=12``),
      * left points/knowledge_units at ``ts=0`` even for chapters that started
        deep inside the video,
      * truncated chapter coverage to a round number well below ``duration``,
      * forgot to attach frames to chapters whenever any sibling chapter had
        even one LLM-provided frame.

    The tests below pin the post-processing fixes that turn those failures
    into deterministic, user-correct anchors.
    """

    def _ctx(self, *, duration: int = 800) -> LecturizeContext:
        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD"
        return LecturizeContext(
            bv_id="BV1xx411c7mD",
            url="https://www.bilibili.com/video/BV1xx411c7mD",
            title="ts integrity smoke",
            author="up",
            duration=duration,
            cover_url="",
            segments=[
                SubtitleSegment(start=10, end=15, text="开篇问题：什么是注意力机制"),
                SubtitleSegment(start=180, end=185, text="Plan Mode 是一种强制的协作模式"),
                SubtitleSegment(start=540, end=545, text="update_plan 通过提交完整快照实现 replan"),
                SubtitleSegment(start=720, end=725, text="收尾：和 Default Mode 的边界"),
            ],
            frame_descs=[
                FrameDescription(
                    timestamp=120.133,
                    path=str(kf_dir / "00120133.jpg"),
                    caption="Plan Mode 切换提示",
                    ocr_text="Plan Mode",
                    visual_type="slide_text",
                    importance_score=0.9,
                ),
                FrameDescription(
                    timestamp=540.5,
                    path=str(kf_dir / "00540500.jpg"),
                    caption="update_plan 参数表",
                    ocr_text="explanation, plan",
                    visual_type="slide_text",
                    importance_score=0.85,
                ),
                FrameDescription(
                    timestamp=771.033,
                    path=str(kf_dir / "00771033.jpg"),
                    caption="结尾对比表",
                    ocr_text="Plan Mode vs update_plan",
                    visual_type="diagram",
                    importance_score=0.92,
                ),
            ],
        )

    def test_visual_evidence_ts_uses_cache_truth_not_llm_ts(self):
        """LLM hallucinates ``ts=12`` for a frame at 771.033s; cache wins."""
        ctx = self._ctx()
        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD"
        data = {
            "chapters": [
                {"title": "ch1", "start": 0, "end": 200, "points": [], "frames": []},
                {"title": "ch2", "start": 200, "end": 800, "points": [], "frames": []},
            ],
            # The LLM "wrote" ts=12 for what is really a 771.033s frame.
            "visual_evidence": [
                {"ts": 12, "path": str(kf_dir / "00771033.jpg"), "selected": True},
                {"ts": 0, "path": str(kf_dir / "00120133.jpg"), "selected": True},
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        ts_by_path = {v["path"]: v["ts"] for v in data["visual_evidence"]}
        assert ts_by_path[str(kf_dir / "00771033.jpg")] == 771.033
        assert ts_by_path[str(kf_dir / "00120133.jpg")] == 120.133

    def test_attach_visuals_distributes_to_empty_chapters(self):
        """ch1 gets a LLM frame, ch2 is empty: ch2 must still receive frames."""
        ctx = self._ctx()
        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD"
        data = {
            "chapters": [
                {
                    "title": "ch1",
                    "start": 0,
                    "end": 200,
                    "points": [],
                    # LLM only attached one frame, to ch1.
                    "frames": [
                        {
                            "path": str(kf_dir / "00120133.jpg"),
                            "caption": "explicit",
                        }
                    ],
                },
                # ch2 is empty -- the bug used to leave it empty forever.
                {"title": "ch2", "start": 500, "end": 800, "points": [], "frames": []},
            ],
            "visual_evidence": [
                {
                    "ts": 0,
                    "path": str(kf_dir / "00540500.jpg"),
                    "selected": True,
                    "importance_score": 0.85,
                },
                {
                    "ts": 0,
                    "path": str(kf_dir / "00771033.jpg"),
                    "selected": True,
                    "importance_score": 0.92,
                },
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        ch1_frames = data["chapters"][0]["frames"]
        ch2_frames = data["chapters"][1]["frames"]
        # ch1's LLM-attached frame is preserved (1 entry, original path).
        assert len(ch1_frames) == 1
        assert ch1_frames[0]["path"].endswith("00120133.jpg")
        # ch2 must now contain at least one auto-distributed frame in its range.
        assert ch2_frames, "empty chapter should have been auto-populated"
        for fr in ch2_frames:
            assert 500 <= fr["ts"] <= 800

    def test_chapter_frame_ts_overridden_from_cache(self):
        """LLM gives a chapter frame ``ts=2`` for a real 120.133s frame."""
        ctx = self._ctx()
        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1xx411c7mD"
        data = {
            "chapters": [
                {
                    "title": "ch1",
                    "start": 100,
                    "end": 200,
                    "points": [],
                    "frames": [
                        {"ts": 2, "path": str(kf_dir / "00120133.jpg"), "caption": "frame"}
                    ],
                }
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        frame = data["chapters"][0]["frames"][0]
        assert frame["ts"] == 120.133, "cache ts must override LLM-provided ts"

    def test_point_ts_zero_in_nonzero_chapter_recovers(self):
        """Point with ``ts=0`` in chapter [120, 240] should snap to a real subtitle."""
        ctx = self._ctx()
        data = {
            "chapters": [
                {
                    "title": "Plan Mode",
                    "start": 120,
                    "end": 240,
                    "points": [
                        {
                            "text": "Plan Mode 是协作模式",
                            "ts": 0,
                            "quote": "Plan Mode 是一种强制的协作模式",
                        }
                    ],
                }
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        pt = data["chapters"][0]["points"][0]
        assert 120 <= pt["ts"] <= 240, f"ts={pt['ts']} fell outside chapter range"
        # The subtitle at 180s carries the canonical phrase, so we should land there.
        assert pt["ts"] == 180

    def test_chapter_range_stretched_to_duration(self):
        """LLM stops at ts=600 for an 800s video; last chapter is stretched out."""
        ctx = self._ctx(duration=800)
        data = {
            "chapters": [
                {"title": "ch1", "start": 0, "end": 200, "points": []},
                {"title": "ch2", "start": 200, "end": 400, "points": []},
                {"title": "ch3", "start": 400, "end": 600, "points": []},
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        last = data["chapters"][-1]
        # Coverage was 600/800 = 75% (< 90%), so the last chapter stretches to duration.
        assert last["end"] == 800, f"last chapter end={last['end']}, expected 800"

    def test_point_ts_match_outside_chapter_falls_back_to_midpoint(self):
        """When the only matching subtitle is in a different chapter, prefer the
        chapter's own midpoint over an out-of-range jump."""
        ctx = self._ctx()
        # Subtitles place "update_plan" at 540s but we file the point under
        # ch2 [120, 240]. The LLM mis-classified the chapter; we must keep
        # the user inside the declared chapter rather than jumping to ch3.
        data = {
            "chapters": [
                {
                    "title": "Plan Mode",
                    "start": 120,
                    "end": 240,
                    "points": [
                        {
                            "text": "update_plan 工具",
                            "ts": 0,
                            "quote": "update_plan 通过提交完整快照实现 replan",
                        }
                    ],
                }
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        pt = data["chapters"][0]["points"][0]
        assert 120 <= pt["ts"] <= 240, (
            f"ts={pt['ts']} jumped outside chapter; should fall back to midpoint"
        )
        # The fallback is the chapter midpoint: (120 + 240) / 2 = 180.
        assert pt["ts"] == 180

    def test_point_ts_near_boundary_is_repaired_into_chapter(self):
        ctx = self._ctx()
        data = {
            "chapters": [
                {
                    "title": "Plan Mode",
                    "start": 120,
                    "end": 240,
                    "points": [
                        {
                            "text": "边界漂移",
                            "ts": 116.5,
                            "quote": "开篇问题",
                        }
                    ],
                }
            ],
        }
        hydrate_lecture_ir_data(data, ctx)
        pt = data["chapters"][0]["points"][0]
        assert pt["ts"] == 120

    def test_v1_point_ts_near_boundary_is_repaired_into_chapter(self):
        from app.understand.lecturize import _hydrate_lecture_data

        ctx = self._ctx()
        data = {
            "chapters": [
                {
                    "title": "Plan Mode",
                    "start": 120,
                    "end": 240,
                    "points": [
                        {
                            "text": "边界漂移",
                            "ts": 244.5,
                            "quote": "收尾",
                        }
                    ],
                }
            ],
        }
        _hydrate_lecture_data(data, ctx)
        pt = data["chapters"][0]["points"][0]
        assert pt["ts"] == 180


# ---------------- v2: Reviser preservation safety net (P3b) ----------------


class TestReviserPreservation:
    """The Reviser system prompt forbids dropping high-value arrays, but
    deepseek-flash occasionally regresses anyway. The post-validation
    safety net :func:`_restore_dropped_content` backfills lost items
    from the previous IR so the user never sees a worse lecture after
    a critic round.
    """

    def _ir_with_code(self, *, code_blocks: list[dict] | None = None) -> LectureIR:
        from app.understand.ir import IRCodeBlock

        ir = _sample_ir()
        if code_blocks is None:
            code_blocks = [
                {
                    "language": "python",
                    "code": "def foo():\n    return 1",
                    "ts": 50,
                    "chapter_index": 1,
                    "source": "ocr",
                    "explanation": "demo",
                    "related_frame_paths": [],
                },
                {
                    "language": "python",
                    "code": "x = 2\nprint(x)",
                    "ts": 60,
                    "chapter_index": 1,
                    "source": "ocr",
                    "explanation": "demo2",
                    "related_frame_paths": [],
                },
            ]
        typed_blocks = [IRCodeBlock.model_validate(b) for b in code_blocks]
        chapters = list(ir.chapters)
        chapters[0] = chapters[0].model_copy(update={"code_blocks": typed_blocks})
        return ir.model_copy(update={"chapters": chapters})

    def test_restore_chapter_code_blocks_when_reviser_drops_them(self):
        from app.understand.ir_builder import _restore_dropped_content

        prev = self._ir_with_code()
        # Reviser returned the same IR but with chapter 1's code_blocks
        # silently emptied (a real failure mode observed in production).
        revised_chapters = list(prev.chapters)
        revised_chapters[0] = revised_chapters[0].model_copy(update={"code_blocks": []})
        revised = prev.model_copy(update={"chapters": revised_chapters})

        merged = _restore_dropped_content(prev, revised)
        assert len(merged.chapters[0].code_blocks) == 2
        # Verbatim restore: codes are byte-for-byte identical to prev.
        assert {b.code for b in merged.chapters[0].code_blocks} == {
            "def foo():\n    return 1",
            "x = 2\nprint(x)",
        }

    def test_restore_top_level_knowledge_units_when_reviser_drops_them(self):
        from app.understand.ir_builder import _restore_dropped_content

        prev = _sample_ir()
        # Force prev to have at least one knowledge unit if _sample_ir's
        # default produced none -- robust to factory changes.
        if not prev.knowledge_units:
            from app.understand.schema import KnowledgeUnit

            prev = prev.model_copy(
                update={
                    "knowledge_units": [
                        KnowledgeUnit(
                            id="ku-1",
                            type="concept",
                            title="测试",
                            explanation="x",
                            ts=10,
                            chapter_index=1,
                        )
                    ]
                }
            )
        revised = prev.model_copy(update={"knowledge_units": []})
        merged = _restore_dropped_content(prev, revised)
        assert len(merged.knowledge_units) == len(prev.knowledge_units)

    def test_restore_does_not_inflate_when_reviser_legitimately_grows(self):
        """If the reviser added entries, do not "restore" old (smaller) state."""
        from app.understand.ir_builder import _restore_dropped_content

        prev = self._ir_with_code(code_blocks=[
            {
                "language": "python",
                "code": "first()",
                "ts": 50,
                "chapter_index": 1,
                "source": "ocr",
                "explanation": "demo",
                "related_frame_paths": [],
            }
        ])
        # Reviser ADDED one block as instructed by the critic.
        revised_chapters = list(prev.chapters)
        new_blocks = list(revised_chapters[0].code_blocks) + [
            revised_chapters[0].code_blocks[0].model_copy(update={"code": "second()"})
        ]
        revised_chapters[0] = revised_chapters[0].model_copy(update={"code_blocks": new_blocks})
        revised = prev.model_copy(update={"chapters": revised_chapters})

        merged = _restore_dropped_content(prev, revised)
        # The growth is preserved; we did NOT downgrade to prev's single block.
        assert len(merged.chapters[0].code_blocks) == 2


# ---------------- v2: Reviser gate (LECTURE_REVISER_ENABLED) ----------------


class TestReviserGate:
    """``LECTURE_REVISER_ENABLED=false`` keeps the Critic but never invokes
    the Reviser. This is the recommended default because the full-IR
    rewrite is the slowest stage and is the primary cause of long-video
    timeouts. Short / medium / code videos can still opt in.
    """

    def _stub_critic(self, *, with_issues: bool):
        from app.understand.agents import CritiqueIssue, CritiqueResult

        if with_issues:
            issues = [
                CritiqueIssue(
                    kind="missing_code",
                    severity="medium",
                    location="chapters[0]",
                    evidence="代码片段未被抽取到 code_blocks",
                    suggestion="补 code_blocks",
                )
            ]
            verdict = "needs_revision"
        else:
            issues = []
            verdict = "ok"
        critique = CritiqueResult(verdict=verdict, summary="stub", issues=issues, usage={})
        revise_calls = {"count": 0}

        class _StubAgent:
            async def critique(self, ctx, ir_json, study_questions):
                return critique

            async def revise(self, ctx, ir_json, issues_list):
                revise_calls["count"] += 1
                # Pretend reviser added nothing actionable so the loop
                # exits cleanly even when reviser_enabled=True.
                return None, {}

        return _StubAgent(), revise_calls

    def _builder_with_stub(self, agent):
        from app.understand.ir_builder import LectureIRBuilder

        builder = LectureIRBuilder()
        builder._critic_agent = agent
        return builder

    def test_reviser_disabled_skips_revise_call(self):
        import asyncio

        agent, calls = self._stub_critic(with_issues=True)
        builder = self._builder_with_stub(agent)
        ir = _sample_ir()

        async def _run():
            return await builder._run_critic_loop(
                ctx=None, ir=ir, study_questions=["q?"], reviser_enabled=False
            )

        critique, latest_ir, rounds, usages, timing = asyncio.run(_run())

        assert calls["count"] == 0, "revise() must not be called when reviser_enabled=False"
        assert rounds == 0
        assert latest_ir is ir
        assert timing["reviser_sec"] == 0.0
        # Critic still ran exactly once and its verdict survived.
        assert critique.verdict == "needs_revision"
        assert len(critique.issues) == 1

    def test_reviser_enabled_calls_revise(self):
        import asyncio

        agent, calls = self._stub_critic(with_issues=True)
        builder = self._builder_with_stub(agent)
        ir = _sample_ir()

        async def _run():
            return await builder._run_critic_loop(
                ctx=None, ir=ir, study_questions=["q?"], reviser_enabled=True
            )

        asyncio.run(_run())
        assert calls["count"] == 1, "revise() should be called when reviser_enabled=True"

    def test_reviser_disabled_default_in_settings_schema(self):
        """``LECTURE_REVISER_ENABLED`` field default must be False.

        Tested at the model-field level (not by instantiating Settings)
        so the assertion is independent of any local .env override.
        """
        from app.config import Settings

        field = Settings.model_fields["lecture_reviser_enabled"]
        assert field.default is False


# ---------------- v2: Critic input projection (cost reduction) ----------------


class TestCriticInputProjection:
    """``CriticReviserAgent.critique`` projects subtitles/frames against
    the IR draft so long videos don't blow up the Critic prompt. The
    projection is purely deterministic so we can unit-test it without
    spinning up an LLM call.
    """

    def _ctx_long(self):
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.vlm import FrameDescription

        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1critic"
        # 60 evenly spaced 30s subtitle lines covering a 30-min video.
        segments = [
            Seg(start=i * 30.0, end=i * 30.0 + 28.0, text=f"字幕段 {i}：示例文本。" * 4)
            for i in range(60)
        ]
        # Mix of frame visual types so we can verify high-value bias.
        frames = []
        for i, vt in enumerate(
            [
                "code", "formula", "diagram",
                "slide_text", "slide_text", "slide_text",
                "table", "table",
                "ui", "ui",
                "person", "person", "person",
                "other", "other", "other",
                "other", "other", "other",
                "slide_text",
            ]
        ):
            frames.append(
                FrameDescription(
                    timestamp=float(i) * 90.0 + 5.0,
                    path=kf_dir / f"{i:05d}.jpg",
                    caption=f"frame-{i}",
                    ocr_text="",
                    visual_type=vt,
                    importance_score=0.4 + (i % 5) * 0.1,
                )
            )
        return LecturizeContext(
            bv_id="BV1critic000",
            url="https://www.bilibili.com/video/BV1critic000",
            title="critic projection long video",
            author="测试",
            duration=1800,  # 30 min
            cover_url="",
            segments=segments,
            frame_descs=frames,
        )

    def _ctx_short(self):
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.vlm import FrameDescription

        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1short"
        segments = [
            Seg(start=i * 10.0, end=i * 10.0 + 9.5, text=f"短字幕 {i}")
            for i in range(20)
        ]
        frames = [
            FrameDescription(
                timestamp=i * 30.0 + 5.0,
                path=kf_dir / f"{i:05d}.jpg",
                caption=f"frame-{i}",
                ocr_text="",
                visual_type="formula" if i % 3 == 0 else "other",
                importance_score=0.5,
            )
            for i in range(7)
        ]
        return LecturizeContext(
            bv_id="BV1short0000",
            url="https://www.bilibili.com/video/BV1short0000",
            title="critic projection short video",
            author="测试",
            duration=200,  # 3:20 min
            cover_url="",
            segments=segments,
            frame_descs=frames,
        )

    def _ir_with_chapter_window(self, *, start: float, end: float, ts: float) -> dict:
        return {
            "chapters": [
                {
                    "start": start,
                    "end": end,
                    "points": [{"ts": ts, "quote": ""}],
                    "code_blocks": [],
                    "formula_blocks": [],
                }
            ],
            "knowledge_units": [],
            "visual_evidence": [],
        }

    # ---- segments ----

    def test_critic_segments_keep_chapter_windows(self):
        from app.understand.agents import _critic_segments

        ctx = self._ctx_long()
        # One chapter covering 0-300s only — anything past 300s + margin
        # should be projected away.
        ir = self._ir_with_chapter_window(start=0.0, end=300.0, ts=120.0)
        kept = _critic_segments(ctx, ir, max_chars=0)
        assert kept, "projection must not return empty when chapter window exists"
        # First chapter window with 5% margin = ±15s, so segments up to
        # ~315s end. With a point ts=120s we also get ±30s anchor (90-150).
        # Anything that starts after 320s should be dropped.
        max_start = max(s.start for s in kept)
        assert max_start <= 320.0
        # And the chapter-window segments are present.
        assert any(0 <= s.start <= 30 for s in kept)
        assert any(280 <= s.start <= 300 for s in kept)

    def test_critic_segments_respect_max_chars(self):
        from app.understand.agents import _critic_segments

        ctx = self._ctx_long()
        # Single huge chapter window that would keep all 60 segments.
        ir = self._ir_with_chapter_window(start=0.0, end=1800.0, ts=900.0)
        unbounded = _critic_segments(ctx, ir, max_chars=0)
        assert len(unbounded) == 60
        # Tight budget forces uniform downsampling.
        capped = _critic_segments(ctx, ir, max_chars=400)
        assert 1 <= len(capped) < len(unbounded)
        # Result must remain time-sorted.
        starts = [s.start for s in capped]
        assert starts == sorted(starts)

    def test_critic_segments_unchanged_for_short_video(self):
        """Short videos already fit; projection should keep ~all lines."""
        from app.understand.agents import _critic_segments

        ctx = self._ctx_short()
        ir = self._ir_with_chapter_window(start=0.0, end=200.0, ts=80.0)
        kept = _critic_segments(ctx, ir, max_chars=24000)
        # 20 segments / single chapter window + small budget overhead =
        # everything stays.
        assert len(kept) == len(ctx.segments)

    # ---- frames ----

    def test_critic_frames_prefer_high_importance_and_code_formula(self):
        from app.understand.agents import _critic_frames

        ctx = self._ctx_long()
        # IR with only one anchor far from any specific frame; high-value
        # visual types should still be included regardless.
        ir = {
            "chapters": [{"points": [{"ts": 600.0}]}],
            "knowledge_units": [],
            "visual_evidence": [],
        }
        kept = _critic_frames(ctx, ir)
        kept_types = {f.visual_type for f in kept}
        # Long video target = 24, but we only have 20 frames so all will
        # be returned. Verify the high-value types are present.
        assert "code" in kept_types
        assert "formula" in kept_types
        assert "diagram" in kept_types
        # Result is time-sorted.
        ts_list = [f.timestamp for f in kept]
        assert ts_list == sorted(ts_list)

    def test_critic_frames_short_video_caps_at_16(self):
        """Short video cap is 16 even when more frames are available."""
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.vlm import FrameDescription
        from app.understand.agents import _critic_frames

        kf_dir = Path(os.environ["DATA_DIR"]) / "keyframes" / "BV1short16"
        frames = [
            FrameDescription(
                timestamp=float(i) * 5.0,
                path=kf_dir / f"{i:05d}.jpg",
                caption=f"f{i}",
                ocr_text="",
                visual_type="other",
                importance_score=0.5,
            )
            for i in range(30)
        ]
        ctx = LecturizeContext(
            bv_id="BV1short1600",
            url="x",
            title="t",
            author="a",
            duration=200,
            cover_url="",
            segments=[Seg(start=0, end=1, text="x")],
            frame_descs=frames,
        )
        ir = {"chapters": [{"start": 0, "end": 200}], "knowledge_units": [], "visual_evidence": []}
        kept = _critic_frames(ctx, ir)
        assert len(kept) == 16

    # ---- metrics surfacing ----

    def test_critique_to_dict_includes_metrics(self):
        from app.understand.ir_builder import _critique_to_dict
        from app.understand.agents import CritiqueResult

        c = CritiqueResult(
            verdict="ok",
            summary="x",
            issues=[],
            usage={"total_tokens": 1},
            metrics={"critic_prompt_chars": 1234, "critic_segments_kept": 5},
        )
        d = _critique_to_dict(c)
        assert d["metrics"] == {"critic_prompt_chars": 1234, "critic_segments_kept": 5}

    def test_critic_max_prompt_chars_default_is_24000(self):
        from app.config import Settings

        f = Settings.model_fields["lecture_critic_max_prompt_chars"]
        assert f.default == 24000

    def test_critic_segments_not_over_pruned_when_ir_is_large(self):
        """Regression: a code-rich short/medium video has a huge IR
        (>20k chars) but a moderate subtitle (<24k chars). Earlier the
        agent computed ``seg_budget = max_chars - len(ir_json) - ...``
        which collapsed to the floor and over-pruned subtitles, causing
        the Critic to invent ``dubious_quote`` issues for nearly every
        point. ``_critic_segments`` must treat ``max_chars`` as the
        subtitle excerpt cap, not a total-prompt cap.
        """
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.agents import _critic_segments

        # 200 segments, full subtitle ~6000 chars (well under 24k cap).
        segments = [
            Seg(start=i * 4.0, end=i * 4.0 + 3.5, text=f"片段{i:03d}描述")
            for i in range(200)
        ]
        ctx = LecturizeContext(
            bv_id="BV1bigir000",
            url="x",
            title="t",
            author="a",
            duration=900,  # 15 min
            cover_url="",
            segments=segments,
            frame_descs=[],
        )
        # Wide chapter covering the whole timeline, no anchors needed.
        ir = {
            "chapters": [{"start": 0, "end": 900, "points": []}],
            "knowledge_units": [],
            "visual_evidence": [],
        }
        kept = _critic_segments(ctx, ir, max_chars=24000)
        # Subtitles fit comfortably under the cap → projection must
        # keep all (or essentially all) segments.
        assert len(kept) >= int(0.95 * len(segments)), (
            f"projection over-pruned: kept {len(kept)}/{len(segments)}"
        )


# ---------------- v2: StudyQuestion multi-window sampling ----------------


class TestStudyQuestionSampling:
    """``StudyQuestionAgent`` now samples subtitles and frames from
    multiple time windows (head + mid + tail) instead of only the first
    five minutes. The sampling helpers are deterministic so we can
    unit-test them without an LLM call.
    """

    def test_short_video_falls_back_to_head_only_window(self):
        from app.understand.agents import _study_question_windows

        windows = _study_question_windows(360.0)  # 6 min
        assert len(windows) == 1
        assert windows[0].label == "开头"
        assert windows[0].start == 0.0

    def test_medium_video_yields_three_windows(self):
        from app.understand.agents import _study_question_windows

        windows = _study_question_windows(20 * 60.0)  # 20 min
        labels = [w.label for w in windows]
        assert labels == ["开头", "中段", "尾声"]
        assert windows[0].start == 0.0
        assert windows[-1].end <= 20 * 60.0

    def test_long_video_yields_five_windows(self):
        from app.understand.agents import _study_question_windows

        windows = _study_question_windows(75 * 60.0)  # 75 min
        labels = [w.label for w in windows]
        assert labels == ["开头", "中段-1", "中段-2", "中段-3", "尾声"]
        # Mid windows are spaced through the 25%/50%/75% marks.
        starts = [w.start for w in windows[1:4]]
        assert starts == sorted(starts)
        # No window extends past the video.
        for w in windows:
            assert w.end <= 75 * 60.0 + 0.001

    def test_multi_window_disabled_returns_only_head(self):
        from app.understand.agents import _study_question_windows

        windows = _study_question_windows(45 * 60.0, multi_window=False)
        assert len(windows) == 1
        assert windows[0].label == "开头"

    def test_format_subtitle_windows_groups_by_label(self):
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.agents import (
            _format_subtitle_windows,
            _study_question_windows,
        )

        segments = [
            Seg(start=10.0, end=14.0, text="开头第一句"),
            Seg(start=200.0, end=204.0, text="开头后期句"),
            # 20-min video, mid window centered around 600s
            Seg(start=590.0, end=594.0, text="中段重点句"),
            Seg(start=1130.0, end=1134.0, text="结尾收束句"),
        ]
        ctx = LecturizeContext(
            bv_id="BV1mid_test",
            url="x",
            title="t",
            author="a",
            duration=20 * 60,
            cover_url="",
            segments=segments,
            frame_descs=[],
        )
        windows = _study_question_windows(20 * 60.0)
        out = _format_subtitle_windows(ctx, windows)
        assert "[开头" in out
        assert "[中段" in out
        assert "[尾声" in out
        # Each line shows up under its window heading.
        assert "开头第一句" in out
        assert "中段重点句" in out
        assert "结尾收束句" in out

    def test_coverage_warning_when_questions_skip_second_half(self):
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.agents import _study_question_coverage_warnings

        # 30-min video; second half subtitle introduces the term
        # "梯度反传" — questions never mention it.
        segments = [
            Seg(start=i * 30.0, end=i * 30.0 + 28.0, text="前半段讲注意力机制示例")
            for i in range(30)
        ] + [
            Seg(start=900.0 + i * 30.0, end=900.0 + i * 30.0 + 28.0, text="后半段讲梯度反传")
            for i in range(30)
        ]
        ctx = LecturizeContext(
            bv_id="BV1cov_test1",
            url="x",
            title="t",
            author="a",
            duration=30 * 60,
            cover_url="",
            segments=segments,
            frame_descs=[],
        )
        questions = [
            "注意力机制如何工作？",
            "QKV 是什么？",
            "softmax 在注意力里怎么用？",
            "维度变化怎么发生？",
        ]
        warnings = _study_question_coverage_warnings(questions, ctx)
        assert warnings, "expected coverage warning when questions ignore second half"
        assert any("second_half_keyword_ratio_low" in w for w in warnings)

    def test_coverage_no_warning_when_second_half_covered(self):
        from app.ingest.subtitle import SubtitleSegment as Seg
        from app.understand.lecturize import LecturizeContext
        from app.understand.agents import _study_question_coverage_warnings

        segments = [
            Seg(start=i * 30.0, end=i * 30.0 + 28.0, text="前半段示例")
            for i in range(30)
        ] + [
            Seg(start=900.0 + i * 30.0, end=900.0 + i * 30.0 + 28.0, text="后半段讲梯度反传与正则化")
            for i in range(30)
        ]
        ctx = LecturizeContext(
            bv_id="BV1cov_test2",
            url="x",
            title="t",
            author="a",
            duration=30 * 60,
            cover_url="",
            segments=segments,
            frame_descs=[],
        )
        questions = [
            "注意力机制如何工作？",
            "梯度反传的具体步骤是什么？",
            "正则化在训练中如何使用？",
            "QKV 维度怎么变化？",
        ]
        warnings = _study_question_coverage_warnings(questions, ctx)
        assert warnings == []

    def test_user_template_includes_window_labels_when_filled(self):
        from app.understand.prompts import STUDY_QUESTIONS_USER_TEMPLATE

        rendered = STUDY_QUESTIONS_USER_TEMPLATE.format(
            bv_id="BV1xx",
            title="t",
            author="a",
            duration=1200,
            minutes=20,
            domain_hint="技术/编程类",
            q_min=5,
            q_max=8,
            subtitle_windows="[开头 00:00 - 04:00]\nx\n\n[中段 09:30 - 10:30]\ny",
            frames_windows="[ts=10s] z",
        )
        assert "[开头 00:00 - 04:00]" in rendered
        assert "[中段 09:30 - 10:30]" in rendered
        assert "覆盖度要求" in rendered

    def test_study_questions_result_default_warnings_and_metrics(self):
        from app.understand.agents import StudyQuestionsResult

        r = StudyQuestionsResult(questions=["a？"], usage={"x": 1})
        assert r.warnings == []
        assert r.metrics == {}

    def test_multi_window_default_is_true(self):
        from app.config import Settings

        f = Settings.model_fields["lecture_study_question_multi_window"]
        assert f.default is True
