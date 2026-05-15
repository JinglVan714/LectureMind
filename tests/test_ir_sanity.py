"""Unit tests for the post-IR sanity validator.

Covers four classes of LLM regressions surfaced on real BV runs:

* ``process_steps`` items with leading list-numbering prefixes
  (``"1. "``, ``"Step1:"``, ``"①"``).
* ``mainline`` degraded into a chapter-title list
  (``["第 1 章 · …", "第 2 章 · …"]``).
* ``core_question`` that echoes a chapter title or the lecture title.
* Non-contiguous / non-monotonic chapter indices, with
  ``knowledge_units`` cross-references that have to follow the rewrite.
"""

from __future__ import annotations

from app.understand.ir_sanity import (
    sanitize_ir_data,
    strip_list_prefix,
)


# --------------------------------------------------------------------- #
# strip_list_prefix                                                       #
# --------------------------------------------------------------------- #


def test_strip_list_prefix_handles_common_llm_styles() -> None:
    cases = [
        ("1. 内容编写阶段：同时产出脚本", "内容编写阶段：同时产出脚本"),
        ("1）配置环境", "配置环境"),
        ("1) 配置环境", "配置环境"),
        ("1： Dispatch（分发）", "Dispatch（分发）"),
        ("1： Dispatch", "Dispatch"),
        ("Step1: Dispatch（分发）", "Dispatch（分发）"),
        ("step 2. Linear-1", "Linear-1"),
        ("阶段3、应用 SwiGLU", "应用 SwiGLU"),
        ("步骤4) 合并结果", "合并结果"),
        ("（1） 安装 Claude Code", "安装 Claude Code"),
        ("① 调用 X", "调用 X"),
        ("② 状态切换", "状态切换"),
    ]
    for raw, expected in cases:
        cleaned, stripped = strip_list_prefix(raw)
        assert stripped, f"expected to strip prefix from {raw!r}"
        assert cleaned == expected, f"{raw!r} -> {cleaned!r}"


def test_strip_list_prefix_leaves_real_content_alone() -> None:
    # No leading numbering — should pass through unchanged.
    cases = [
        "调用 X 传入 Y",
        "1.5x speed",  # decimal number, NOT a list prefix
        "v1.0 发布",
        "一、内容编写",  # Chinese ordinal — distinct convention, leave alone
        "",
    ]
    for raw in cases:
        cleaned, stripped = strip_list_prefix(raw)
        assert not stripped, f"unexpected strip on {raw!r}"
        assert cleaned == raw


# --------------------------------------------------------------------- #
# process_steps cleanup                                                   #
# --------------------------------------------------------------------- #


def test_sanitize_strips_process_steps_prefix() -> None:
    data = {
        "title": "示例",
        "chapters": [
            {
                "index": 1,
                "title": "Skill 工作流详解",
                "process_steps": [
                    "1. 内容编写阶段：同时产出脚本和 outline。",
                    "2. 第一个人工检查点：Agent 被强制停下。",
                    "Step1: Dispatch（分发）",
                    "无前缀的真实条目",
                ],
            }
        ],
    }
    sanity = sanitize_ir_data(data)
    assert sanity["process_steps_prefix_stripped"] == 3
    assert data["chapters"][0]["process_steps"] == [
        "内容编写阶段：同时产出脚本和 outline。",
        "第一个人工检查点：Agent 被强制停下。",
        "Dispatch（分发）",
        "无前缀的真实条目",
    ]


# --------------------------------------------------------------------- #
# mainline degradation                                                    #
# --------------------------------------------------------------------- #


def test_sanitize_clears_mainline_when_it_echoes_chapter_titles() -> None:
    data = {
        "title": "Harness 实践",
        "chapters": [
            {"index": 1, "title": "开场与问题引入"},
            {"index": 2, "title": "视频制作思路与可控性设计"},
            {"index": 3, "title": "实战演示"},
        ],
        "mainline": [
            "第 1 章 · 开场与问题引入",
            "第 2 章 · 视频制作思路与可控性设计",
            "第 3 章 · 实战演示",
        ],
        "core_question": "",
    }
    sanity = sanitize_ir_data(data)
    assert sanity["mainline_degraded"] is True
    assert data["mainline"] == []


def test_sanitize_keeps_real_mainline() -> None:
    data = {
        "title": "Harness 实践",
        "chapters": [
            {"index": 1, "title": "开场"},
            {"index": 2, "title": "演示"},
        ],
        "mainline": [
            "提出 Harness 想解决的核心问题：自动化讲义生成的可控性",
            "引入双 Agent 协作机制并对比单 Agent 的边界",
            "展示一段真实复现流程并指出三个常见踩坑",
        ],
    }
    sanity = sanitize_ir_data(data)
    assert sanity["mainline_degraded"] is False
    assert len(data["mainline"]) == 3


def test_sanitize_keeps_mainline_with_one_bad_item() -> None:
    # Only one item is degraded — keep the rest, since a real cognitive
    # path with one accidental restatement is more useful than nothing.
    data = {
        "title": "示例",
        "chapters": [
            {"index": 1, "title": "开场"},
            {"index": 2, "title": "演示"},
            {"index": 3, "title": "总结"},
            {"index": 4, "title": "尾声"},
        ],
        "mainline": [
            "第 1 章 · 开场",
            "引入 Y 概念",
            "演示 Z 流程",
            "对比 W 方法",
        ],
    }
    sanity = sanitize_ir_data(data)
    assert sanity["mainline_degraded"] is False
    assert len(data["mainline"]) == 4


# --------------------------------------------------------------------- #
# core_question                                                           #
# --------------------------------------------------------------------- #


def test_sanitize_clears_core_question_that_echoes_chapter_title() -> None:
    data = {
        "title": "Harness 实践",
        "chapters": [{"index": 1, "title": "开场与问题引入"}],
        "core_question": "第 1 章 · 开场与问题引入。",
    }
    sanity = sanitize_ir_data(data)
    assert sanity["core_question_dropped"] is True
    assert data["core_question"] == ""


def test_sanitize_clears_core_question_equal_to_lecture_title() -> None:
    data = {
        "title": "Harness 实践",
        "chapters": [{"index": 1, "title": "开场"}],
        "core_question": "Harness 实践",
    }
    sanity = sanitize_ir_data(data)
    assert sanity["core_question_dropped"] is True


def test_sanitize_keeps_real_core_question() -> None:
    data = {
        "title": "Harness 实践",
        "chapters": [{"index": 1, "title": "开场"}],
        "core_question": "如何让 Agent 全自动产出可控的讲解视频？",
    }
    sanity = sanitize_ir_data(data)
    assert sanity["core_question_dropped"] is False
    assert data["core_question"] == "如何让 Agent 全自动产出可控的讲解视频？"


# --------------------------------------------------------------------- #
# chapter index renumber                                                  #
# --------------------------------------------------------------------- #


def test_sanitize_renumbers_chapter_indices_starting_at_two() -> None:
    data = {
        "title": "示例",
        "chapters": [
            {"index": 2, "title": "A"},
            {"index": 3, "title": "B"},
            {"index": 4, "title": "C"},
        ],
        "knowledge_units": [
            {"id": "ku-1", "title": "u1", "chapter_index": 2},
            {"id": "ku-2", "title": "u2", "chapter_index": 4},
            {"id": "ku-3", "title": "u3", "chapter_index": 99},  # out of range
        ],
    }
    sanity = sanitize_ir_data(data)
    assert sanity["chapter_renumbered"] is True
    assert sanity["chapter_original_indices"] == [2, 3, 4]
    assert [ch["index"] for ch in data["chapters"]] == [1, 2, 3]
    # KU cross-refs follow the rewrite; OOR pinned to chapter 1.
    assert data["knowledge_units"][0]["chapter_index"] == 1
    assert data["knowledge_units"][1]["chapter_index"] == 3
    assert data["knowledge_units"][2]["chapter_index"] == 1


def test_sanitize_keeps_already_contiguous_chapters() -> None:
    data = {
        "title": "示例",
        "chapters": [
            {"index": 1, "title": "A"},
            {"index": 2, "title": "B"},
            {"index": 3, "title": "C"},
        ],
    }
    sanity = sanitize_ir_data(data)
    assert sanity["chapter_renumbered"] is False
    assert [ch["index"] for ch in data["chapters"]] == [1, 2, 3]


def test_sanitize_renumbers_code_block_cross_references() -> None:
    data = {
        "title": "示例",
        "chapters": [
            {
                "index": 2,
                "title": "A",
                "code_blocks": [{"code": "x", "chapter_index": 2}],
                "formula_blocks": [{"latex": "y", "chapter_index": 2}],
            },
            {
                "index": 5,
                "title": "B",
                "code_blocks": [{"code": "z", "chapter_index": 5}],
                "formula_blocks": [],
            },
        ],
    }
    sanitize_ir_data(data)
    assert [ch["index"] for ch in data["chapters"]] == [1, 2]
    assert data["chapters"][0]["code_blocks"][0]["chapter_index"] == 1
    assert data["chapters"][0]["formula_blocks"][0]["chapter_index"] == 1
    assert data["chapters"][1]["code_blocks"][0]["chapter_index"] == 2


# --------------------------------------------------------------------- #
# Combined / regression                                                   #
# --------------------------------------------------------------------- #


def test_sanitize_reproduces_bv1ypdgbcebee9b_scenario() -> None:
    """Mirrors the BV1ypdgBCE9B production regression captured in
    ``handoff_closeout.md``: mainline collapsed into chapter titles,
    core_question echoes the first chapter title, and one chapter has
    LLM-emitted "1. " prefixes in process_steps.
    """
    data = {
        "title": "Harness 实践：让 Agent 全自动制作知识讲解视频",
        "core_question": "第 1 章 · 开场与问题引入。",
        "mainline": [
            "第 1 章 · 开场与问题引入",
            "第 2 章 · 视频制作思路与可控性设计",
            "第 3 章 · 视频制作流程的四个关键步骤",
            "第 4 章 · Skill 工作流详解",
            "第 5 章 · 实战演示",
        ],
        "chapters": [
            {"index": 1, "title": "开场与问题引入"},
            {"index": 2, "title": "视频制作思路与可控性设计"},
            {"index": 3, "title": "视频制作流程的四个关键步骤"},
            {
                "index": 4,
                "title": "Skill 工作流详解：从文章到视频的四阶段与人工检查点",
                "process_steps": [
                    "1. 内容编写阶段：同时产出脚本和 outline。",
                    "2. 第一个人工检查点：Agent 被强制停下。",
                    "3. 开发阶段：outline 并非一次性完成。",
                ],
            },
            {"index": 5, "title": "实战演示"},
        ],
    }
    sanity = sanitize_ir_data(data)
    assert sanity["mainline_degraded"] is True
    assert data["mainline"] == []
    assert sanity["core_question_dropped"] is True
    assert data["core_question"] == ""
    assert sanity["process_steps_prefix_stripped"] == 3
    assert data["chapters"][3]["process_steps"][0].startswith("内容编写阶段")
    assert sanity["chapter_renumbered"] is False
