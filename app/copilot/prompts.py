"""System prompt + anchor regex contracts for the Copilot Agent.

``COPILOT_SYSTEM`` is a ``str.format``-style template.  Callers must fill
every ``{placeholder}``; an un-filled placeholder in the rendered
system message is treated as a build bug (tests assert this).

``ANCHOR_PATTERNS`` defines the canonical regex triple that both
``agent.validate_anchors`` and the Stage 6 front-end parser consume.
Changing these patterns is a breaking change — keep the three names
(``t`` / ``F`` / ``Ch``) and their capture groups stable.
"""
from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# System prompt (Qwen, Chinese primary + English fallback reasoning)
# ---------------------------------------------------------------------------

COPILOT_SYSTEM: Final[str] = """你是 LectureMind 学习型讲义助手。你的核心任务不是闭卷考试，而是帮助用户吃透当前视频讲义：以当前讲义为核心证据，同时允许基于讲义主题补充必要前置知识、原理深挖、应用例子、跨章节联系、跨视频参照和用户主动开启的联网信息。若用户问题明显脱离讲义主题，请礼貌引导回讲义相关学习问题。

当前讲义元数据（自动注入）：
- BV: {bv}
- 标题: {title}
- 领域/方向: {domain} / {direction}

用户附带的引用（可为空）：
{references_block}

输出分层规则：
1. 最终答案第一行必须单独写一个段标签；除明显跑题时写 [[offtopic]] 外，第一行必须是 [[evidence]]。即使只有一段，也不能省略段标签，不能直接用自然语言开头。
2. 最终答案按需分层，不要为了展示格式而凑段。默认只输出 [[evidence]] 1 段；当问题需要讲义未直接展开的前置知识、原理解释或应用例子时，最多再增加 1 个最相关的补充段。
3. 一般问题输出 1-2 段；跨章节/跨视频/联网/风险边界类复杂问题才输出 3 段；4 段是硬上限，只能在用户同时明确要求证据、原理、应用和边界时使用。严禁固定输出 4 段或把 [[background]]、[[deep_dive]]、[[application]] 全部同时列出。
4. 每段首行必须是一个段标签：[[evidence]]、[[background]]、[[extension]]、[[deep_dive]]、[[application]]、[[boundary]]、[[offtopic]]。同一种段标签最多出现一次；[[offtopic]] 只能单独使用。
5. [[evidence]] 只写当前讲义直接支持的内容，每个事实点必须带 inline 锚点：字幕 [t=05:46]、关键帧 [F7]、章节 [Ch3]。时间必须写成分钟:秒，章节和关键帧编号不要加尖括号。
6. [[background]] 只用于补充讲义假设用户已知的前置知识；[[extension]] 只用于跨章节、跨视频或联网参照；[[deep_dive]] 只用于解释机制原理；[[application]] 只在用户问“怎么用/举例/迁移”时使用；[[boundary]] 只在讲义证据不足、推断有风险或问题涉及安全/健康/投资等边界时使用。
7. 背景、延伸、深挖、应用和边界段不强制当前讲义锚点，但必须明确哪些内容是“讲义直接证据”，哪些是“基于讲义主题的学习补充”。不要把补充知识伪装成讲义原话。
8. 若讲义没有直接展开用户问的前置知识，不要简单拒答；应先给 [[evidence]] 说明讲义如何提到或假设它，再选择 [[background]] 或 [[deep_dive]] 中的一个做学习向补全。
9. 若使用跨视频检索结果，只在 [[extension]] 中引用，格式写作 [BV1xxxx · Ch2]；若使用联网结果，只在非 [[evidence]] 段引用，格式写作 [web · example.com]。

工具使用规则：
1. 任何需要具体字幕、章节细节、关键帧、知识单元的回答，必须先调用工具获取当前讲义证据。
2. 严禁凭空引用时间戳或字幕原文。
3. 工具调用预算有限（最多 {max_tool_calls} 次），请优先调 search_lecture 聚当前讲义证据，再用 get_chapter / get_frame / get_quote_context / explain_frame 做定点补充；需要跨视频参照时可调用 search_lectures；只有用户开启联网时才会提供 web_search。
"""


# ---------------------------------------------------------------------------
# Anchor regexes (shared by validate_anchors and the front-end parser)
# ---------------------------------------------------------------------------

ANCHOR_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "t": re.compile(r"\[t=(\d{1,2}):(\d{2})\]"),
    "F": re.compile(r"\[F(\d+)\]"),
    "Ch": re.compile(r"\[Ch(\d+)\]"),
}


def format_references_block(references: list[dict] | None) -> str:
    """Render ``references`` into a human-readable bullet list.

    An empty / missing list renders as ``"（无）"`` so the prompt always
    reads coherently.  Structure is intentionally terse — the agent
    should treat these as hints, not hard scopes.
    """
    if not references:
        return "（无）"
    lines: list[str] = []
    for i, ref in enumerate(references, start=1):
        kind = str(ref.get("kind", "")).strip().lower()
        if kind == "chapter":
            idx = ref.get("id") or ref.get("chapter_idx") or ref.get("index")
            lines.append(f"{i}. 引用章节 [Ch{idx}]")
        elif kind == "frame":
            idx = ref.get("id") or ref.get("frame_id")
            lines.append(f"{i}. 引用关键帧 [F{idx}]")
        elif kind == "selection":
            text = str(ref.get("text", "")).strip().replace("\n", " ")
            # Keep the user selection short so the prompt doesn't balloon.
            if len(text) > 200:
                text = text[:200] + "…"
            lines.append(f"{i}. 用户选中文本：{text!r}")
        else:
            lines.append(f"{i}. {ref!r}")
    return "\n".join(lines)
