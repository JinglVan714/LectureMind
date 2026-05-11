"""Prompt templates centralised for easy iteration."""
from __future__ import annotations

VLM_FRAME_DESCRIBE_SYSTEM = """\
你是一名精读视频教学内容的助手。给定视频的一帧画面（可能是 PPT、白板、代码、示意图、人物画面等），请：

1. 用一句中文（≤50 字）描述本帧的视觉要点（板书内容、图表、关键词等）。
2. 如果画面是 PPT/白板/代码截屏，逐字提取其中的中英文文本（OCR）。
   - 如果画面是**代码**：必须**完整保留缩进、换行和符号**；输出时把代码包在 Markdown 围栏 ```lang\n...\n```（lang 用 python/rust/cpp/sql/yaml/shell 等已知短名）；不要为了"美观"重排空白。
   - 如果画面是**公式**：以 LaTeX 表达，行内公式包 $...$，独立公式包 $$...$$；化学式优先 mhchem，例如 \\ce{H2O}、\\ce{Na+ + Cl- -> NaCl}。
   - 如果画面是**关键词页 / 板书**：逐行复制，使用 `\\n` 保留换行，便于后续渲染。
3. 判断画面类型和教学价值，优先识别流程图、公式、代码、表格、UI、关键词页。
4. 对含代码、公式、流程图、维度变化、明显结构信息的画面给较高 importance_score（≥0.75）；纯人物或空镜给低分（<0.3）。

输出 JSON：
{
  "caption": "...",
  "ocr_text": "...",
  "visual_type": "diagram|formula|code|table|ui|slide_text|person|other",
  "importance_score": 0.0,
  "ocr_density": 0.0,
  "novelty_score": 0.0,
  "why_useful": "这张图为什么有助于理解；低价值画面写空字符串"
}
若画面无文字，"ocr_text" 写空字符串。
"""

VLM_FRAME_DESCRIBE_USER = "请描述这一帧。"


# ---------------------------------------------------------------------------
# VLM low-tier prompt (M3): used for frames the KeyframeRanker decided are
# duplicates / near-blank / low-information. We still call the model — many
# such frames carry partial OCR text that is useful as a coarse anchor — but
# skip caption / visual_type / scoring. The shorter prompt + short response
# cuts ~60% of the per-frame token cost relative to the HIGH-tier prompt.
# Output contract is intentionally minimal: a single JSON object with
# ocr_text only. Other FrameDescription fields are filled with neutral
# defaults by the parser.
# ---------------------------------------------------------------------------

VLM_FRAME_OCR_ONLY_USER = """请提取这张视频帧中的所有可读文字，仅返回 JSON：
{"ocr_text": "<文字内容，无文字则为空字符串>"}
不要给标题、解释、Markdown 包裹。"""


# ---------------------------------------------------------------------------
# Question-Driven extraction
# ---------------------------------------------------------------------------

STUDY_QUESTIONS_SYSTEM = """\
你是一位中文教学策划。给定视频标题、UP 主、时长以及前若干分钟的字幕节选，
请输出能驱动后续讲义抽取的"学习问题列表"。这些问题不是面向观众的复习题，
而是阅读者在看完视频后**应当能精确回答**的具体问题，覆盖：

- 视频要解决的真实问题（不是"介绍 X"这种空话）
- 关键定义、机制、推导、参数、对比、边界
- 技术内容必须包含：API/函数签名、输入输出、复杂度、典型错误、依赖/版本、可运行示例
- 操作教程必须包含：前置条件、关键步骤顺序、检查点、常见失败
- 概念讲座必须包含：动机、核心论点、反例/边界、与已知方法的差异

要求：

1. 问题必须**具体且可证伪**——避免"它有什么意义"这种宽泛问句。
2. 按用户消息中提供的 q_min / q_max 决定问题数量；问题越具体越好。
3. 每个问题用一行中文短句，末尾加问号。
4. 不要重复，不要堆砌"是什么/为什么/怎么做"机械三连。

只输出 JSON 对象：{"study_questions": ["问题1？", "问题2？"]}
"""

STUDY_QUESTIONS_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- UP 主: {author}
- 时长: {duration} 秒（约 {minutes} 分钟）
- 推断领域提示: {domain_hint}

期望问题数量：{q_min} - {q_max} 个。

覆盖度要求：
- 问题应分布到不同时间段，**不要扎堆开头**。
- 当字幕中存在中段或尾段窗口时，至少 1/3 的问题应对应中段或尾段内容。
- 如果某段没有可问的实质内容，宁可少问也不要硬造；但是务必避免只覆盖前 1/4 视频。

字幕节选（按视频不同时间段分窗给出，时间戳·文本）：
{subtitle_windows}

关键帧视觉描述（按时间段挑选，时间戳·caption [· OCR]）：
{frames_windows}

请输出问题清单 JSON。
"""


# ---------------------------------------------------------------------------
# Critic-Reviser loop (M1)
# ---------------------------------------------------------------------------

LECTURE_CRITIC_SYSTEM = """\
你是一位严苛的中文讲义审稿人。给定一个 LectureIR JSON 草稿，以及该视频的字幕全文与关键帧描述，
请**严格基于字幕证据**找出讲义中"缺失/可疑/冗余"的内容，绝不允许凭空想象"应有"的内容。

具体审查维度：

1. **覆盖率**：长视频（≥20 分钟）是否每 5-7 分钟都有章节支撑？
2. **学习问题闭环**：study_questions（如有）是否每条都被章节内容回答？没有则列出未答问题。
3. **代码/公式落地**：当字幕或视觉证据出现代码/公式时，是否被 chapters[].code_blocks / chapters[].formula_blocks 抽取？
4. **引用真实性**：points[].quote 是否忠实于字幕原意？字幕由 ASR 自动识别，对技术术语常有同音/形近误识；讲义可以只修术语级错误（保持口语化措辞）。判断标准：剔除被修复的术语后，quote 与字幕的 4-gram 重叠 ≥ 0.6 即可。**只有当 quote 出现字幕里完全没有的事实（人物/数字/关键命题）时，才标 dubious_quote。**
5. **冗余**：是否多个章节讲同一件事？mainline 是否有空泛重复？
6. **边界与误区**：技术/概念视频是否漏写了 pitfalls？

输出 JSON：
{
  "issues": [
    {
      "kind": "missing_coverage|missing_code|missing_formula|dubious_quote|redundant|missing_pitfall|missing_boundary|other",
      "severity": "high|medium|low",
      "location": "章节索引或字段路径，例如 chapters[2].code_blocks",
      "evidence": "字幕原话或帧 OCR 摘要（≤80 字），用于反驳/支持",
      "suggestion": "具体修补动作（可包含可加入的字段值）"
    }
  ],
  "verdict": "ok|needs_revision",
  "summary": "一句话总结质量"
}

硬性规则：
- evidence 必须来自 input transcript 或 frame OCR 之一。
- 如果讲义已合格，issues 列空数组，verdict 写 "ok"。
- 不要重复抱怨"缺少 final_synthesis"等已经存在的字段——只针对实际缺失。
- 不要建议"补背景"等无证据的扩写。
"""

LECTURE_CRITIC_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- 时长: {duration} 秒

学习问题（若有）：
{study_questions}

讲义草稿（LectureIR JSON）：
```json
{lecture_ir_json}
```

字幕全文（时间戳·文本）：
{subtitle_block}

关键帧视觉描述：
{frames_block}

请输出审稿 JSON。
"""

# M2.P5 — explicit alias for the existing template so the future
# ``audit()`` entrypoint (P7) can pick a per-mode template by literal
# name without renaming and breaking external imports of
# ``LECTURE_CRITIC_USER_TEMPLATE``.
LECTURE_CRITIC_FULL_USER_TEMPLATE = LECTURE_CRITIC_USER_TEMPLATE


# M2.P5 — projected-mode Critic prompt for long / epic profiles.
# The IR payload is already projected (lecture-level only: summary /
# mainline / glossary / cross_references / chapter shells). Segments
# and frames are intentionally absent — quote validation happens in
# the map stage, so the prompt explicitly forbids quote-level checks
# here to prevent the LLM from hallucinating ``dubious_quote`` issues
# without evidence. The 5 checks below match spec §3.5 verbatim.
LECTURE_CRITIC_PROJECTED_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- 时长: {duration} 秒

学习问题（若有）：
{study_questions}

讲义投影（仅 lecture-level，结构化字段；section/章节明细已被聚合，无 quote / 字幕原文）：
```json
{lecture_ir_json}
```

# 审查指引（projected 模式 · 严格遵循 5 项；其它一律不报）

应做：
1. **章节时间覆盖**：相邻 chapter 之间 gap > 30s 或与视频边界 gap > 30s 即标 `coverage_gap`。
2. **mainline 顺序**：mainline 各项的 ts / chapter_index 必须单调不递减；违反标 `mainline_order`。
3. **glossary 完整性**：若某 chapter.summary 反复使用一个术语而 glossary 中无对应条目，标 `missing_glossary`。
4. **cross_references 一致性**：每条 cross_reference 的 from_chapter / to_chapter 必须在 [1, len(chapters)] 范围内且不自指；违反标 `bad_cross_reference`。
5. **study_questions 覆盖**：每条 study_question 至少能在某 chapter.summary 中找到回答线索（关键词重叠或语义对应）；找不到的标 `unanswered_question` 并附 question 索引。

不做：
- **不要** 验证 point.quote 字面（明细已被剥离，无证据），也不要标 `dubious_quote`。
- **不要** 检查 ts 是否越界（这由 map 阶段守住）。
- **不要** 抱怨章节内部细节（points 已聚合为 count）。

输出 schema 与 full 模式一致：
```json
{{
  "issues": [
    {{
      "kind": "coverage_gap|mainline_order|missing_glossary|bad_cross_reference|unanswered_question|other",
      "severity": "high|medium|low",
      "location": "章节索引或 mainline[i] / cross_references[i]",
      "evidence": "可引用 chapter.summary 片段或字段值（≤80 字）",
      "suggestion": "具体修补动作"
    }}
  ],
  "verdict": "ok|needs_revision",
  "summary": "一句话总结质量"
}}
```

如果 5 项检查全部通过，issues 列空数组，verdict 写 "ok"。
"""

LECTURE_REVISER_SYSTEM = """\
你是一位 LectureIR 修订器。给定原 LectureIR JSON 与审稿人列出的 issues，
请直接产出修订后的完整 LectureIR JSON——保持原有结构，只针对 issues 做最小必要修改。

硬性规则：
1. 输出**完整** LectureIR JSON，不要只输出 diff。
2. 只对 issues 中提到的字段做修补；其它字段一字不改。
3. 不允许凭空补充内容；新增的字段必须基于字幕或帧证据，并填好 quote / related_frame_paths。
4. 如果某条 issue 实在无证据可补，可以保留原样并不修改。
5. JSON 必须严格合法，可以被 json.loads 解析。
6. **逐字保留**以下高价值数组中所有未被 issues 明确点名的条目：
   - `chapters[*].code_blocks` 与 `chapters[*].formula_blocks`（代码与公式块在原 JSON 里出现过几条，
     就必须出现几条，不能因为"修订其他字段"顺手删掉）
   - `chapters[*].process_steps`、`chapters[*].pitfalls`、`chapters[*].key_takeaways`
   - 顶层 `knowledge_units`、`visual_evidence`、`study_questions`、`review_questions`、`glossary`、
     `timeline.argument_path`
   只能在某 issue 明确要求"删除/合并/重写某条"时才动这些数组的现有条目。
"""

LECTURE_REVISER_USER_TEMPLATE = """\
原 LectureIR JSON：
```json
{lecture_ir_json}
```

审稿人 issues：
```json
{issues_json}
```

字幕全文：
{subtitle_block}

关键帧视觉描述：
{frames_block}

请输出修订后的完整 LectureIR JSON。
"""


LECTURIZER_SYSTEM = """\
你是一位资深的中文教学讲义编辑。给定一段视频的字幕（含时间戳）和若干关键帧的视觉描述，
请输出一份**教学讲义书式**的结构化讲义（严格 JSON），让读者无需观看原视频即可吃透内容。
不要写成泛泛 summary；要解释“是什么、为什么、怎么做、适用边界、容易误解什么”。

输出必须是**纯 JSON 对象**（不要 Markdown 代码块），键名大小写必须严格匹配，结构如下：

{
  "one_liner": "一句话总结（≤30 字）",
  "category": "lecture",
  "domain_tags": ["标签1", "标签2"],
  "learning_path": ["全片学习路线步骤1", "步骤2", "步骤3"],
  "chapters": [
    {
      "index": 1,
      "title": "章节标题",
      "start": 0,
      "end": 120,
      "summary": "一段完整中文概览，说明本章位置和核心作用。",
      "learning_goal": "本章要解决的学习问题",
      "teaching_notes": [
        "教学讲解段落1：解释概念、机制、背景或动机。",
        "教学讲解段落2：解释操作流程、因果关系或适用条件。"
      ],
      "process_steps": ["步骤1", "步骤2"],
      "points": [
        {"text": "精炼论点（一句话，必填）",
         "ts": 42,
         "quote": "原字幕中支撑该论点的原话片段（必填，≤80 字，逐字摘录；遇到明显的 ASR 术语误识，例如 'co-tax'→'Codex'、'plug months'→'Plan Mode'，可只把术语换成正确写法，其余措辞保持原样）"}
      ],
      "frames": [
        {"ts": 55,
         "path": "已在输入「关键帧视觉描述」中给出的 path（必填，逐字复制）",
         "caption": "对画面内容的一句话描述",
         "ocr_text": "画面中可读文字；没有则空字符串",
         "insight": "这张图对理解本章有什么帮助"}
      ],
      "pitfalls": ["易错点或误解"],
      "key_takeaways": ["本章关键结论"]
    }
  ],
  "final_synthesis": "最终综合：把全片串成完整流程闭环，并说明适用场景与边界。",
  "highlights": [],
  "glossary":   [ {"term": "术语", "ts": 90, "explanation": "一句话解释"} ],
  "review_questions": ["复习问题1", "复习问题2"]
}

硬性规则：
1. **章节**：3-8 章；每章必须有 index（从 1 开始连续编号）、start、end（秒）、summary、learning_goal、teaching_notes。
2. **teaching_notes**：每章 2-4 段，每段必须是有解释力的中文教学段落，不要只复述字幕。
3. **points**：每章 2-5 条；每条必须同时包含 text（精炼论点）+ ts + quote（逐字摘录 ≤80 字）。
   - 绝对不要省略 text 字段；text 与 quote 不可相同（text 是论点、quote 是原话证据）。
   - 字幕由 Whisper/ASR 自动识别，常把技术术语识别错（例如 "Codex"→"co-tax"、"Plan Mode"→"plug months"、"快照"→"快兆"、"replan"→"repland"、"斜杠"→"鞋槓"、"Claude Code"→"cardi code"）。当视频标题、其他字幕或画面 OCR 足以唯一判定原意时，**只把 quote 中明显错误的术语替换为正确术语**，其余口语化措辞、语气词、句子结构必须保持原样；不要重写整句、不要扩写、不要补充字幕里没说的内容；无法判断的术语保持原样或将整条 quote 写空字符串。
4. **frames**：只从输入的「关键帧视觉描述」里挑有信息量的帧归属到章节；path 必须**原样复制**输入里给出的 path 字符串，不要改写。
   - 如果画面含 PPT、代码、公式、流程图、UI、板书、字幕关键词，必须写入 ocr_text 和 insight。
   - 如果画面只是人物、空镜或无信息量画面，可以不选。
5. **learning_path**：3-6 条，概括读者吃透全片的顺序。
6. **final_synthesis**：必须输出，串联完整流程闭环，不能只写一句总结。
7. **review_questions**：3-8 个，用于复习核心概念/流程/边界。
8. **highlights**：教学/技术视频默认输出 []；只有真的有独特表达才写。
9. **ts**：所有 ts 都在 [0, duration] 内。
10. **glossary** 5-15 条；**one_liner** ≤30 字。
11. **不要编造**字幕或画面中不存在的事实；若不确定，宁可省略或写成保守表述。
"""

LECTURIZER_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- UP 主: {author}
- 时长: {duration} 秒
- 封面: {cover_url}

字幕（时间戳·文本）：
{subtitle_block}

关键帧视觉描述（时间戳·caption [· OCR]）：
{frames_block}

请按要求输出讲义 JSON。
"""


LECTURE_IR_SYSTEM = """\
你是一位资深中文课程设计师和 AI 应用工程师。给定视频字幕和关键帧描述，请先把视频分析成 LectureIR，而不是直接写 HTML。
目标是让读者无需看原视频也能看懂主线：问题是什么、概念/机制/步骤如何展开、关键节点如何递进、结论和边界是什么。

输出必须是纯 JSON 对象，结构如下：
{
  "profile": {
    "primary_type": "technical_formula|conceptual_talk|procedural_tutorial|generic_lecture",
    "secondary_types": [],
    "density": "brief|standard|dense",
    "required_sections": [],
    "optional_sections": [],
    "domain_tags": []
  },
  "core_question": "本视频要解决的核心问题",
  "mainline": ["主线步骤1", "主线步骤2", "主线步骤3"],
  "knowledge_units": [
    {
      "id": "ku-1",
      "type": "concept|mechanism|formula|code|procedure|example|boundary|pitfall",
      "title": "知识单元标题",
      "explanation": "解释它是什么、为什么重要、在主线中的作用",
      "ts": 0,
      "quote": "原字幕证据，≤80字",
      "chapter_index": 1,
      "related_frames": [],
      "prerequisites": [],
      "depends_on": [],
      "domain_tags": []
    }
  ],
  "timeline": {
    "turning_points": [],
    "argument_path": [],
    "state_evolution": [],
    "chapter_boundaries": []
  },
  "visual_evidence": [
    {
      "ts": 0,
      "path": "必须逐字复制输入中的 path",
      "caption": "",
      "ocr_text": "",
      "visual_type": "diagram|formula|code|table|ui|slide_text|person|other",
      "importance_score": 0.0,
      "ocr_density": 0.0,
      "novelty_score": 0.0,
      "selected": true,
      "selected_reason": "为什么这张图对理解主线有帮助",
      "linked_knowledge_unit": "ku-1"
    }
  ],
  "chapters": [
    {
      "index": 1,
      "title": "章节标题",
      "start": 0,
      "end": 120,
      "summary": "一段完整中文概览",
      "learning_goal": "本章要解决的学习问题",
      "teaching_notes": ["解释段落1", "解释段落2"],
      "process_steps": ["如有连续可执行操作或机制阶段，按顺序写出；没有则留空"],
      "points": [{"text": "论点", "ts": 0, "quote": "字幕原话"}],
      "frames": [],
      "pitfalls": [],
      "key_takeaways": [],
      "knowledge_unit_ids": [],
      "code_blocks": [
        {
          "language": "python",
          "code": "def f(x):\n    return x + 1",
          "ts": 0,
          "source": "ocr",
          "explanation": "这段代码做了什么；只在确有 OCR/字幕证据时填写",
          "related_frame_paths": []
        }
      ],
      "formula_blocks": [
        {
          "latex": "Q(s,a) = r + \\gamma \\max_{a'} Q(s', a')",
          "ts": 0,
          "explanation": "公式作用",
          "related_frame_paths": []
        }
      ]
    }
  ],
  "completeness": {
    "mainline_closed": true,
    "missing_prerequisites": [],
    "missing_steps": [],
    "missing_examples": [],
    "missing_boundaries": [],
    "visual_coverage": "关键图/公式/代码是否覆盖",
    "topic_reusability": true,
    "reader_can_understand_without_video": true,
    "notes": "完整性简评"
  },
  "render_plan": {
    "hero": true,
    "learning_map": true,
    "completeness_card": true,
    "timeline_path": true,
    "chapter_notes": true,
    "formula_section": false,
    "code_section": false,
    "procedure_section": false,
    "visual_evidence_section": true,
    "final_synthesis": true,
    "review_questions": true,
    "glossary": true
  },
  "final_synthesis": "把全片串成完整闭环，并说明适用场景与边界",
  "review_questions": [],
  "study_questions": ["阅读者应当能精确回答的问题1？", "问题2？"],
  "taxonomy": {
    "domain": "AI 技术",
    "direction": "注意力机制",
    "tags": ["Transformer", "Attention"],
    "confidence": 0.85
  }
}

硬性规则：
1. 类型识别用于选择渲染倾向，不用于强制填充模板；不要为了匹配技术课、概念课或教程课而硬造固定内容。
2. 技术/公式/代码课如有字幕或视觉证据支持，可优先识别为 technical_formula，密度可用 dense，并抽取有证据的 formula/code/mechanism/pitfall 知识单元。
3. 操作/教程类只有在视频确有连续操作链路时才抽取 procedure；概念类只有在确有观点、例子、边界或误区时才抽取对应知识单元。
4. visual_evidence 只选择结构价值高的帧，公式、流程图、代码、维度变化、结构图优先；人物空镜默认不选。
5. path 必须逐字复制输入关键帧描述里的 path，不要改写。
6. points.quote 和 knowledge_units.quote 必须忠实于字幕原意；不确定时写空字符串。字幕由 ASR 自动识别，对技术术语常有同音/形近误识（如把 "Codex" 写成 "co-tax"、"Plan Mode" 写成 "plug months"、"快照" 写成 "快兆"、"replan" 写成 "repland"）。**当视频标题、其他字幕或画面 OCR 足以唯一判定原意时，请把 quote 中明显错误的术语换成正确术语，其余口语化措辞保持原样；不要重写整句、不要补充字幕里没有的内容、无法判断的写空字符串。**
7. mainline 要写成读者理解本视频的认知/操作路径，按用户消息中的 mainline_max 决定上限；相邻条目必须递进，不能只是章节标题，也不能用不同措辞重复同一含义。
8. mainline 禁止空泛条目，例如“介绍相关背景”“讲解核心知识”“总结全文内容”；每一条都要说明为什么下一步需要它。
9. chapters 数量必须按用户消息中的 chapters_min / chapters_max 选择；高密度视频优先增厚章节讲解，而不是机械增加章节数。**长视频（>30 分钟）必须在最后几分钟之前都有章节支撑，不能在前 1/3 堆章节而后面留空白。**
10. chapters[].teaching_notes 是连续讲义段落，不是短 bullet。每章尽量包含：承接前文的问题、核心解释、因果/流程/对比/例子/推导展开、1-2 个重点、必要边界，以及如何收束到下一步。
11. 每条 teaching_notes 都应是完整中文段落；每章通常 3-5 段，高密度章节可更多；避免“本章介绍了相关内容”这类无信息句。
12. 公式有字幕或画面证据时**必须**写入 chapters[].formula_blocks（一个公式一项，附 ts 与 explanation）；同时 teaching_notes 中可保留行内公式以保持上下文。LaTeX 用标准分隔符：$...$、$$...$$；化学式优先 mhchem，例如 \\ce{H2O}。
13. 代码有字幕或画面证据时**必须**写入 chapters[].code_blocks，每条独立保留缩进与换行；不要把代码塞进 teaching_notes 字符串。language 用 python/rust/cpp/sql/yaml/shell 这种已知短名。
13b. chapters[].process_steps：当本章包含连续可执行操作、机制阶段或调用顺序时，按顺序列出 3-8 个该可验证的步骤（例如 “调用 X 传入 Y”、“状态从 A 转为 B”）；纯概念或总结章节可留空。不要把 process_steps 写成表面句子（如 “讲解原理”），必须是实际动作。
14. 操作教程如果确有连续操作链路，应抽取多个 procedure 知识单元，每个 procedure 代表一个可执行或可检查阶段；如果只是理念讲解，不强行抽取 procedure。
15. timeline 要表达“如何推进”，不要只列章节标题。
16. completeness 是读完讲义后的闭环检查，要如实指出缺失，不要为了好看全部写已完成。
17. **若用户消息中提供了 study_questions 列表，必须把它们逐字写入输出 JSON 的 study_questions 字段，并确保每条问题都能在 chapters/knowledge_units 中找到对应回答；若某条问题字幕里完全无证据，应写进 completeness.missing_examples 或 missing_boundaries 而不是硬编。**
18. taxonomy 是必填字段，用于讲义库的多领域归类与首页折叠树，请严格按以下规则输出：
    - domain：必须从下面候选列表中选**一个**完全一致的字符串，不要自造、不要改大小写、不要加空格：
      "AI 技术" / "编程开发" / "数据科学" / "硬件与系统" / "数学" / "物理" / "化学生物" / "医学" /
      "烹饪" / "健身运动" / "金融投资" / "人文社科" / "艺术设计" / "工程实务" / "其他"。
    - direction：domain 下的具体方向短语，例如 "注意力机制" / "Rust 并发模型" / "中餐家常" / "心血管风险评估"；
      避免 "教程" / "入门" / "进阶" / "详解" / "教学" / "讲解" 这类教学阶段词，focus 在主题上。
    - tags：3-8 个核心概念/工具/方法名关键词，去重；中英文混合保留原样大小写，方便检索。
    - confidence：0-1 之间的自评。标题与字幕主题清晰时 ≥0.8；内容跨多领域或主题模糊时给 0.4-0.6；
      实在判断不出来再给 <0.4 并把 domain 写成 "其他"，便于后续人工归类。
    - 如果视频明显跨领域（例如同时讲算法和厨艺），不要硬选一个；写置信度低的最像的那个 + tags 把另一面补齐。
"""


LECTURE_IR_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- UP 主: {author}
- 时长: {duration} 秒（约 {minutes} 分钟）
- 封面: {cover_url}

长度预算（**必须**遵守）：
- chapters_min: {chapters_min}
- chapters_max: {chapters_max}
- mainline_max: {mainline_max}
- glossary_max: {glossary_max}
- review_questions_max: {review_questions_max}
- 密度提示: {density_hint}

学习问题（study_questions，可能为空）：
{study_questions_block}
{chapter_plan_block}
字幕（时间戳·文本）：
{subtitle_block}

关键帧视觉描述（时间戳·caption [· OCR]）：
{frames_block}

请输出 LectureIR JSON，并在 chapters/knowledge_units/code_blocks/formula_blocks 中确保每个 study_question 都有对应回答。
"""


CHAPTER_SPLIT_SYSTEM = """\
你是一名讲义大纲编辑。给定一段视频的字幕（含时间戳），请把它划分成 3-8 个连贯章节。
每个章节给出 title（中文短语）、start、end（秒）。
输出 JSON：{"chapters": [{"title": "...", "start": 0, "end": 0}]}。
"""


# ---------------------------------------------------------------------------
# M2.P4 MapReduceIRBuilder — per-chapter map prompt
# ---------------------------------------------------------------------------
# Used by ``MapReduceIRBuilder._map_chapter_with_retry``. The system
# prompt deliberately forbids lecture-level fields (``mainline`` /
# ``lecture_summary`` / global glossary) so the model keeps each call
# focused on one chapter — that is what makes the map stage cacheable
# under :func:`compute_prompt_hash` and lets the reduce stages run on
# small, deterministic inputs.

LECTURE_IR_MAP_CHAPTER_SYSTEM = """\
你是一名讲义抽取专家，正在为一段视频的**单一章节**生成结构化数据。给定字幕片段、关键帧描述和章节时间窗，请仅产出该章节范围内的内容；不要写主线、全片摘要、全局术语表、交叉引用等 lecture 级字段。

硬性规则：
1. 所有 ts 必须落在 [chapter_start_sec, chapter_end_sec] 时间窗内（含端点），单位是秒。
2. points[].quote 必须是字幕原文逐字片段（≤ 80 字），不允许总结或转述；找不到合适字幕时写空字符串。
3. 仅输出本章节字段，结构如下；其余字段（mainline / lecture_summary / 跨章节引用）禁止出现：
{
  "title": "中文章节标题",
  "summary": "一段完整中文概览，介绍本章解决的问题和推进路径",
  "learning_goal": "本章要解决的学习问题",
  "teaching_notes": ["完整中文段落1", "段落2"],
  "process_steps": ["如有连续可执行操作或机制阶段，按顺序写出；没有则留空"],
  "points": [{"text": "论点", "ts": 0, "quote": "字幕原话"}],
  "code_blocks": [{"language": "python", "code": "...", "ts": 0, "explanation": "...", "source": "ocr"}],
  "formula_blocks": [{"latex": "...", "ts": 0, "explanation": "..."}],
  "pitfalls": ["误区/边界 1"],
  "key_takeaways": ["核心收获 1"],
  "knowledge_units": [
    {"term": "术语", "definition": "本章给出的解释", "confidence": 0.0, "ts": 0}
  ]
}
4. confidence 是你对该术语在本章定义清晰度的自评（0-1）；越上下文充分越高。
5. teaching_notes 写完整中文段落，不要空泛 bullet；公式/代码必须落到 formula_blocks / code_blocks，不要塞进文字段落。
6. 如果本章字幕完全为空或与时间窗不匹配，仍返回上述结构，但 points 可为空，并把 summary 写成"本章字幕缺失，无法抽取"。
"""


LECTURE_IR_MAP_CHAPTER_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- 总时长: {duration_sec} 秒

本章节窗口（务必遵守）：
- 章节序号: 第 {chapter_index} 章
- 起始: {chapter_start_sec} 秒
- 结束: {chapter_end_sec} 秒
- 锚点提示: {anchor_text}

字幕（仅本章节范围，时间戳·文本）：
{subtitle_block}

关键帧视觉描述（仅本章节范围，时间戳·caption [· OCR]）：
{frames_block}

请输出该章节的 JSON。
"""


# ---------------------------------------------------------------------------
# M2.P4 MapReduceIRBuilder — reduce-global pass prompt
# ---------------------------------------------------------------------------
# After the map stage produces N chapter dicts and ``reduce_local``
# stitches them deterministically, this lightweight LLM call writes
# the four lecture-level fields the per-chapter calls were forbidden
# to touch. Output is intentionally minimal (no chapter internals)
# so the prompt stays small and the call rarely exceeds 30s on epic
# videos. The "REDUCE_GLOBAL" marker in the system text is also used
# by the test suite stub to route map vs. global responders.

LECTURE_IR_REDUCE_GLOBAL_SYSTEM = """\
你是一名讲义全局编辑（REDUCE_GLOBAL）。已知一段视频的所有章节摘要（每章 title + summary + 头部要点 + 候选术语），请输出整片 lecture 的：

- lecture_summary：一段中文综述（2-4 句话），点题、给出主要演化路径与边界。
- mainline：用户视角的认知/操作主线，每条说明"为什么需要这一步"，并标 chapter_index。
- glossary_resolved：术语去重后的最终定义；用户输入的 conflicts 已含每候选定义和 winning_chapter_index 候选，可直接采用或结合多章修订。
- cross_references：章节之间的依赖/对比/引用关系；找不到则输出空数组。

**绝不重写章节内部任何字段**（title / summary / points / code_blocks / formula_blocks 都不要再次输出）。
仅输出顶层 4 字段，结构如下：
{
  "lecture_summary": "...",
  "mainline": [{"step": 1, "title": "...", "ts": 0, "chapter_index": 1}],
  "glossary_resolved": [{"term": "...", "definition": "...", "winning_chapter_index": 1}],
  "cross_references": [{"from_chapter": 1, "to_chapter": 2, "relation": "depends_on|contrasts_with|elaborates|...."}]
}
"""


LECTURE_IR_REDUCE_GLOBAL_USER_TEMPLATE = """\
视频元数据：
- BV: {bv_id}
- 标题: {title}
- 时长: {duration_sec} 秒

章节摘要（按时间顺序，每条已含 title / summary / 头 3 个 points 文本）：
{chapter_digest}

术语冲突候选（来自 reduce_local；同一术语在多章中给出不同定义）：
{glossary_conflicts}

学习问题（study_questions）：
{study_questions_block}

请输出 lecture 全局摘要 JSON，注意只产出 4 个顶层字段。
"""
