# LectureIR：中间表示与 Schema

## 文档定位

`Internship/02` 提到了「LectureIR 是中间表示」「双投影」，但没有展开数据形态本身。本篇把项目里两份最重要的 Pydantic 文件拆开讲：

- `app/understand/ir.py` — `LectureIR` 内部表示（builder 直接产出）。
- `app/understand/schema.py` — `LectureJSON` 对外投影 + `LectureNoteIR` + `EvidenceIndex` + `Composition`。

加上 `app/understand/ir_sanity.py` 这层后置净化器，完整说清「LLM 产出的原始 JSON 怎么一步步变成可信、可校验、可双投影的结构化数据」。

## 一句话定位

`LectureIR` 是这个项目最重要的事实源。所有 LLM 输出最终都要塞回这一份 Pydantic 模型；HTML、Copilot 检索、MCP 工具调用全部基于它的投影。任何「LLM 给了什么」不重要，「LectureIR 长什么样」才决定下游一切。

## LectureIR 内部表示（`app/understand/ir.py`）

### 顶层 dataclass

`@d:\Diet_Agent_NEW\app\understand\ir.py:165-194`：

```python
class LectureIR(BaseModel):
    bv_id: str
    url: str
    title: str
    author: str
    duration: float
    cover: str
    profile: LectureProfile                  # 视频画像
    core_question: str                       # 一句话核心问题
    mainline: list[str]                      # 全局主线
    knowledge_units: list[KnowledgeUnit]     # 知识单元
    timeline: Timeline                       # 时间轴 / 论证路径 / 状态演化
    visual_evidence: list[VisualEvidence]    # 全局视觉证据池
    chapters: list[IRChapter]                # 章节列表
    completeness: Completeness               # 完整性自评
    render_plan: RenderPlan                  # 渲染开关
    final_synthesis: str                     # 总结
    review_questions: list[str]              # 复习题
    study_questions: list[str]               # Question-Driven 阶段产物
    taxonomy: Taxonomy | None                # 自动归类
```

注意区分两个 `LectureProfile`：

- `app/understand/profile.py::LectureProfile` 是**路由策略**（4 档 dataclass）。
- `app/understand/ir.py::LectureProfile` 是**视频画像**（Pydantic 模型，含 `primary_type` / `density` / `required_sections`）。两个同名但语义不同。

### IRChapter：章节是「教学单元 + 证据」的容器

`@d:\Diet_Agent_NEW\app\understand\ir.py:78-99`：

```python
class IRChapter(BaseModel):
    index: int                      # 1-based
    title: str
    start: float                    # 起始秒
    end: float                      # 结束秒
    summary: str                    # 一段完整段落
    learning_goal: str              # 学习目标
    teaching_notes: list[str]       # 教学笔记段落
    process_steps: list[str]        # 流程步骤
    points: list[IRPoint]           # 论点 + 引用
    frames: list[IRFrame]           # 该章关联的关键帧
    pitfalls: list[str]             # 陷阱
    key_takeaways: list[str]        # 要点
    knowledge_unit_ids: list[str]   # 引用的 KU id
    code_blocks: list[IRCodeBlock]  # 代码块
    formula_blocks: list[IRFormulaBlock]  # 公式块
```

`@model_validator(mode="after")` 强约束 `end >= start`，否则直接报错。

### IRPoint、IRFrame、IRCodeBlock、IRFormulaBlock

`@d:\Diet_Agent_NEW\app\understand\ir.py:39-76` 定义了 4 个原子数据类型：

```python
class IRPoint(BaseModel):
    text: str           # 精炼论点
    ts: float           # 时间戳（秒）
    quote: str          # 原字幕引用

class IRFrame(BaseModel):
    ts: float
    path: str
    caption: str
    ocr_text: str
    insight: str
    visual_type: VisualType  # diagram / formula / code / table / ui / slide_text / person / other
    selected_reason: str
    importance_score: float  # [0, 1]
```

`IRCodeBlock` 和 `IRFormulaBlock` 是「专为高质量呈现而生」的一等公民结构，避免把代码 / 公式塞到 `teaching_notes` 字符串里再被 Markdown 解析弄坏。

### KnowledgeUnit

`@d:\Diet_Agent_NEW\app\understand\ir.py:102-114`：

```python
class KnowledgeUnit(BaseModel):
    id: str                       # 全局唯一
    type: KnowledgeType           # concept / mechanism / formula / code / procedure / example / boundary / pitfall
    title: str
    term_short: str
    explanation: str
    ts: float
    quote: str
    chapter_index: int
    related_frames: list[str]
    prerequisites: list[str]
    depends_on: list[str]
    domain_tags: list[str]
```

`depends_on` / `prerequisites` 是一对显式的有向关系，让 Copilot 在回答「这个概念依赖什么」时能给出有依据的链路，而不是凭空联想。

### Timeline / VisualEvidence / Completeness / RenderPlan

- **`Timeline`** — `turning_points / argument_path / state_evolution / chapter_boundaries`，把视频当成一段「论证 + 状态变化」过程。
- **`VisualEvidence`** — 全局视觉池，每张图带 `importance_score / ocr_density / novelty_score / selected / linked_knowledge_unit`。`selected=True` 的图才进入「全局关键图解」区。
- **`Completeness`** — 自评字段：`mainline_closed / missing_steps / missing_examples / missing_boundaries / visual_coverage / topic_reusability / reader_can_understand_without_video`。后两条是 Critic 在 prompt 里直接读的关键约束。
- **`RenderPlan`** — 渲染开关 dict：`hero / learning_map / completeness_card / formula_section / code_section / procedure_section / final_synthesis / review_questions / glossary`。LLM 决定哪些区块需要在 HTML 里渲染。

### 顶层校验

`@d:\Diet_Agent_NEW\app\understand\ir.py:189-194`：

```python
@model_validator(mode="after")
def _check_timestamps(self) -> "LectureIR":
    for ch in self.chapters:
        if ch.end > self.duration + 1.0:
            raise ValueError(f"chapter {ch.index} end {ch.end} > duration {self.duration}")
    return self
```

容忍 1 秒越界（处理 LLM 把章节结束时间填到 `duration + 0.x` 这种小漂移）。

## hydrate_lecture_ir_data：从 LLM 原始 JSON 到 IR

`@d:\Diet_Agent_NEW\app\understand\ir.py:197-237` 的 `hydrate_lecture_ir_data(data, ctx)` 是一段「容错巨长但目的非常单一」的函数。它的工作流程：

1. 把 `bv_id / url / title / author / duration / cover` 从 `ctx` 写回 `data`（LLM 可能漏写）。
2. `_hydrate_profile` — profile 字段缺失时用 `infer_primary_type` 启发式补全（基于标题 + 字幕前 3000 字 + 帧文本前 2000 字）。
3. `_normalise_chapters` — 章节兜底。`title` 缺失 → `f"第 {idx} 章"`；`teaching_notes` 缺失 → 用 summary 填；timestamps 缺失 → `_fill_chapter_ranges` 自动按视频时长等分。
4. `sanitize_ir_data(data)` — Q1-Q4 后置净化（详见下一节）。
5. `_normalise_visuals` — 把 `visual_evidence` 与 `ctx.frame_descs` 按 path 合并，丢失字段用 VLM 缓存里的值兜底。
6. `_attach_visuals_to_chapters` — 没明确归属章节的图，按 `ts ∈ [ch.start, ch.end]` 自动分配，每章上限由 `chapter_frame_max(ch_seconds)` 决定。
7. `_normalise_units` — 知识单元的 `type` 必须落在 `KnowledgeType.__args__`，时间戳用 `_locate_anchor_ts` 锚定。
8. `_hydrate_timeline / _hydrate_completeness / _hydrate_render_plan` — 三个面向 HTML 的字段兜底。
9. `final_synthesis` / `review_questions` / `study_questions` 兜底。

整段函数的核心思想是「**LLM 给什么就用什么，缺了就启发式补，错位了就纠正**」，让最终交给 Pydantic 校验的 dict 永远能通过。

### _locate_anchor_ts：处理 LLM 时间戳幻觉

`@d:\Diet_Agent_NEW\app\understand\ir.py:460-486` 是一段很有代表性的「LLM 输出校正」：

> LLM frequently writes `ts=0` even when the chapter starts deep into the video, or echoes back the minute portion of `[MM:SS]` prompt formatting.

策略：

- LLM 给的 `ts` 在 chapter 范围 `[start-5, end+5]` 内 → 信任。
- 越界 → 用 `quote / text / title` 在字幕里反查最接近的位置。
- 字幕也找不到 → 退化到 chapter 中点。

这是项目里「不信 LLM 输出，但也不抛弃 LLM 输出」的经典写法。

### _repair_point_timestamps：上游修复 + 下游兜底

`@d:\Diet_Agent_NEW\app\understand\ir.py:489-499` 在 `_fill_chapter_ranges` 之后专门跑一次，把每个 point 的 `ts` 钳回所属 chapter 的范围内。这是 commit `3f87c1a` 修过的「point timestamps out of chapter range」问题的修复点。

## ir_sanity：Q1-Q4 后置净化

`@d:\Diet_Agent_NEW\app\understand\ir_sanity.py`（12452 字节）专门处理 4 类典型 LLM 错误：

- **Q1**：把 `process_steps` 开头的「1.」「2、」「①」等列表序号前缀剥掉。
- **Q2**：当 `mainline` / `core_question` 折叠为「章节标题的回声」（例如 mainline 全部 `=` 章节标题），清空它们，让下游用 fallback 重建。
- **Q3**：章节 `index` 字段重新连续编号 `1..N`，让 chapter 间引用稳定。
- **Q4**：mainline / glossary 含通用占位词（「介绍背景」「讲解核心知识」「可控性」「网页方案」等）时降权或剔除。`@d:\Diet_Agent_NEW\app\understand\ir_builder.py:48-72` 定义了这两份 banlist。

净化器把每一处修改写进 `ctx.ir_sanity` 报告，供 pipeline_stats / 验证脚本看「LLM 又犯了什么错」。

## LectureJSON：对外投影

`@d:\Diet_Agent_NEW\app\understand\schema.py:454-557` 是真正落地数据库 / 渲染 HTML / 入 RAG 的对外结构：

```python
class LectureJSON(BaseModel):
    bv_id: str
    url: str
    title: str
    author: str
    duration: float
    cover_url: str
    one_liner: str                # 必填，一句话总结
    category: str = "lecture"
    domain_tags: list[str]
    learning_path: list[str]
    chapters: list[Chapter]       # 注意是 schema.Chapter 不是 ir.IRChapter
    final_synthesis: str
    highlights: list[Highlight]
    glossary: list[GlossaryItem]
    review_questions: list[str]
    profile: LectureProfileView
    core_question: str
    mainline: list[str]
    timeline: TimelineView
    completeness: CompletenessView
    render_plan: RenderPlanView
    knowledge_units: list[KnowledgeUnitView]
    visual_evidence: list[Frame]
    composition: CompositionView
    lecture_blueprint: LectureBlueprint | None     # 双投影：blueprint
    lecture_note_ir: LectureNoteIR | None          # 双投影：note 视角
    evidence_index: EvidenceIndex | None           # 双投影：evidence 视角
    study_questions: list[str]
    taxonomy: Taxonomy | None
    generation_mode: str = "v1"
```

### 与 LectureIR 的区别

- `LectureJSON` 类型的字段都是面向消费层（HTML / RAG / Copilot），**带 `View` 后缀的子模型是 IR 字段的「展示视图」**，例如 `LectureProfileView` 没有 `secondary_types` / `domain_tags`，只保留 HTML 用得到的几个字段。
- 多出 4 个字段（`one_liner / category / learning_path / generation_mode`）是面向 SQLite 列和 UI 的，IR 里没有。
- 多出 4 个**投影结构**（`lecture_blueprint / lecture_note_ir / evidence_index / composition`），由 `lecture_compiler.compile_lecture` 填充，**LectureIR 本身不持有这些**。

`generation_mode` 取值 `"v1" / "lecture_ir_v2" / "v1_fallback" / "map_reduce_v2"`，是验证脚本和 UI 用来识别「这份讲义经历了哪条路径」的关键。

### 时间戳健壮性校验

`@d:\Diet_Agent_NEW\app\understand\schema.py:496-513`：

```python
@model_validator(mode="after")
def _check_timestamps(self) -> "LectureJSON":
    for ch in self.chapters:
        if ch.end > self.duration + 1.0:
            raise ValueError(...)
        for p in ch.points:
            if p.ts > self.duration + 1.0:
                raise ValueError(...)
        for f in ch.frames:
            if f.ts > self.duration + 1.0:
                raise ValueError(...)
    return self
```

下游 RAG `chunk.t_start/t_end` 强类型 `int | None`，所以 LectureJSON 层的时间戳越界会被立刻挡住。

### `validate_quotes` / `mark_unverified_points`

`@d:\Diet_Agent_NEW\app\understand\schema.py:515-552` 给出两个原文校验工具：

- `validate_quotes(transcript)` — 返回错误列表（用于自动验证脚本）。
- `mark_unverified_points(transcript)` — 在 `Point.text` 前加 ⚠️ 前缀（让 UI 自然显示「这一句的原文锚定不可信」）。

匹配策略写在 `@d:\Diet_Agent_NEW\app\understand\schema.py:36-62`：先 normalise（去标点空白）后做 substring；如果 quote ≥ 6 个字符，再做 4-gram 重叠率 ≥ 0.6 的模糊匹配。这一段是为了**容忍 LLM paraphrase** 又**抓住 LLM 编造原文**这两个目标之间找平衡。

## 双投影的具体结构

### LectureBlueprint（编译过渡产物）

`@d:\Diet_Agent_NEW\app\understand\schema.py:298-307` 含 `front_matter_plan / body_unit_plan / back_matter_plan` 三段，描述「讲义应该怎么组织」的策略。它是 `compile_lecture` 内部产物，外部消费方一般直接读 `note_ir`。

### LectureNoteIR（面向读者）

`@d:\Diet_Agent_NEW\app\understand\schema.py:346-381` 三段式：

```python
class LectureNoteIR(BaseModel):
    front_matter: LectureNoteFrontMatter
    body: LectureNoteBody              # 含 teaching_units: list[TeachingUnit]
    back_matter: LectureNoteBackMatter
```

`TeachingUnit`（`@d:\Diet_Agent_NEW\app\understand\schema.py:332-343`）每个单元含 `unit_id / title / teaching_goal / unit_role / core_message / content_blocks / visual_slots / evidence_refs / source_chapter_refs`。

`ContentBlock`（`@d:\Diet_Agent_NEW\app\understand\schema.py:320-329`）的 `block_role` 取值 `claim / process / boundary / synthesis / mechanism`，由 `lecture_compiler` 按章节风格决策。

### EvidenceIndex（面向系统）

`@d:\Diet_Agent_NEW\app\understand\schema.py:393-421`：

```python
class EvidenceIndex(BaseModel):
    evidence_objects: list[EvidenceObject]
    evidence_relations: list[EvidenceRelation]
    anchor_map: dict[str, Any]         # {video_anchor, page_anchor, evidence_anchor}
    projection_views: dict[str, Any]   # {source_index_view, glossary_view, ...}
```

`EvidenceObject` 含 5 类：`chapter_evidence / quote_evidence / frame_evidence / procedure_evidence / concept_evidence`，每个都内嵌一个 `RagChunk` 列表，方便 RAG 索引器直接吃。

`EvidenceRelation.relation` 目前用到的有 `supports / derived_from`。

详见 `Detail/07-双投影编译.md`。

## Taxonomy：自动归类

`@d:\Diet_Agent_NEW\app\understand\schema.py:438-451`：

```python
class Taxonomy(BaseModel):
    domain: str = ""        # 归一化后必须落在 DOMAIN_WHITELIST
    direction: str = ""
    tags: list[str] = []
    confidence: float = 0.5
```

LLM 在 IR 生成时一并产出 `taxonomy`，之后被 `app/copilot/taxonomy.py::normalize` 修复（白名单 + 后缀剥离 + 标签去重 + 置信度钳定）。详见 `Detail/09-Copilot-Agent-与-SSE.md`。

## RagChunk：把证据导向检索

`@d:\Diet_Agent_NEW\app\understand\schema.py:383-390`：

```python
class RagChunk(BaseModel):
    kind: str = "teaching_note"
    chapter_idx: int
    t_start: float
    t_end: float
    text: str
    note_node_id: str
    meta: dict[str, Any]
```

EvidenceIndex 里每个 `EvidenceObject.rag_chunks` 直接列出该证据应当被切成的 chunk。`copilot/rag.py::chunk_lecture` 在有 `evidence_index` 时优先用它，否则才走「chapter / point / pitfall / frame」回退路径。

## 写简历可以怎么提炼

- 设计了讲义中间表示 `LectureIR`，把视频内容结构化为 profile / mainline / chapters / knowledge_units / visual_evidence / completeness / render_plan 7 个互相正交的字段，并通过 Pydantic 校验把「LLM 给的脏数据」和「下游能安全消费」的边界守住。
- 实现了 `hydrate_lecture_ir_data` + `sanitize_ir_data` 两层后置净化，处理 LLM 时间戳幻觉、列表序号前缀污染、mainline 与章节标题塌缩、通用占位词污染等典型问题，并把每一处修改打成 telemetry。
- 通过 LectureJSON 把 IR 投影成对外结构，附带 `lecture_blueprint` / `lecture_note_ir` / `evidence_index` / `composition` 四套子结构，使 HTML 渲染、RAG 索引、Copilot 工具调用、MCP 对外接口共用同一份事实源。
- 为 `Point.quote` 与 `transcript` 之间设计了「精确 substring + 4-gram 模糊」双层匹配，既能容忍 LLM 轻度 paraphrase 又能在严重幻觉时给点打上 ⚠️ 标记。

## 面试可展开点

### 1. 为什么不直接让 LLM 输出 LectureJSON

两个原因：

- `LectureJSON` 多了 `lecture_blueprint / lecture_note_ir / evidence_index / composition` 四套结构，这些是**确定性编译**的产物，让 LLM 输出反而会引入更多幻觉。
- IR 是项目里的事实源；如果 LLM 直接输出最终结构，未来一次 schema 微调可能要重提示词。隔一层让 schema 演进对 prompt 透明。

### 2. 为什么把代码 / 公式做成一等公民

代码 / 公式塞到 `teaching_notes` 字符串里，Markdown 渲染会丢缩进、KaTeX 会因转义出错、RAG 会切丢上下文。一等公民对象保留 `language / code / explanation / related_frame_paths`，HTML 可以单独走 `<pre><code>` 高亮 + KaTeX 渲染，RAG 也能给它独立 chunk kind。

### 3. _locate_anchor_ts 为什么要这么写

因为 LLM 时间戳容易出错（ts=0 / 只取分钟 / 越界）。但完全不信 LLM ts 又会丢失它对原文位置的「软知识」。这段函数的做法是「LLM 信号优先，但用结构性约束兜底」，是典型的「LLM + 规则混合」实现。

### 4. mark_unverified_points 为什么选 4-gram 阈值 0.6

太低 → 任何短词重叠都通过，幻觉漏抓；太高 → LLM 改一个词就 false-positive。4-gram + 0.6 是手动验过的均衡点，写在注释里有完整动机。

### 5. taxonomy 为什么要在 schema 层留 None

老的 LectureIR 调试 dump 里没有 taxonomy 字段；如果设成必填，老 BV 的 HTML 重渲染会因为 Pydantic 校验失败。设 None 默认让历史数据 replay 安全。

## 当前文档中的事实与推断

### 事实

- `LectureIR` 和 `LectureJSON` 是两份独立 Pydantic 模型，分别在 `app/understand/ir.py` 和 `app/understand/schema.py`。
- `LectureIR.profile` 与 `app/understand/profile.py::LectureProfile` 同名但语义不同（一个是视频画像，一个是路由策略）。
- `hydrate_lecture_ir_data` 内含 9 步后置兜底；`ir_sanity` 处理 4 类典型 LLM 错误。
- `LectureJSON` 里有 4 个可选投影字段，由 `lecture_compiler` 填充。
- `RagChunk` 是 EvidenceObject 内嵌结构，向 RAG 直接传导。

### 推断

- 这一层是项目里「Pydantic + 启发式后置 + 错误容忍」三件套最典型的实现，足以独立成为 LLM-Agent 应用的设计模式参考。
