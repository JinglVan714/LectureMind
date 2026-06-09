# 双投影编译：Lecture Note IR 与 Evidence Index

## 文档定位

`Internship/02` 把「双投影」作为项目的架构主卖点，但没有展开**编译器到底做了哪几步**。本篇细化 4 个文件：

- `app/understand/lecture_compiler.py` — 编译总线，串起 5 个阶段。
- `app/understand/lecture_blueprint.py` — 把 LectureJSON 变成 `LectureBlueprint`（中间策略）。
- `app/understand/evidence_index.py` — 构造 `EvidenceIndex` + `RagChunk[]`。
- `app/understand/composition.py` — `CompositionView`，HTML 渲染用的兼容投影。

## 一句话定位

「编译」不是 IR → HTML 的简单格式转换，而是把 `LectureIR` 同时投影成「面向读者的 `LectureNoteIR`」和「面向系统的 `EvidenceIndex`」两套结构，并附带一个 `LectureBlueprint`（编排策略）和一个 `CompositionView`（渲染兼容层）。

## 编译总线：`compile_lecture`

`@d:\Diet_Agent_NEW\app\understand\lecture_compiler.py:21-36`：

```python
def compile_lecture(lecture_ir: LectureIR) -> LectureJSON:
    materials = material_normalization(lecture_ir)
    reading = semantic_reading(materials)
    blueprint = build_lecture_blueprint(materials["legacy_lecture"], reading)
    note_ir = note_composition(materials["legacy_lecture"], blueprint, reading)
    evidence_index = build_evidence_index(materials["legacy_lecture"], blueprint, note_ir)

    lecture = project_legacy_lecture_json(lecture_ir)
    lecture.lecture_blueprint = blueprint
    lecture.lecture_note_ir = note_ir
    lecture.evidence_index = evidence_index
    lecture.composition = compose_from_lecture_json(lecture)
    return lecture
```

整条编译链 5 步：

1. **material_normalization** — 把 `LectureIR` 投影成 `LectureJSON`（legacy_lecture）。
2. **semantic_reading** — 计算阅读画像（语义类型 / 阅读负担 / 教学重心）。
3. **build_lecture_blueprint** — 按 reading 把 lecture 拆成「编排计划」。
4. **note_composition** — 把 blueprint + lecture 翻译成 `LectureNoteIR`。
5. **build_evidence_index** — 把 lecture + blueprint + note_ir 投影成 `EvidenceIndex`。

最后 `compose_from_lecture_json` 把 note + evidence 兼容投影回 `CompositionView` 给 HTML 渲染。

## semantic_reading：阅读画像决策

`@d:\Diet_Agent_NEW\app\understand\lecture_compiler.py:51-77`：

```python
def semantic_reading(materials):
    primary_type = lecture_ir.profile.primary_type
    semantic_portrait_map = {
        "procedural_tutorial": "流程教学",
        "technical_formula": "技术讲解",
        "conceptual_talk": "概念讲解",
        "generic_lecture": "综合讲座",
    }
    semantic_portrait = semantic_portrait_map.get(primary_type, "综合讲座")

    if lecture.composition.summary_mode in {"handout", "segmented_handout"} or lecture.duration >= 1800:
        reading_burden = "high"
    elif lecture.duration >= 900 or len(lecture.chapters) >= 4:
        reading_burden = "medium"
    else:
        reading_burden = "low"

    if any(ch.process_steps for ch in lecture.chapters):
        center = "process"
    elif any(ch.key_takeaways for ch in lecture.chapters):
        center = "claim"
    else:
        center = "mechanism"

    return {"semantic_portrait", "reading_burden", "teaching_center_of_gravity"}
```

三个信号共同决定后续编译策略：

- **semantic_portrait** — 4 选 1：流程教学 / 技术讲解 / 概念讲解 / 综合讲座。
- **reading_burden** — 3 选 1：low / medium / high。30 分钟以上自动 high。
- **teaching_center_of_gravity** — 3 选 1：process（有流程步骤）/ claim（有要点）/ mechanism（默认）。

## LectureBlueprint：编排计划

`@d:\Diet_Agent_NEW\app\understand\lecture_blueprint.py:14-89` 的 `build_lecture_blueprint(lecture, semantic_reading)` 产出三段：

### front_matter_plan

包含 `one_sentence_claim / reader_orientation / takeaways_top / reading_map / reader_prerequisites / suitable_for / not_suitable_for`。

- `takeaways_top` 取所有章节的 `key_takeaways` 前 4 个，缺则用 `learning_path` 或 `final_synthesis`。
- `reader_prerequisites` 取术语前 4 个，缺则用 `domain_tags` 或 `category`。
- `suitable_for` / `not_suitable_for` 与 `semantic_portrait` + `reading_burden` 挂钩，例如 `reading_burden=high` 时建议「愿意阅读结构化长讲义的读者」。

### body_unit_plan

每章一个 `BodyUnitPlan`：

```python
BodyUnitPlan(
    unit_id=f"teaching-unit-{chapter.index}",
    title=chapter.title,
    teaching_goal=chapter.learning_goal or chapter.summary,
    unit_role=_infer_unit_role(chapter, ...),  # claim / mechanism / process / case / problem / boundary / synthesis
    core_message=_chapter_core_message(chapter),
    transition_from_previous="从「<prev_title>」继续推进到「<curr_title>」。",
    source_chapter_refs=[chapter.index],
    visual_needs=[{...}],
    appendix_candidates=[...],
)
```

`_infer_unit_role` 启发式规则（`@d:\Diet_Agent_NEW\app\understand\lecture_blueprint.py:110-124`）：

- 最后一章 + 有 pitfalls / 标题含「总结」/ `synthesis` → `synthesis`
- 有 pitfalls / 标题含「边界 / 风险」→ `boundary`
- 有 process_steps / 标题含「步骤 / 流程 / 过程」→ `process`
- 标题含「案例 / case」→ `case`
- 标题含「为什么 / why / 问题」→ `problem`
- 标题含「机制 / 原理 / mechanism」→ `mechanism`
- 默认 → `claim`

这种「7 种 unit_role」是双投影面向读者侧的核心抽象——它决定了 HTML 渲染时该用什么模板组织段落。

### back_matter_plan

- `boundary_and_risks` — 所有章节 pitfalls 的前 6 条。
- `source_index_entrypoints` — 前 4 个 unit 的 title，作为「证据入口」给读者跳转。
- `appendices` — 含 pitfalls 的章节自动加 `boundary_review` appendix。
- `transfer_and_next_steps` — `review_questions` 前 4 个，缺则用 `study_questions` 前 4 个。

## LectureNoteIR：面向读者投影

`@d:\Diet_Agent_NEW\app\understand\lecture_compiler.py:80-126` 的 `note_composition(lecture, blueprint, semantic_reading)`：

每个 chapter → 一个 `TeachingUnit`：

```python
TeachingUnit(
    unit_id=plan.unit_id,
    ordinal=str(chapter.index),
    title=plan.title or chapter.title,
    teaching_goal=plan.teaching_goal,
    unit_role=plan.unit_role,
    core_message=_dedupe_unit_core_message(plan.core_message, content_blocks),
    transition_from_previous=plan.transition_from_previous,
    content_blocks=content_blocks,     # 关键
    visual_slots=_build_visual_slots(chapter),
    evidence_refs=_build_unit_evidence_refs(chapter),
    source_chapter_refs=[chapter.index],
)
```

### ContentBlock：按内容类型组织段落

`_build_content_blocks(chapter, semantic_reading)`（`@d:\Diet_Agent_NEW\app\understand\lecture_compiler.py:129-164`）按 `_classify_content_kind` 分三种内容类型：

- **conceptual** — `_build_conceptual_block_specs`：主段（summary + teaching_notes + 第一个 key_takeaway）+ 可选「边界与风险」块。
- **procedural** — `_build_procedural_block_specs`：主段 + 「步骤」块 + 可选「容易出错的地方」。
- **technical** — `_build_technical_block_specs`：claim + mechanism + process（如有）+ synthesis + boundary。

`block_role` 取值：`claim / mechanism / process / synthesis / boundary`。每个 block 含：

```python
ContentBlock(
    block_id=f"teaching-unit-{chapter.index}-block-{i}",
    title="" | "步骤" | "边界与风险" | ...,
    block_role=...,
    paragraphs=[...],
    source_chapter_refs=[chapter.index],
    source_timestamps=[float(chapter.start)],
    evidence_refs=["chapter-N", "quote-N-M", ...],
)
```

### 去重：避免「summary 和第一个段落几乎一样」

`_texts_highly_overlap` 用 `normalize + 长度比 + 包含关系 + 字符集 Jaccard`（`@d:\Diet_Agent_NEW\app\understand\lecture_compiler.py:442-454`）。`_dedupe_unit_core_message`、`_merge_distinct_paragraphs`、`_is_redundant_against_group` 都基于它。这一层让最终读到的讲义不会有大段重复。

### Visual Slots

`_build_visual_slots(chapter)` 把章节关联的前 2 张帧封成 VisualSlot：

```python
VisualSlot(
    slot_id=f"visual-slot-{chapter.index}-{index}",
    visual_role="process_figure" if frame.visual_type in {"code", "formula"} else "keyframe_explainer",
    title=frame.caption or f"Visual {index}",
    caption=frame.insight or frame.caption,
    source_paths=[frame.path],
    evidence_refs=[f"frame-{chapter.index}-{index}"],
    ts=float(frame.ts),
)
```

`visual_role` 取值与 `evidence_refs` 都和 EvidenceIndex 里的 `evidence_id` 一一对应，让读者侧和系统侧通过 string id 连接。

## EvidenceIndex：面向系统投影

`@d:\Diet_Agent_NEW\app\understand\evidence_index.py:8-328` 的 `build_evidence_index(lecture, blueprint, note_ir)` 是项目最核心的「数据投影器」。

### 5 类证据对象

每个章节 / 论点 / 关键帧 / 流程步骤 / 概念都被映射为不同 kind 的 `EvidenceObject`：

| kind | 来源 | id 模式 | RagChunk kind |
|------|------|---------|--------------|
| `chapter_evidence` | 章节本身 | `chapter-{N}` | `teaching_note` |
| `quote_evidence` | 章节 points | `quote-{N}-{M}` | `quote` |
| `frame_evidence` | 章节 frames | `frame-{N}-{M}` | `frame_ocr` |
| `procedure_evidence` | 章节 process_steps | `procedure-{N}-{M}` | `teaching_note` |
| `concept_evidence` | glossary | `concept-{N}` | `knowledge_unit` |
| `compound_evidence` | blueprint 的 unit plan | `compound-{unit_id}` | — |

每个 `EvidenceObject` 含：

```python
EvidenceObject(
    evidence_id="chapter-3",
    kind="chapter_evidence",
    title=chapter.title,
    summary=chapter.summary,
    chapter_index=chapter.index,
    text=chapter.summary,
    ts=float(chapter.start),
    t_end=float(chapter.end),
    note_node_ids=[unit_id],     # 关联到 TeachingUnit 的 id
    anchors=evidence_anchor["chapter-3"],
    rag_chunks=[RagChunk(...)],   # 直接喂给 RAG 索引器
    source_payload={"chapter_title": chapter.title},
)
```

### EvidenceRelation：证据间的有向关系

`@d:\Diet_Agent_NEW\app\understand\evidence_index.py:297-308` 显式建立两类关系：

- `supports` — 该证据支撑某个 note_node（例如 `chapter-3 supports teaching-unit-3`）。
- `derived_from` — note unit 是从原 chapter 派生的（`teaching-unit-3 derived_from chapter-3`）。

这两条关系让 Copilot 能在回答时回答「这条 claim 的证据是哪里来的」。

### anchor_map：三套锚点统一索引

```python
anchor_map = {
    "video_anchor": {"t=120": {"t_start": 120.0, "t_end": 240.0}, ...},
    "page_anchor": {"teaching-unit-3": {...}, "teaching-unit-3-block-1": {...}, ...},
    "evidence_anchor": {"chapter-3": {video_anchor: "t=120", page_anchor: "teaching-unit-3", ...}, ...},
}
```

`_register_anchor(...)` 在创建每个证据对象时统一注册三种锚点，下游 Copilot 拿到 `evidence_id` 就能跨视频时间点和讲义页面锚点跳转。

### projection_views

```python
projection_views = {
    "source_index_view": [{label: "[t=02:00]", supports_text: "支撑讲义单元：xxx", evidence_id, page_anchor, video_anchor, kind}],
    "glossary_view": [{term, explanation, evidence_id}],
    "appendix_view": blueprint.appendix_candidates,
    "chapter_map": [{chapter_index, title, start, end}],
    "evidence_quotes": [{label, quote, ts, chapter_index}][:6],
    "term_index": [{term, ts, explanation}][:8],
    "tool_appendix": [],
}
```

这些是给 HTML 渲染层和 Copilot 工具调用直接消费的视图——已经投影好的字典，不需要重新计算。

## CompositionView：HTML 兼容投影

`@d:\Diet_Agent_NEW\app\understand\composition.py:34-89` 的 `compose_from_lecture_json(lecture)` 有两条路径：

- **`compat_projection_from_note_and_evidence`** —— 当有 `lecture_note_ir + evidence_index` 时（双投影都成功），用它们派生 `CompositionView`。
- **legacy 路径** —— 没有双投影时，按 chapter / glossary 自己拼。

legacy 路径里有一个 `_summary_mode` 决策：

- `burden_score < 0.18` → `skim`
- `< 0.45` → `note`
- `< 0.72` → `handout`
- 否则 → `segmented_handout`

`burden_score` 由多个信号加权而来（duration / chapter_count / avg_chapter_length / 等）。这个分数同时影响 HTML 渲染的「该用紧凑还是分段长文」。

### ASR 错误词替换

`@d:\Diet_Agent_NEW\app\understand\composition.py:20-31`：

```python
_ASR_TERM_REPLACEMENTS = (
    ("可不搞", "口播稿"),
    ("开外计划", "开发大纲"),
    ("上下门管理", "上下文管理"),
    ("注意注意", "注意"),
)
_BODY_NOISE_MARKERS = ("语音识别错误", "字幕中为", "术语在字幕中", "实际应为")
```

这是真实跑过哪些 BV 才知道要专门加的「ASR-paraphrase 反幻觉规则」。LLM 有时把 ASR 错误词标注成「字幕中为 X，实际应为 Y」混进段落里，composer 会把这种段落丢掉。

## 三层投影一图回顾

```
LectureIR (内部表示)
    │
    │  compile_lecture
    ▼
LectureJSON
    ├── lecture_blueprint   (编排策略 / 中间产物)
    ├── lecture_note_ir     (面向读者：teaching_units → content_blocks → visual_slots)
    ├── evidence_index      (面向系统：evidence_objects → rag_chunks → anchor_map)
    └── composition         (面向 HTML：body_sections → supporting_visuals)
```

每一层都可以独立持久化、独立校验、独立替换。例如未来想换一种渲染风格，只改 `composition` 一层就够，`note_ir` 和 `evidence_index` 不用动。

## 写简历可以怎么提炼

- 实现了讲义中间表示到面向读者 / 面向系统两套结构的双投影编译器：`LectureNoteIR` 强调教学单元 / 段落分组 / 视觉槽位；`EvidenceIndex` 强调证据对象 / 关系图 / 三套锚点（video / page / evidence）。
- 设计了 7 种 `unit_role`（claim / mechanism / process / case / problem / boundary / synthesis）的启发式分类，让 HTML 渲染层能按章节内容类型选择不同的模板与段落组织策略。
- 引入了 5 类 `EvidenceObject`（chapter / quote / frame / procedure / concept / compound）的对象模型 + 两类有向关系（supports / derived_from），并在每个证据上内嵌 `RagChunk` 列表，使 RAG 索引器、Copilot 工具调用、MCP 对外接口共享一份证据图。
- 通过 ASR 错误词替换 + 噪声段标记过滤，把 LLM 在 ASR-paraphrase 场景下的幻觉污染挡在渲染层之外。

## 面试可展开点

### 1. 为什么 EvidenceIndex 里的每条对象都自带 RagChunk

把 chunk 表示绑在证据对象上而不是单独跑一遍 chunker，可以让「证据→chunk」的映射是确定性、可审计的。Copilot 拿到一条 chunk hit 就能反查出对应的 evidence_id，然后顺着 anchor_map 回到视频时间点和讲义页面位置。这是 note-first / evidence-second 检索模式的基础。

### 2. anchor_map 为什么要三套（video / page / evidence）

读者关心「这一段对应视频哪一秒」（video_anchor），系统关心「这一段对应哪一段讲义节点」（page_anchor），Copilot 关心「这一段对应哪一个证据对象」（evidence_anchor）。三套锚点同时存让不同消费方都能直接索引，不用每次都做联表查找。

### 3. LectureBlueprint 为什么是「中间策略」而不是直接产出 NoteIR

把「这一章应该是哪种 unit_role」「需要哪些 visual_needs」这种**策略性决定**单独成 blueprint，让 note_composition / evidence_index 两条投影路径都能读同一份策略。如果策略直接嵌在 note 里，evidence 想用就得反向解码 note，跨结构耦合就开始了。

### 4. 为什么要做 `_texts_highly_overlap` 这种字符级去重

LLM 在写章节 summary 和 teaching_notes 时常常重复同样意思。如果不去重，HTML 渲染出来一段 200 字的 summary 紧跟着 4 段几乎一样的 teaching_notes。字符级去重虽然粗，但对中文 paraphrase 鲁棒，比起跑 embedding 相似度便宜得多。

### 5. ASR_TERM_REPLACEMENTS 为什么要硬编码

这些是实跑过真实 BV 才发现的错误词。从 LLM 角度它们没什么共同特征，没法靠规则推出来。硬编码 + 注释里写明出现过的场景，是「补丁式工程」典型操作，比起花一周改 prompt 性价比高得多。

## 当前文档中的事实与推断

### 事实

- 编译总线 5 步：material → reading → blueprint → note → evidence。
- `unit_role` 有 7 种，由 `_infer_unit_role` 启发式分类。
- `EvidenceObject.kind` 有 6 种（含 compound）。
- `EvidenceRelation.relation` 用到 `supports` / `derived_from` 两种。
- `anchor_map` 三套锚点统一注册，由 `_register_anchor` 在创建证据对象时触发。

### 推断

- 双投影 + 三套锚点 + 5 类证据这种「面向消费形态做结构设计」是项目最值得在面试里展开的架构亮点之一。
