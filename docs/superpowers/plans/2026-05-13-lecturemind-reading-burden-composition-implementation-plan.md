# LectureMind 目录化阅读单元与轻量讲义化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在现有 `summary_mode + composition` 主链已接通的基础上，把 handout 主稿从 `body_sections` 平铺升级为“目录化阅读单元”，让用户一眼知道当前在读哪一部分，并显著降低长段拥挤感。

**架构：** 保留现有 `LectureIR -> lecture_ir_to_lecture_json -> Renderer` 主链，不重写底层 IR 生成。变化集中在 `app/understand/composition.py` 的正文编排层：为每个 `body_section` 生成面向用户的 `outline_blocks`，并保留开发期内部提示字段用于生成与验收。模板与样式只消费用户可见字段，不渲染内部提示，同时继续坚持 `light reorder` 和“视频原有表达优先”。

**技术栈：** Python 3.11、Pydantic v2、Jinja2、pytest

---

## 文件结构

### 修改

- `app/understand/schema.py`
  - 为 `body_sections` 增加 `outline_blocks` 视图模型，显式区分用户可见字段和开发期内部提示字段。
- `app/understand/composition.py`
  - 在现有 section 投影后生成目录化阅读单元，补 block 标题、层级编号、时间锚点，并收敛统一模板腔。
- `app/render/renderer.py`
  - 把 `outline_blocks` dump 到模板层，保证内部提示字段不会直接进入渲染上下文。
- `app/render/templates/lecture.html.j2`
  - 把 section 正文从单层 paragraph 流改成 `section -> outline_block -> paragraph` 层级。
- `app/static/lecture.css`
  - 增加 block 层级、编号、锚点、段间留白样式，解决“挤成一坨”的观感问题。
- `tests/test_composition.py`
  - 定向覆盖：outline block 生成、轻量改述约束、不同 profile 的轻量表达差异、内部提示字段只停留在数据层。
- `tests/test_smoke.py`
  - 定向覆盖：HTML 中能看到层级编号和时间锚点，但看不到开发期内部提示字段。
- `docs/superpowers/specs/2026-05-13-lecturemind-standalone-handout-composition-design.md`
  - 与本计划保持同步；若执行中发现字段名或约束调整，必须先回写 spec。
- `docs/superpowers/handoffs/2026-05-13-lecturemind-reading-burden-composition-handoff.md`
  - 记录本阶段推进状态、真实样本变化、下一步切入点和验证口径。

### 不改

- `app/understand/prompts.py`
  - 本阶段不扩 prompt，先在现有 IR 素材上做更强的编排与渲染。
- `app/understand/ir_map_reduce.py`
  - 本阶段不改变 map-reduce 或运行 profile 路由。

---

### 任务 1：定义 `outline_blocks` 数据模型与开发期提示字段边界

**文件：**
- 修改：`app/understand/schema.py`
- 测试：`tests/test_composition.py`

- [ ] **步骤 1：编写失败的测试**

```python
from app.understand.composition import compose_from_lecture_json
from app.understand.schema import LectureJSON


def test_composition_section_can_carry_outline_blocks_with_hidden_hints():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVoutline",
            "title": "Harness 核心流程",
            "duration": 900,
            "one_liner": "解释从文章到视频的主流程。",
            "chapters": [
                {
                    "index": 1,
                    "title": "为什么不用视频生成模型",
                    "start": 0,
                    "end": 300,
                    "summary": "本章解释为什么作者优先选择网页方案。",
                    "key_takeaways": ["网页方案更可控。"],
                },
                {
                    "index": 2,
                    "title": "四个关键步骤",
                    "start": 300,
                    "end": 900,
                    "summary": "本章拆解从文章到视频的四个关键步骤。",
                    "key_takeaways": ["先改写口播稿，再拆开发大纲。"],
                },
            ],
            "final_synthesis": "核心不是单个工具，而是整条可控流程。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    target = next(section for section in composed.body_sections if section.id.startswith("chapter-") or section.id.startswith("group-"))
    assert target.outline_blocks
    first = target.outline_blocks[0]
    assert first.ordinal
    assert first.title
    assert first.source_timestamps
    assert first.block_role
    assert first.topic_hint
    assert first.source_hint
```

- [ ] **步骤 2：运行测试验证失败**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py::test_composition_section_can_carry_outline_blocks_with_hidden_hints -v`

预期：FAIL，`BodySectionView` 不存在 `outline_blocks` 字段或其子字段不完整。

- [ ] **步骤 3：编写最少实现代码**

```python
# app/understand/schema.py
class OutlineBlockView(BaseModel):
    id: str
    ordinal: str = ""
    title: str
    lead: str = ""
    paragraphs: list[str] = Field(default_factory=list)
    source_chapter_refs: list[int] = Field(default_factory=list)
    source_timestamps: list[float] = Field(default_factory=list)

    # 开发期脚手架：允许 composition 生成时保留，但最终模板不渲染
    block_role: str = ""
    topic_hint: str = ""
    source_hint: str = ""


class BodySectionView(BaseModel):
    id: str
    title: str
    section_role: str = "concept"
    summary: str = ""
    paragraphs: list[str] = Field(default_factory=list)
    outline_blocks: list[OutlineBlockView] = Field(default_factory=list)
    supporting_visuals: list[SupportingVisualView] = Field(default_factory=list)
    source_chapter_refs: list[int] = Field(default_factory=list)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py::test_composition_section_can_carry_outline_blocks_with_hidden_hints -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add app/understand/schema.py tests/test_composition.py
git commit -m "feat(schema): add outline block view for handout sections"
```

### 任务 2：在 composition 中把 section 平铺改成目录化阅读单元

**文件：**
- 修改：`app/understand/composition.py`
- 测试：`tests/test_composition.py`

- [ ] **步骤 1：编写失败的测试**

```python
from app.understand.composition import compose_from_lecture_json
from app.understand.schema import LectureJSON


def test_composition_splits_dense_section_into_outline_blocks_without_inventing_missing_topics():
    lecture = LectureJSON.model_validate(
        {
            "bv_id": "BVdenseblocks",
            "title": "Harness 实战",
            "duration": 1500,
            "one_liner": "解释文章转视频的完整链路。",
            "chapters": [
                {
                    "index": 1,
                    "title": "视频制作思路与可控性设计",
                    "start": 0,
                    "end": 450,
                    "summary": "本章解释为什么作者放弃视频生成模型，转而使用网页方案来追求可控性。",
                    "key_takeaways": ["网页方案比视频生成模型更可控。"],
                },
                {
                    "index": 2,
                    "title": "视频制作流程的四个关键步骤",
                    "start": 450,
                    "end": 900,
                    "summary": "本章把文章转视频拆成口播稿、开发大纲、视觉演示和节奏对齐四步。",
                    "process_steps": ["改写口播稿", "拆开发大纲", "设计视觉演示", "对齐口播节奏"],
                },
            ],
            "final_synthesis": "先建立可控性原则，再展开主流程。",
        }
    )

    composed = compose_from_lecture_json(lecture)

    target = next(section for section in composed.body_sections if section.id.startswith("group-"))
    titles = [block.title for block in target.outline_blocks]

    assert len(target.outline_blocks) >= 2
    assert any("可控性" in title for title in titles)
    assert any("关键步骤" in title or "流程" in title for title in titles)
    assert all("容易混淆" not in title for title in titles)
    assert all("最容易出错" not in title for title in titles)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py::test_composition_splits_dense_section_into_outline_blocks_without_inventing_missing_topics -v`

预期：FAIL，当前 section 仍只有 `paragraphs`，没有实际 block 拆分。

- [ ] **步骤 3：编写最少实现代码**

```python
# app/understand/composition.py
def _body_section_from_group(group: list[Any], composition_profile: str) -> BodySectionView | None:
    paragraphs = _group_paragraphs(group, composition_profile)
    if not paragraphs:
        return None
    section = BodySectionView(
        id=_group_id(group),
        title=_group_title(group),
        section_role=_group_role(group, composition_profile),
        summary=_group_summary(group),
        paragraphs=paragraphs,
        source_chapter_refs=[chapter.index for chapter in group],
    )
    section.outline_blocks = _outline_blocks_for_group(section, group, composition_profile)
    return section


def _outline_blocks_for_group(
    section: BodySectionView,
    group: list[Any],
    composition_profile: str,
) -> list[OutlineBlockView]:
    blocks: list[OutlineBlockView] = []
    for offset, chapter in enumerate(group, start=1):
        title = _outline_block_title(chapter, composition_profile)
        paragraphs = _outline_block_paragraphs(chapter, composition_profile)
        if not title or not paragraphs:
            continue
        blocks.append(
            OutlineBlockView(
                id=f"{section.id}-block-{offset}",
                ordinal="",  # 下一任务统一补编号
                title=title,
                paragraphs=paragraphs,
                source_chapter_refs=[chapter.index],
                source_timestamps=[float(chapter.start), float(chapter.end)],
                block_role=_outline_block_role(chapter, composition_profile),
                topic_hint=title,
                source_hint=_clean_text(chapter.title),
            )
        )
    return blocks
```

- [ ] **步骤 4：运行测试验证通过**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py::test_composition_splits_dense_section_into_outline_blocks_without_inventing_missing_topics -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add app/understand/composition.py tests/test_composition.py
git commit -m "feat(composition): split dense sections into outline blocks"
```

### 任务 3：按四类 `composition_profile` 收敛 block 标题和正文句式

**文件：**
- 修改：`app/understand/composition.py`
- 测试：`tests/test_composition.py`

- [ ] **步骤 1：编写失败的测试**

```python
from app.understand.composition import compose_from_lecture_json
from app.understand.schema import LectureJSON


def test_composition_uses_profile_aware_phrasing_without_formulaic_openers():
    conceptual = LectureJSON.model_validate(
        {
            "bv_id": "BVconceptual",
            "title": "Harness 的核心思想",
            "duration": 600,
            "one_liner": "解释 Harness 为什么有效。",
            "chapters": [
                {
                    "index": 1,
                    "title": "Harness 的核心价值",
                    "start": 0,
                    "end": 600,
                    "summary": "本章解释 Harness 的核心价值在于把模型、工具和人的判断组织成稳定流程。",
                    "key_takeaways": ["核心不是单个工具，而是整条可控流程。"],
                }
            ],
            "final_synthesis": "Harness 的重点是过程可控，而不是端到端黑盒生成。",
        }
    )

    procedural = LectureJSON.model_validate(
        {
            "bv_id": "BVprocedural",
            "title": "四个关键步骤",
            "duration": 600,
            "one_liner": "拆解文章转视频的主流程。",
            "chapters": [
                {
                    "index": 1,
                    "title": "四个关键步骤",
                    "start": 0,
                    "end": 600,
                    "summary": "本章把文章转视频拆成口播稿、开发大纲、视觉演示和节奏对齐四步。",
                    "process_steps": ["改写口播稿", "拆开发大纲", "设计视觉演示", "对齐口播节奏"],
                }
            ],
            "final_synthesis": "关键在于先把流程拆清，再做自动化。",
        }
    )

    conceptual_composed = compose_from_lecture_json(conceptual)
    procedural_composed = compose_from_lecture_json(procedural)

    conceptual_text = " ".join(
        paragraph
        for section in conceptual_composed.body_sections
        for block in section.outline_blocks
        for paragraph in block.paragraphs
    )
    procedural_titles = [
        block.title
        for section in procedural_composed.body_sections
        for block in section.outline_blocks
    ]

    assert "围绕“" not in conceptual_text
    assert "这里真正要抓的是" not in conceptual_text
    assert any("步骤" in title or "口播稿" in title or "开发大纲" in title for title in procedural_titles)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py::test_composition_uses_profile_aware_phrasing_without_formulaic_openers -v`

预期：FAIL，当前 block 标题和正文仍会继承统一模板腔或标题过于抽象。

- [ ] **步骤 3：编写最少实现代码**

```python
# app/understand/composition.py
def _outline_block_title(chapter: Any, composition_profile: str) -> str:
    title = _clean_body_text(chapter.title)
    if composition_profile == "procedural":
        return title or "本节步骤"
    if composition_profile == "argumentative":
        return title or "本节判断"
    if composition_profile == "case_based":
        return title or "本节案例"
    return title or "本节主题"


def _outline_block_paragraphs(chapter: Any, composition_profile: str) -> list[str]:
    summary = _best_body_summary(chapter)
    detail = _chapter_detail_sentence(chapter, composition_profile)
    parts = [text for text in [summary, detail] if text]
    return _split_dense_paragraphs(parts)


def _split_dense_paragraphs(parts: list[str]) -> list[str]:
    flattened = [_clean_body_text(part) for part in parts if _clean_body_text(part)]
    out: list[str] = []
    for text in flattened:
        out.extend(_paragraph_chunks(text, limit=90, max_chunks=3) or [text])
    return _dedupe_texts(out)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py::test_composition_uses_profile_aware_phrasing_without_formulaic_openers -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add app/understand/composition.py tests/test_composition.py
git commit -m "feat(composition): add profile-aware block phrasing"
```

### 任务 4：让模板渲染目录层级、时间锚点，并隐藏开发提示字段

**文件：**
- 修改：`app/render/renderer.py`
- 修改：`app/render/templates/lecture.html.j2`
- 修改：`app/static/lecture.css`
- 测试：`tests/test_smoke.py`

- [ ] **步骤 1：编写失败的测试**

```python
from app.render.renderer import Renderer
from app.understand.schema import CompositionView


def test_render_composition_blocks_show_outline_and_hide_internal_hints():
    lec = _sample_lecture()
    lec.composition = CompositionView(
        summary_mode="handout",
        body_sections=[
            {
                "id": "section-2",
                "title": "为什么网页方案更可控",
                "section_role": "reason",
                "summary": "先解释核心判断，再展开原因。",
                "outline_blocks": [
                    {
                        "id": "section-2-block-1",
                        "ordinal": "2.1",
                        "title": "先把可控性原则说清",
                        "lead": "这里先立判断。",
                        "paragraphs": ["作者先把“可控性”确立为首要原则。"],
                        "source_chapter_refs": [2],
                        "source_timestamps": [120.0, 240.0],
                        "block_role": "reason",
                        "topic_hint": "可控性原则",
                        "source_hint": "视频制作思路与可控性设计",
                    }
                ],
            }
        ],
    )

    html_path = Renderer().render_lecture(lec, css_inline="")
    html = html_path.read_text(encoding="utf-8")

    assert "2.1" in html
    assert "先把可控性原则说清" in html
    assert "02:00" in html or "2:00" in html
    assert "topic_hint" not in html
    assert "source_hint" not in html
    assert "block_role" not in html
```

- [ ] **步骤 2：运行测试验证失败**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_smoke.py -q -k "render_composition_blocks_show_outline_and_hide_internal_hints"`

预期：FAIL，当前模板尚未渲染 `outline_blocks`，或直接透出内部字段。

- [ ] **步骤 3：编写最少实现代码**

```python
# app/render/renderer.py
def _composition_body_sections(lecture: LectureJSON) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    for sec_index, section in enumerate(lecture.composition.body_sections, start=1):
        payload = section.model_dump(mode="json")
        visible_blocks = []
        for block_index, block in enumerate(payload.get("outline_blocks") or [], start=1):
            visible_blocks.append(
                {
                    "id": block["id"],
                    "ordinal": block.get("ordinal") or f"{sec_index}.{block_index}",
                    "title": block["title"],
                    "lead": block.get("lead", ""),
                    "paragraphs": block.get("paragraphs") or [],
                    "source_timestamps": block.get("source_timestamps") or [],
                }
            )
        payload["outline_blocks"] = visible_blocks
        sections.append(payload)
    return sections
```

```jinja2
{# app/render/templates/lecture.html.j2 #}
{% for sec in composition_body_sections %}
<article class="body-section body-section-{{ sec.section_role }}" id="{{ sec.id }}">
  <header class="body-section-head">
    <div>
      <span class="section-ordinal">{{ loop.index }}</span>
      <h2>{{ sec.title | rich_text }}</h2>
    </div>
  </header>
  {% for block in sec.outline_blocks %}
  <section class="outline-block" id="{{ block.id }}">
    <div class="outline-block-head">
      <div>
        <span class="outline-ordinal">{{ block.ordinal }}</span>
        <h3>{{ block.title | rich_text }}</h3>
      </div>
      {% if block.source_timestamps %}
      <a class="outline-anchor" href="{{ bili_link(lecture.bv_id, block.source_timestamps[0]) }}" target="_blank" rel="noopener">
        {{ block.source_timestamps[0] | fmt_ts }}
      </a>
      {% endif %}
    </div>
    {% if block.lead %}<p class="outline-lead">{{ block.lead | rich_text }}</p>{% endif %}
    {% for paragraph in block.paragraphs %}
    <div class="rich-paragraph">{{ paragraph | rich_text }}</div>
    {% endfor %}
  </section>
  {% endfor %}
</article>
{% endfor %}
```

```css
/* app/static/lecture.css */
.outline-block {
  margin-top: 1.25rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
}

.outline-block-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: baseline;
}

.outline-ordinal {
  display: inline-block;
  margin-right: 0.5rem;
  font-variant-numeric: tabular-nums;
  color: var(--muted-text);
}

.outline-lead {
  margin: 0.5rem 0 0.75rem;
  color: var(--muted-text);
}
```

- [ ] **步骤 4：运行测试验证通过**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_smoke.py -q -k "render_composition_blocks_show_outline_and_hide_internal_hints"`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add app/render/renderer.py app/render/templates/lecture.html.j2 app/static/lecture.css tests/test_smoke.py
git commit -m "feat(render): render outline blocks with ordinals and anchors"
```

### 任务 5：用真实样本收口文案、handoff 与验证口径

**文件：**
- 修改：`docs/superpowers/handoffs/2026-05-13-lecturemind-reading-burden-composition-handoff.md`
- 测试：`tests/test_composition.py`, `tests/test_smoke.py`
- 验证样本：`data/reports/BV1ypdgBCE9B.html`

- [ ] **步骤 1：补一个真实样本导向的定向测试**

```python
def test_composition_blocks_do_not_leak_dev_hints_or_fixed_question_templates():
    lecture = _make_grouped_harness_lecture()

    composed = compose_from_lecture_json(lecture)

    visible_titles = [
        block.title
        for section in composed.body_sections
        for block in section.outline_blocks
    ]
    visible_paragraphs = [
        paragraph
        for section in composed.body_sections
        for block in section.outline_blocks
        for paragraph in block.paragraphs
    ]

    assert all("容易混淆" not in title for title in visible_titles)
    assert all("最容易出错" not in title for title in visible_titles)
    assert all("围绕“" not in paragraph for paragraph in visible_paragraphs)
    assert all("这里真正要抓的是" not in paragraph for paragraph in visible_paragraphs)
```

- [ ] **步骤 2：运行定向测试**

运行：`D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py -q -k "outline_blocks or profile_aware_phrasing or fixed_question_templates"`

预期：PASS

- [ ] **步骤 3：重渲真实样本并人工检查关键点**

运行：`D:\anaconda\envs\myagent\python.exe -m scripts.rerender_reports --only BV1ypdgBCE9B`

检查：
- section 与 block 层级编号已出现
- block 标题能一眼看出主题
- 正文不再挤成大段
- HTML 中没有 `block_role / topic_hint / source_hint`
- `keyframe` 等图注标签如仍裸露，记录到 handoff 而不是在本任务顺手扩 scope

- [ ] **步骤 4：更新 handoff**

写明：
- 本阶段已批准并落地的设计约束：
  - `light reorder`
  - `视频原有表达优先`
  - `outline_blocks`
  - `开发期保留提示、最终不渲染`
- 真实样本当前状态
- 下一步从 `app/understand/composition.py` 的 block 投影和 `lecture.html.j2` 的 block 模板开始
- 推荐验证命令

- [ ] **步骤 5：Commit**

```bash
git add tests/test_composition.py docs/superpowers/handoffs/2026-05-13-lecturemind-reading-burden-composition-handoff.md
git commit -m "docs(handoff): update next-stage composition rollout route"
```

---

## 自检

### 规格覆盖度

- `轻量改述优先`：任务 2、任务 3、任务 5
- `不预设固定子问题`：任务 2、任务 3、任务 5
- `目录化阅读单元`：任务 1、任务 2、任务 4
- `开发期保留提示、最终不渲染`：任务 1、任务 4、任务 5
- `时间锚点优先于章节硬绑定`：任务 2、任务 4
- `内容质量、可读性强、容易理解`：任务 3、任务 4、任务 5

### 占位符扫描

- 无 `TODO` / `待定` / “后续实现”
- 每个任务都给出精确文件、命令与最小代码片段
- 未使用“类似任务 N”式引用

### 类型一致性

- `summary_mode` 固定为 `skim / note / handout / segmented_handout`
- `composition_profile` 固定为 `conceptual / procedural / argumentative / case_based`
- `body_sections[].outline_blocks[]` 统一使用：
  - `id`
  - `ordinal`
  - `title`
  - `lead`
  - `paragraphs`
  - `source_chapter_refs`
  - `source_timestamps`
  - `block_role`
  - `topic_hint`
  - `source_hint`
- 最终模板只消费：
  - `id`
  - `ordinal`
  - `title`
  - `lead`
  - `paragraphs`
  - `source_timestamps`

---

计划已完成并保存到 `docs/superpowers/plans/2026-05-13-lecturemind-reading-burden-composition-implementation-plan.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

**选哪种方式？**
