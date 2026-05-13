# LectureMind 阅读负担驱动讲义编排 Handoff

## 当前目标

把 `docs/superpowers/specs/2026-05-13-lecturemind-standalone-handout-composition-design.md`
落成可持续迭代的工程实现，并确保后续 session 能直接基于真实产物继续修正。

这份 handoff 以真实样本 `BV1ypdgBCE9B` 为主要验收对象，而不是只看单元测试是否通过。

## 已批准规格与计划

- Spec:
  - `docs/superpowers/specs/2026-05-13-lecturemind-standalone-handout-composition-design.md`
- Plan:
  - `docs/superpowers/plans/2026-05-13-lecturemind-reading-burden-composition-implementation-plan.md`

关键约束：

- 不再默认“长视频 = 重讲义”。
- 先判定 `summary_mode`，再决定 `body_sections` 的编排方式。
- 第一阶段坚持 `light reorder`，不重写整条 IR 生成链。
- handout 正文必须优先消费 `composition.body_sections`，旧 `chapters` 只作为 fallback/来源层。

## 本轮已完成

### 1. 编排层已接入主链

- 新增并接入：
  - `app/understand/composition.py`
  - `app/understand/schema.py`
  - `app/understand/ir_builder.py`
- `LectureJSON` 已具备：
  - `burden_score`
  - `burden_signals`
  - `summary_mode`
  - `composition_profile`
  - `body_sections`
  - `source_index`
  - `visual_plan`

### 2. 模板已切到 handout 主稿层

- 主要文件：
  - `app/render/renderer.py`
  - `app/render/templates/lecture.html.j2`
  - `app/static/lecture.css`
- 已实现：
  - handout overview
  - “如果只读 3 分钟”
  - handout body / source index 分层
  - `composition` 模式下隐藏 legacy 的重复区块
  - `section-ref` 标签压缩显示
  - section 内图片图注拆成元信息行 + 说明段

### 3. 真实产物驱动修复已做过两轮

针对 `data/reports/BV1ypdgBCE9B.html` 已修掉的主要问题：

- 图片重新接回来了。
- 去掉了 `目标：/带走：/证据：/边界：` 这类字段腔。
- 去掉了 `.../…` 这类正文截断痕迹。
- 章节标题不再是直接的 `A / B` 拼接。
- handout 已具备 overview / body / appendix 的基本结构。
- `section-ref` 不再机械展开为超长章节列表；现在会显示：
  - `综合全片`
  - `对应原视频前 4 章`
  - `对应原视频第 3-5 章`

### 4. 正文质量又收了一轮

本轮继续针对 `BV1ypdgBCE9B` 做了 composition 层收口：

- `可不搞 / 开外计划 / 上下门管理 / 注意注意` 已在投影层做正规化。
- 含 `语音识别错误 / 字幕中为 / 实际应为` 的编辑性修补说明不再进入 handout 正文。
- 对已经很长、信息已经够密的章节摘要，不再强行追加一条 detail 句。
- 真实样本里的 `chapter-6` 已不再突然切到 `Slidescape / Token 开销` 这类工具侧枝。

### 5. 下一阶段 spec / plan 已更新，但尚未实现

本轮已把 spec 与 plan 更新到下一阶段路线，关键变化是：

- 不再默认继续扩写 `body_sections` 平铺正文，而是升级为 `section -> outline_blocks -> paragraphs`。
- `outline_blocks` 面向用户呈现目录层级、主题标题和时间锚点，让读者一眼知道自己正在第几部分、读到哪一小块。
- 开发期允许保留 `block_role / topic_hint / source_hint` 这类内部提示字段帮助生成和验收。
- 最终 HTML 禁止渲染这些内部提示字段；用户只看到层级编号、标题、正文、时间锚点和相关图片。
- 明确坚持 `light reorder`：默认相信视频已经是创作者结构化表达观点的结果，只做轻量讲义化，不预设“误区 / 风险 / 反方 / 混淆点”块。

## 当前真实样本状态

主样本：

- Debug IR:
  - `data/debug/BV1ypdgBCE9B.lecture_ir.json`
- HTML:
  - `data/reports/BV1ypdgBCE9B.html`
- Keyframes:
  - `data/keyframes/BV1ypdgBCE9B`

当前真实 HTML 中已经确认存在的正向变化：

- overview 存在，且“如果只读 3 分钟”可用。
- `section-ref` 已压缩，不再出现 `第 1、2、3、4、5、6 章` 这种冗长标签。
- 图片图注已有 `.visual-meta`，阅读上比之前好很多。
- 主稿正文已不再出现 `目标：`、`带走：`、`证据：`、`边界：`、`...`、`…`。
- 主稿正文已不再直接泄漏 `可不搞 / 开外计划 / 上下门管理 / 注意注意 / 语音识别错误` 这类脏词或修补说明。

## 当前仍有瑕疵

这些问题已确认还在，适合交给下一个 session 继续修：

### 1. 文案仍然偏“章摘要拼接”

在真实 HTML 中，很多正文段仍然有明显的“章节摘要 + 关键结论句”腔调，例如：

- `group-1-2`
- `chapter-3`
- `chapter-4`

问题表现：

- 很多段落现在会以 `围绕“某一部分”...` 起手，虽然比之前顺一点，但仍然有明显模板腔。
- `这里真正要抓的是...` 这类 detail 句开始替代旧的字段腔，但目前仍偏机械。
- 虽然已经不再是字段腔，但仍不像自然讲义，更像“被润色后的结构化摘要”。

补充判断：

- 当前主稿还停留在 `section` 内 paragraph 平铺阶段。
- 用户已经明确要求下一阶段改成更清楚的目录化阅读单元，而不是继续在旧 paragraph 结构里修修补补。

### 2. 高频 ASR 脏词已基本收住，但术语讲义化还没做完

本轮后，真实 HTML 正文里已经不再直接出现这些脏词：

- `可不搞`
- `开外计划`
- `上下门管理`
- `注意注意`

但仍有后续工作：

- 当前只是做了高频正规化，不等于术语已经完全讲义化。
- 类似 `outline / stage3 / TTS` 这类词在不同 section 里的中文解释层级还不一致。
- 如果后续要继续收正文口径，最好把“术语正规化”和“讲义化命名”分成两层处理。

### 3. 个别 section 的 detail 句仍偏“从材料池抽一句”

例如真实样本里：

- `group-1-2`
- `group-3-4`

问题表现：

- detail 句虽然比之前更贴题，但仍常是“章节摘要之后再补一句 takeaways”。
- 某些段落的第二句和前一句语义接近，只是换了一个更像结论句的说法。

下一个 session 需要检查：

- 当前 `_best_body_summary()` / `_chapter_detail_sentence()` 是否还把 `summary + takeaway` 当成默认配对
- 是否应该让 detail 句更像“展开前一句”，而不是“再补一条结论”

### 4. 图注结构比之前好，但还没完全收口

当前图注已经拆成：

- 时间锚点
- `keyframe`
- 说明段

但仍有两个问题：

- `keyframe` 直接裸露成英文类型标签，不够讲义化。
- figcaption 的 HTML 空白和行内结构仍稍粗糙，属于“能用但不够干净”。

### 5. PowerShell 里直接 `Get-Content` 看 HTML 容易出现乱码错觉

注意：

- 文件本身是 UTF-8 没问题。
- 但在当前 Windows / PowerShell 控制台里，直接 `Get-Content` 查看中文 HTML 很容易显示成乱码。
- 用 `rg` 搜中文关键词时，结果更可信。

不要因为 PowerShell 输出乱码，就误判实际 HTML 文件编码坏了。

## 下一位执行者建议先看

按这个顺序进入：

1. `docs/superpowers/specs/2026-05-13-lecturemind-standalone-handout-composition-design.md`
2. `docs/superpowers/plans/2026-05-13-lecturemind-reading-burden-composition-implementation-plan.md`
3. `data/reports/BV1ypdgBCE9B.html`
4. `data/debug/BV1ypdgBCE9B.lecture_ir.json`
5. `app/understand/composition.py`
6. `app/render/templates/lecture.html.j2`

## 下一 session 的建议切入点

建议不要一上来继续改 CSS，也不要继续沿着旧的 `body_sections.paragraphs` 平铺结构修句子。  
下一步应先把正文结构升级成目录化阅读单元，再收表述。

优先级建议：

1. 先扩 schema 与 composition
   - 给 `body_sections` 增加 `outline_blocks`
   - 明确保留开发期提示字段，但最终不渲染
2. 再把 section 平铺改成 block 投影
   - block 标题优先提炼视频现有主题
   - block 数量按内容密度浮动，不预设固定问题模板
3. 然后改模板与 CSS
   - 前台显示 `2 / 2.1 / 2.2`
   - 显示时间锚点
   - 隐藏 `block_role / topic_hint / source_hint`
4. 最后再收正文句式
   - 降低 `围绕“...”`
   - 降低 `这里真正要抓的是...`
   - 让不同 `composition_profile` 的 block 标题和段内展开更自然

推荐起点文件：

1. `app/understand/schema.py`
2. `app/understand/composition.py`
3. `app/render/templates/lecture.html.j2`
4. `app/static/lecture.css`

## 验证命令

建议直接使用环境里的 Python，不要再走 `conda run`：

```powershell
D:\anaconda\envs\myagent\python.exe -m pytest tests/test_composition.py -v
D:\anaconda\envs\myagent\python.exe -m pytest tests/test_smoke.py -q -k "render_prefers_composition_body_sections or render_handout_overview_hides_generation_info or review_questions_hide_when_they_match_study_questions or format_section_ref_label_compacts_broad_ranges"
D:\anaconda\envs\myagent\python.exe -m scripts.rerender_reports --only BV1ypdgBCE9B
```

## 当前工作区说明

- 仓库里还有其他未提交改动，不要顺手回滚。
- 本 handoff 已经更新到“第二轮真实 HTML 收口之后”的状态。
- 当前如果要交接，可以直接基于这份 handoff 继续，不需要再手工整理一次背景。
