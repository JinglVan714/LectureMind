# LectureMind 独立阅读讲义化编排设计（2026-05-13）

> 相关上下文：
> - 现状 handoff：`docs/superpowers/handoffs/video-summary.md`
> - 长视频收口设计：`docs/superpowers/specs/2026-05-12-lecturemind-long-video-lecture-fields-design.md`
> - 代表样本：`data/reports/BV1ypdgBCE9B.html`
> - 参考原则：全局 skill `bilibili-render-pdf` 的“教学可读性优先、图为解释服务、必要时重组叙事”

## 1. 背景

当前视频总结，尤其是已经重点收口过的长视频路径，已经解决了一批“字段缺失、重复块过多、主视觉重复、术语退化成目录”的问题，但产物仍更接近“结构化整理页”，还不是“可独立阅读的高密度讲义”。

以 `BV1ypdgBCE9B` 为例，当前页面已经有：
- `core_question`
- chapter 级 `learning_goal / summary / teaching_notes / process_steps / pitfalls / key_takeaways`
- `final_synthesis`
- `study_questions`
- glossary 与来源链接

但站在用户视角，仍存在以下问题：
- 首屏不能快速回答“这视频真正值得带走什么”
- 页面主体仍被 chapter 字段骨架主导，而非被“阅读单元”主导
- `study_questions`、工具命令、术语、引文、生成信息等内容仍会打断主阅读路径
- 样本中 `mainline=[]`，但渲染后又出现“主线推进路径”和“主线闭环=是”，削弱用户信任
- 视觉支持不足，图像更多像证据仓库，而不是正文中的教学材料
- 现有“按时长推断总结形态”的隐含假设不稳。几分钟到十几分钟的视频也可能有很高的信息密度；23 分钟左右的视频也曾出现信息量大到边界崩溃的情况。时长不能再被当成总结形态的主判据。

结论：当前系统已具备“抽素材”的能力，但还缺少一个“把素材编排成讲义”的中间层。

## 2. 目标

本设计的目标是把视频总结进一步提升为：

- `可独立阅读`
  用户不回看原视频，也能在 5-10 分钟内理解视频的核心内容、主要逻辑、关键边界和可迁移经验。
- `按阅读负担自适应总结形态`
  总结形态不再主要由时长决定，而由阅读负担决定；时长只作为次要信号参与判定。
- `高密度讲义`
  当阅读负担足够高时，产物应升级为完整、系统、可连续阅读的讲义体，而不是浅摘要或卡片流。
- `轻重组`
  原视频顺序优先；只有当原顺序明显不利于独立阅读时，才挪动局部内容。
- `图文协同`
  图不是装饰，而是正文中的教学载体；必要时插入关键帧，必要时自动生成流程图、关系图、对比卡。
- `保留可追溯性`
  讲义虽然重组，但每个重要结论仍能追溯到原视频章节、时间点和证据。

## 3. 非目标

本设计不做以下事情：

- 不重写现有 ingest / chapter planning / map-reduce / IR 生成主链
- 不把所有视频统一套成一份固定目录模板
- 不强迫所有视频都“教学化到同一种语气”
- 不在本阶段扩展所有领域专用 schema（饮食、医学、金融等仍先复用通用讲义框架）
- 不要求第一阶段就完成自动生成全部流程图、思维图；图文策略分阶段接入
- 不在本设计中推翻现有 `tiny / standard / long / epic` 运行 profile；本设计只重定义上层成品形态与讲义编排逻辑

## 4. 核心决策

### 4.1 页面分层

页面拆成两层：

1. `讲义主稿层`
- 默认展开
- 服务“独立阅读”
- 只展示经过编排的正文主线、必要插图、结论与边界

2. `来源索引层`
- 默认收起
- 服务“核对出处 / 回看视频 / 检查证据”
- 包含原视频章节映射、时间点、引文、术语索引、工具附录、图像证据等

### 4.2 编排策略

采用 `轻重组`：
- 尽量保留原视频主脉络
- 允许把明显打断理解的内容后置
- 安装命令、参数列表、工具名录、大段引文、OCR、时间点表等默认后置到来源索引层

### 4.3 结构原则

不设计固定正文模板，而设计：
- `稳定前壳`
- `自适应正文编排`
- `稳定后壳`

稳定前壳：
- 一句话结论
- 最值得带走的 3-5 点
- 适合谁 / 不适合谁
- 如果只读 3 分钟该看哪里

稳定后壳：
- 边界与风险
- 术语 / 工具附录
- 来源索引与时间点

中间正文按“讲义编排画像”动态决定顺序。

### 4.4 按阅读负担决定总结形态

不再把“视频时长区间”当成总结形态的主路由，而改成先判断 `阅读负担`。

`阅读负担` 指的是：当系统试图把一条视频总结成“用户可直接阅读的成品”时，读者需要处理的结构复杂度、信息密度、跨段关联强度和图文支撑需求。

判定时综合考虑但不限于：
- `duration_sec`
- `subtitle_count`
- `chars_per_min`
- `chapter_count_target`
- `knowledge_unit_count`
- `code / formula /术语` 密度
- `visual_evidence` 密度
- 主题切换频率与论证分叉程度
- 当前素材是否足以做到 `reader_can_understand_without_video`

其中 `duration_sec` 是次要信号，不再单独决定总结形态。

### 4.5 运行 profile 与成品形态分离

本设计明确区分两类路由：

1. `运行 profile`
- 面向 pipeline 的算力与流程路由
- 继续沿用现有 `tiny / standard / long / epic`
- 解决的是“怎么跑”

2. `summary_mode`
- 面向用户成品的形态路由
- 由阅读负担决定
- 解决的是“最后给用户什么”

允许出现：
- 底层是 `standard profile`
- 但成品是 `handout`

也允许出现：
- 底层是 `long profile`
- 但成品只需要 `note`

这样系统不会再被“23min 到底算不算 long”这种边界问题卡住。

## 5. 三层判定模型

总结成品由三层共同决定：
- `运行 profile`：底层执行策略
- `summary_mode`：成品形态
- `composition_profile`：正文组织方式

其中 `summary_mode` 是本设计新增的核心决策层。

### 5.1 summary_mode（成品形态）

先收敛为 4 类：

1. `skim`
- 面向低阅读负担视频
- 形态：结论卡 + 少量展开
- 目标：1-3 分钟快速知道讲了什么

2. `note`
- 面向中等阅读负担视频
- 形态：轻量总结
- 目标：3-5 分钟读完，保留关键逻辑，但不展开成完整讲义

3. `handout`
- 面向高阅读负担视频
- 形态：完整高密度讲义
- 目标：5-10 分钟系统读完，不依赖回看视频

4. `segmented_handout`
- 面向极高阅读负担视频
- 形态：分段讲义 / 上卷主稿 + 下卷附录 / 按模块拆分的重型讲义
- 目标：避免单页过载，同时保留独立阅读能力

默认映射原则：
- 低负担：`skim`
- 中负担：`note`
- 高负担：`handout`
- 超高负担：`segmented_handout`

这一步只定义模式，不绑定死具体分钟数。

### 5.2 视频语义画像

描述视频本身在做什么，用于检索、解释和首页聚合：

- `teaching_tutorial`
- `opinion_interview`
- `overview_intro`
- `case_review`
- `commentary_analysis`
- `mixed`

这层不直接决定正文结构。

### 5.3 讲义编排画像

描述讲义应该怎么写，直接决定正文组织方式。先收敛为 4 类：

1. `conceptual`
- 适合：科普、理论、方法论、医学/化学解释
- 优先回答：这是什么、为什么成立、和什么相关、边界是什么

2. `procedural`
- 适合：教程、实操、配置、做法、训练流程
- 优先回答：目标是什么、前提是什么、步骤怎么走、哪里会出错

3. `argumentative`
- 适合：观点视频、访谈、评论、对谈
- 优先回答：作者结论是什么、理由是什么、反方/限制是什么、最终立场是什么

4. `case_based`
- 适合：项目展示、经验复盘、案例讲解
- 优先回答：背景是什么、做了什么、结果如何、为什么会这样、哪些经验可迁移

### 5.4 判定原则

不只依赖标题关键词，应综合：
- 标题、简介信号
- 章节标题分布
- 字幕中的高频动词、连接词、问答模式
- 是否存在明显步骤链、结论-论据结构、案例复盘语气
- 画面主要是 UI / 代码 / 流程页 / 板书 / 对谈人物 / 实物演示中的哪一种

允许主画像 + 次画像，但正文只按主画像组织，避免骨架摇摆。

对 `summary_mode` 的判定还应额外参考：
- 单位时长内的信息压缩程度
- 需要多少正文图才能让读者独立理解
- 是否出现“短时长但高负担”的技术高密视频
- 是否出现“长时长但低负担”的松散访谈或轻评论视频

## 6. 阅读负担与 summary_mode 的关系

### 6.1 负担高于时长

本设计明确采用以下判断：
- `5min` 的高密技术视频可以直接进入 `handout`
- `23min` 的高密方法论视频可以进入 `handout`，甚至必要时进入 `segmented_handout`
- `50min` 的松散访谈可能只需要 `note`
- `90min` 的课程型视频才更可能稳定进入 `segmented_handout`

### 6.2 对现有 profile 的影响

现有 `tiny / standard / long / epic` 继续保留，但只承担：
- chapter planning 方式
- 单调用还是 map-reduce
- critic / reviser / cache 等底层执行策略

它们不再隐含决定最终页面一定是：
- 卡片流
- 轻总结
- 长讲义

页面成品改由 `summary_mode` 决定。

## 7. 四种讲义编排画像的默认正文顺序

### 7.1 conceptual

默认顺序：
1. 核心结论
2. 关键概念定义
3. 原理或因果关系
4. 常见误解 / 边界
5. 如果有，补案例
6. 可迁移结论

### 7.2 procedural

默认顺序：
1. 最终要完成什么
2. 前提条件 / 输入材料
3. 主流程步骤
4. 每步检查点
5. 常见错误 / 风险
6. 工具与参数补充

### 7.3 argumentative

默认顺序：
1. 作者结论
2. 主要论据
3. 论据之间的关系
4. 反方、限制、保留条件
5. 最终立场与读者该如何理解

### 7.4 case_based

默认顺序：
1. 背景与目标
2. 做了什么
3. 结果如何
4. 为什么会这样
5. 哪些经验可复用
6. 哪些条件不可照搬

### 6.5 统一后置规则

以下内容默认不进入主稿主线，而是后置到来源索引层：
- 大段逐字引文
- 密集时间点列表
- 原视频章节全量映射
- 安装命令、参数表、工具名录
- OCR 原文
- 证据截图仓库
- 对理解帮助不大的术语堆

只有在缺它就无法理解正文时，才允许前置。

## 8. 插图策略

参考 `bilibili-render-pdf`，图的原则是：`图是教学载体，不是装饰`。

### 7.1 图的两大来源

1. `原视频关键帧`
- UI 操作界面
- PPT / 板书 / 白板
- 代码编辑器画面
- 实物演示
- 视频本身已经给出的流程页 / 结构图 / 表格

适用条件：
- 视频画面本身就是信息源
- 原图可读
- 读者看原图比看文字复述更快懂

2. `自动生成的讲义图`
- 概念关系图
- 流程图
- 对比卡 / 结论卡
- 时间线
- 决策树 / 论点树

适用条件：
- 原视频没有现成好图
- 原图太碎、太乱、太依赖口播
- 需要把分散内容重组为更适合阅读的图

### 7.2 优先支持的图类型

- `keyframe`
- `flowchart`
- `concept_map`
- `comparison_card`
- `timeline`

### 7.3 使用规则

- 图必须插在对应正文段落附近，不能集中堆到后面
- 一节里宁可只有 1 张真正有用的图，也不要为了“图文并茂”塞多张弱图
- 允许“重绘图 + 关键帧”并用：先用结构图建立整体，再用关键帧做证据

## 9. 新增中间层：composition

本设计不改写底层抽取链路，而是在 `IR -> lecture_json` 之间新增一个讲义编排层，暂命名为 `composition`。

职责：
- 判定 `summary_mode`
- 判定讲义编排画像
- 执行轻重组
- 生成主稿层阅读单元
- 规划正文插图位
- 生成来源索引层

### 8.1 新增字段组

#### A. 编排判定字段

- `burden_score`
- `burden_signals`
- `summary_mode`
- `semantic_profile`
- `composition_profile`
- `reorder_strength`
- `reading_goal`

第一阶段约束：
- `reorder_strength = "light"`
- `reading_goal = "standalone_dense_handout"` 仅在 `summary_mode in {"handout", "segmented_handout"}` 时强制成立

其中：
- `burden_score`：归一化后的综合阅读负担分数
- `burden_signals`：用于解释路由结果的主要特征快照，至少保留时长、字幕量、单位时长字符数、章节目标数、知识单元数、代码/公式密度、视觉证据密度
- `summary_mode`：`skim / note / handout / segmented_handout`

#### B. 主稿层字段

- `hero_summary`
- `key_takeaways_top`
- `audience_fit`
- `body_sections`

说明：
- `skim` 可只使用 `hero_summary + key_takeaways_top + 1-2 个 body_sections`
- `note` 使用轻量 `body_sections`
- `handout` 使用完整 `body_sections`
- `segmented_handout` 允许把 `body_sections` 分为主稿块和后置模块块

每个 `body_section` 至少包含：
- `id`
- `title`
- `section_role`
  允许值：`concept / reason / process / evidence / boundary / case / appendix`
- `summary`
- `paragraphs`
- `supporting_visuals`
- `source_chapter_refs`

#### C. 插图规划字段

- `visual_plan`
- `visual_plan.items[]`

每项至少包含：
- `visual_role`
  允许值：`keyframe / flowchart / concept_map / comparison_card / timeline`
- `placement_section_id`
- `source_mode`
  允许值：`video_frame / generated_diagram / hybrid`
- `must_have`
- `source_refs`

#### D. 来源索引层字段

- `source_index`
- `chapter_map`
- `timestamp_map`
- `evidence_quotes`
- `term_index`
- `tool_appendix`

## 10. 与现有 IR / renderer 的边界

### 10.1 IR 层

继续负责抽素材，不负责讲义成文。保留并复用：
- `chapters`
- `knowledge_units`
- `visual_evidence`
- `code_blocks`
- `formula_blocks`
- `core_question`
- `final_synthesis`
- `taxonomy`

### 10.2 composition 层

新增，用于把现有素材排成“主稿层 + 索引层”，并负责从阅读负担推导 `summary_mode`。

### 10.3 template 层

只负责渲染，不再自己拼阅读逻辑。  
模板的正文主体不应再直接按 `summary / teaching_notes / process_steps / pitfalls / key_takeaways` 顺序展开，而应优先消费 `body_sections`。

## 11. 对 `BV1ypdgBCE9B` 的期望重组

当前主线被安装细节和参数信息过早打散。按本设计，主稿层更合理的顺序应接近：

1. `Harness 的核心结论`
2. `为什么选网页方案而不是视频生成模型`
3. `文章转视频的关键流程`
4. `四阶段工作流与人工检查点`
5. `Harness 的五个设计维度`
6. `哪些结论可迁移到别的 Agent 生产任务`

以下内容后置：
- Claude Code / MiniMax / CC Switch / CLI 安装细节
- 命令验证
- 三种录制模式参数
- 大段引文与 OCR
- 原始章节导航和时间点

插图上，正文至少应具备：
- `网页方案 vs 视频模型` 对比卡
- `文章 -> 口播稿 -> 开发大纲 -> 视觉演示 -> 节奏对齐` 流程图
- `四阶段 + 两个人工检查点` 流程图
- `Harness 五个设计维度` 结构图
- 若干关键帧：网页效果、outline、checkpoint、录制模式

该样本还说明：
- 23 分钟并不天然属于“轻总结”区间
- 当字幕密度、步骤密度和方法论负担都偏高时，应允许 `standard/long` 运行 profile 输出 `handout`，必要时进一步拆成 `segmented_handout`

## 12. 验收标准

### 12.1 独立阅读能力

用户不看视频，读完 5-10 分钟后，至少能回答：
- 这视频核心讲了什么
- 作者的主要结论 / 方法是什么
- 为什么这么做
- 有哪些边界、限制或误区
- 哪些内容值得迁移复用

### 12.2 阅读阻力下降

主稿默认阅读路径不再被以下内容频繁打断：
- 原视频章节目录
- 大段引文
- 工具命令
- OCR
- review / study 问题
- 生成信息

### 12.3 图文解释能力提升

验收不看“图是否变多”，而看：
- 该有流程图的地方有流程图
- 该有概念图的地方有概念图
- 该保留关键帧的地方有关键帧
- 图和对应段落贴在一起

### 12.4 轻重组而不失真

- 每个主稿 section 都能映射回原 chapter / ts / evidence
- 不伪造原视频没有说过的逻辑闭环
- 不为了讲义化牺牲可追溯性

### 12.5 阅读负担路由正确性

至少满足：
- 不再出现“短时长但高密内容被硬压成轻总结”的系统性误判
- 不再出现“长时长但松散内容被硬做成重讲义”的系统性误判
- `summary_mode` 的判定理由可回溯到 `burden_signals`

## 13. 落地顺序

1. `定义 composition 中间层`
- 先定字段和边界
- 不动现有渲染

2. `先做纯文字主稿重组`
- 先验证 `body_sections` 是否能把 chapter dump 变成阅读单元
- 第一阶段不依赖自动画图

3. `再接插图策略`
- 先接正文关键帧
- 再接自动生成的流程图 / 概念图 / 对比卡

4. `最后收口来源索引层`
- 原视频导航、引文、术语、工具、证据、时间点统一后置

## 14. 风险与约束

- 分类判定过度依赖关键词，可能导致正文骨架误选
- 阅读负担估计如果过于粗糙，会让 `summary_mode` 再次退化成“时长换皮”
- “轻重组”边界不清时，容易重新回到固定模板或过度重写两个极端
- 自动生成图如果没有明确 placement / source_refs，容易退化成装饰图
- 如果继续让模板自行拼正文逻辑，新中间层会被架空

约束：
- `summary_mode` 必须先于主稿结构判定
- 第一阶段必须坚持 `light reorder`
- 正文骨架必须由 `composition_profile` 驱动，而不是由历史 template 顺序继承
- 索引层与主稿层必须彻底区分默认展开策略

## 15. 不在本设计范围

- 所有领域专用 schema 的一次性扩展
- 图生成功能的具体模型、提示词、缓存策略
- 首页推荐、搜索、排序如何消费新画像字段
- Copilot 面板如何读取新的 composition 数据

短视频、中视频不再被排除在设计之外；它们只是可能落到不同的 `summary_mode`。

## 16. 设计自检

- [x] 占位符扫描：无 TODO / 待定字段
- [x] 内部一致性：目标、阅读负担路由、`summary_mode`、`composition` 中间层、插图策略和验收标准方向一致
- [x] 范围检查：聚焦“按阅读负担自适应总结形态”的讲义化编排，不扩到全领域 schema 重构
- [x] 模糊性检查：正文固定模板被明确否定；采用“稳定前后壳 + 自适应正文编排”；重组强度明确为 `light`；时长被降为次要信号
