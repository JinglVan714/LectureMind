# LectureMind 统一 Agent Runtime Harness 设计

> 日期：2026-06-09
> 状态：draft-approved-for-review
> 设计类型：运行时架构 / agent harness / 内部编排统一层
> 适用范围：Lecture compile、Copilot ReAct、MCP ingress、运行时沙盒、记忆层、可观测性、verdict
> 相关文档：
> - `docs/superpowers/specs/2026-06-09-lecturemind-project-runtime-harness-design.md`
> - `带教文档/00-项目轮廓与架构全景.md`
> - `带教文档/04-多Agent协作-StudyQuestion-Critic-Reviser.md`
> - `带教文档/06-Copilot-LangGraph-ReAct与RAG检索.md`
> - `带教文档/08-Pipeline全链路与可观测性.md`
> - `app/pipeline.py`
> - `app/understand/agents.py`
> - `app/copilot/agent.py`
> - `app/copilot/mcp_server.py`

---

## 1. 背景

LectureMind 当前已经具备较强的 agent 能力，但这些能力分散在多个产品模块中：

- `Pipeline` 负责视频到讲义的主链编排
- `StudyQuestion / Critic / Reviser` 负责讲义生成内部的多 Agent 协作
- `Copilot` 负责围绕讲义的 ReAct 问答
- `MCP server` 负责对外暴露只读工具
- `RAGStore / Evidence Index / Lecture Note IR` 已经构成长期讲义资产底座
- `last_run_timing / agent usage / QA 文档` 已经构成可观测性和验收证据的雏形

当前真实缺口不是“缺少 agent 功能”，而是缺少一个统一的运行时外壳，把这些能力纳入同一套运行契约。具体表现为：

- `pipeline`、`copilot`、`mcp` 各自有自己的入口，但没有统一的 `Run` 语言
- 多 Agent 已存在，但只有逻辑降级，没有统一的运行时 policy、预算和沙盒边界
- 讲义资产记忆已存在，但没有正式的 runtime memory contract
- `timing`、tool loop、warning、anchor 校验、QA 证据没有统一 verdict 语言
- 新 agent 能力继续增加时，容易变成“功能堆叠”，而不是“可治理的 runtime”

本设计的目标不是重写产品业务逻辑，而是在现有产品之上新增一个统一的 Agent Runtime Harness，使 LectureMind 从“有多种 agent 能力的产品”升级为“由统一 runtime 驱动的 agent 系统”。

---

## 2. 设计目标

本设计希望达到以下结果：

1. `pipeline`、`copilot`、`mcp` 都进入同一套 `Run -> Task -> Agent -> Tool -> Verdict` 契约。
2. 首版 runtime 以内部编排优先，而不是以外部 Agent 平台优先。
3. 运行时默认采用逻辑沙盒，高风险或高代价步骤允许下沉到独立 worker 进程。
4. 首版只做两层记忆：`run memory` 与 `lecture asset memory`。
5. 任何一次运行都能拿到统一的 `run_id / trace_id / event / artifact / verdict`。
6. 运行结果不再只依赖“脚本是否抛异常”，而是有统一的 `accept / revise / block` 判断。
7. 现有 `Pipeline`、`StudyQuestion / Critic / Reviser`、`Copilot ReAct`、`MCP`、`RAG` 尽量原位复用，不推倒重写。

---

## 3. 非目标

本设计不做以下事情：

1. 不把首版 runtime 做成通用多租户 Agent 平台。
2. 不引入用户级长期学习画像或跨用户个性化记忆。
3. 不在首版引入容器级或集群级强隔离沙盒。
4. 不把所有现有任务都立刻迁移成独立进程 worker。
5. 不重写 `Pipeline` 为新的 DAG 框架。
6. 不要求首版接入完整 OpenTelemetry / Prometheus / Grafana。
7. 不引入任意代码执行、shell 执行或未声明的公网访问能力。
8. 不把 MCP 对外平台化作为首版目标。

---

## 4. 统一 Runtime 的核心对象

统一 Agent Runtime 首版以 6 个对象为最小闭环。

### 4.1 `Run`

一次完整运行的顶层容器。它可以对应：

- 一次讲义生成
- 一次 Copilot 问答
- 一次 MCP 工具请求

每个 `Run` 必须至少包含：

- `run_id`
- `trace_id`
- `run_type`
- `entrypoint`
- `target_ref`
- `policy_snapshot`
- `start_ts`
- `end_ts`
- `artifacts`
- `verdict`

### 4.2 `Task`

`Run` 中的阶段化工作单元。典型示例：

- `fetch_metadata`
- `extract_assets`
- `describe_frames`
- `build_lecture_ir`
- `critic_audit`
- `render_report`
- `persist_summary`
- `index_lecture_assets`
- `rag_lookup`
- `compose_answer`

`Task` 是调度、超时、重试、降级、worker 下沉的基本单位。

### 4.3 `Agent`

承担推理、判断或结构化生成责任的执行体。首版纳入的 Agent 包括：

- `StudyQuestionAgent`
- `CriticReviserAgent`
- `Copilot ReAct loop`

Agent 不能直接拥有系统权限。它只能在 runtime 分配的 `PolicySnapshot` 下读取 memory、调用 tool、输出结果。

### 4.4 `Tool`

可被 Agent 调用的能力接口。首版典型工具包括：

- `search_lecture`
- `get_note_unit`
- `search_evidence`
- `get_evidence_object`
- `search_lectures`
- `get_chapter`
- `get_frame`
- `get_quote_context`
- `explain_frame`
- `summarize_video`（默认隐藏）

每个 Tool 必须具有：

- 输入输出 schema
- 风险等级
- 成本标签
- 权限标签
- 可用入口范围

### 4.5 `Memory`

首版只定义两层：

- `run memory`
- `lecture asset memory`

不引入 `user long-term memory`。

### 4.6 `Verdict`

任何 `Run` 结束时都必须给出统一 verdict：

- `accept`
- `revise`
- `block`

并附带：

- `reasons`
- `warnings`
- `next_actions`
- `evidence_refs`

---

## 5. 统一 Runtime 生命周期

首版统一 runtime 生命周期如下：

1. `Ingress`
   将 API、SSE、MCP 请求标准化为 `RunRequest`
2. `Policy Bind`
   为这次运行绑定 `PolicySnapshot`
3. `Plan / Dispatch`
   根据 `run_type` 选择具体任务图
4. `Execute`
   执行 Task / Agent / Tool，写入 `run memory`
5. `Observe`
   记录统一 event、artifact、timing、warning、fallback
6. `Judge`
   基于 rubric 与运行结果生成统一 `verdict`
7. `Expose`
   向 API、SSE、MCP 返回对应结果，但底层共享同一条 runtime record

`lecture_compile`、`copilot_answer`、`mcp_tool_request` 的入口不同，但必须共享这条生命周期。

---

## 6. Runtime Kernel 与 Policy Layer

统一 runtime 拆成两层：

- `Runtime Kernel`
- `Policy Layer`

### 6.1 `Runtime Kernel`

负责运行共性，不关心具体业务 prompt 或产品页面。

建议职责：

- 创建和结束 `Run`
- 分配 `run_id / trace_id`
- 调度 task graph
- 记录 event / artifact
- 聚合成本、warning、fallback
- 产出统一 `RunResult`

### 6.2 `Policy Layer`

负责本次运行允许做什么、做到什么程度。

至少覆盖：

- `execution_mode`
- `sandbox_level`
- `allowed_tools`
- `budget`
- `memory_scope`
- `memory_writeback`
- `failure_policy`

### 6.3 设计原则

首版必须坚持以下边界：

- kernel 不知道具体 prompt 内容
- policy 不执行具体业务逻辑
- 业务逻辑继续留在 `pipeline / understand / copilot` 模块
- runtime 是统一收口层，不是业务重写层

---

## 7. 混合沙盒模型

首版采用混合沙盒，而不是全量强隔离。

### 7.1 沙盒等级

#### `S0: policy-only sandbox`

同进程执行，但受 runtime policy 约束：

- token budget
- timeout
- tool 白名单
- memory scope
- fallback policy

适用：

- `StudyQuestionAgent`
- `Critic audit`
- `Copilot ReAct loop`
- `RAG lookup`
- `MCP readonly tools`
- `validate_anchors`

#### `S1: worker-process sandbox`

由 runtime dispatch 到独立 worker 进程执行：

- 独立超时
- 独立预算
- 独立输入输出 schema
- 主进程可强制 kill

适用：

- `pipeline.run()` 顶层 compile path
- `FrameDescriber.describe_all`
- `LectureIRBuilder.build_with_agents`
- `MapReduceIRBuilder` 的 map/reduce 阶段
- `Renderer + persist + index_lecture`
- 未来的联网搜索 worker

#### `S2: forbidden in v1`

首版明令禁止：

- 任意 shell / code execution
- 未注册工具的动态调用
- 任意路径写文件
- 默认开放公网访问
- 用户级长期记忆写入

### 7.2 首版沙盒目标

混合沙盒解决的是以下现实问题：

1. 把高成本长链路与低延迟交互解耦
2. 把高风险步骤下沉到可终止的 worker
3. 为未来联网、多 Agent 扩展、代码执行工具预留边界

首版不追求企业级零信任沙盒。

---

## 8. 分层记忆设计

### 8.1 `RunMemory`

单次运行的 working memory。

保存内容包括：

- 标准化输入摘要
- `PolicySnapshot`
- 当前 task graph 状态
- 中间产物引用
- critique issues / patch proposals
- tool call records
- warning / retry / fallback
- verdict draft

设计约束：

- 生命周期仅限单次 run
- 默认不长期持久化完整工作态
- 只保留结构化摘要和 artifact 引用

### 8.2 `LectureAssetMemory`

围绕某个 lecture 的长期资产记忆。

首版直接承认现有这些就是长期记忆骨架：

- `summary_json / LectureJSON`
- `Lecture Note IR`
- `Evidence Index`
- `lecture_chunks + FTS + vector index`
- `report_path / HTML`
- `timing.json / debug artifacts`
- DB 中的 taxonomy / token_cost / status

### 8.3 记忆读写边界

首版必须明确：

1. `RunMemory` 可以读 `LectureAssetMemory`
2. `RunMemory` 不能直接改写长期资产
3. 只有明确的 commit task 才能写 `LectureAssetMemory`
4. `Copilot` 默认只读 lecture assets
5. `Critic / Reviser` 的结果先进入 `RunMemory`，只有 compile path 通过相应条件才允许写回长期资产

### 8.4 首版明确不做的记忆能力

- 不保存用户长期学习画像
- 不保存跨 lecture 的用户偏好
- 不做自动 prompt 经验回灌
- 不允许 Copilot 自动把对话写回 note / evidence
- 不做全局知识图谱式合并记忆

---

## 9. 可观测性：Trace / Event / Artifact

### 9.1 `run_id / trace_id`

每次运行必须统一生成：

- `run_id`
- `trace_id`

两者可以同值，但必须在所有入口统一使用。

附带标准字段：

- `run_type`
- `entrypoint`
- `target_ref`
- `policy_snapshot_hash`

### 9.2 统一 Event 模型

首版最小 event 类型：

- `run_started`
- `run_finished`
- `run_failed`
- `task_started`
- `task_finished`
- `task_failed`
- `agent_invoked`
- `agent_completed`
- `tool_called`
- `tool_returned`
- `worker_dispatched`
- `worker_joined`
- `fallback_triggered`
- `warning_emitted`
- `memory_committed`
- `verdict_issued`

统一字段：

- `run_id`
- `event_type`
- `ts`
- `stage`
- `component`
- `status`
- `duration_ms`
- `token_usage`
- `payload`

### 9.3 Artifact 视图

首版统一记录以下 artifact 类型：

- `lecture_ir_debug`
- `lecture_json`
- `report_html`
- `timing_record`
- `rag_index_result`
- `copilot_answer`
- `anchor_validation_report`
- `critic_report`
- `reviser_patch_set`
- `mcp_result`

每个 artifact 至少有：

- `run_id`
- `artifact_type`
- `path_or_ref`
- `producer_task`
- `size`
- `created_at`

---

## 10. 统一 Verdict 语言

### 10.1 Verdict 状态

首版严格只保留三态：

- `accept`
- `revise`
- `block`

### 10.2 Verdict 内容

每个 verdict 都必须附带：

- `status`
- `reasons`
- `warnings`
- `next_actions`
- `evidence_refs`
- `rubric_scores`（可选）

### 10.3 不同 run 类型的判定方式

#### `lecture_compile`

`accept` 条件示例：

- IR / lecture 生成成功
- HTML report 生成成功
- persist 成功
- RAG 已完成或被合法安排为后台任务
- 无关键 block 级 warning

`revise` 条件示例：

- 讲义已生成，但 Critic 给出高优先级 issues
- 存在 placeholder 章节但仍可交付人工复核

`block` 条件示例：

- IR 构建失败且无有效回退
- render / persist 失败
- 资产写回不一致

#### `copilot_answer`

`accept` 条件示例：

- 回答完成
- anchor validation 无致命问题
- 工具调用在预算内

`revise` 条件示例：

- 回答可返回，但出现弱锚点或 `[⚠...]`

`block` 条件示例：

- lecture asset 无法加载
- 关键 tool path 崩溃且无降级回答
- anchor validation 命中硬错误

#### `mcp_tool_request`

`accept` 条件示例：

- 工具返回合法 schema

`revise` 条件示例：

- 返回成功但伴随 warning 或局部缺项

`block` 条件示例：

- policy 不允许
- 目标 lecture 不存在
- tool execution 失败

### 10.4 关键原则

- 抛异常不等于一定 `block`
- 没异常不等于一定 `accept`
- verdict 来自 runtime judgment，而不是单个模块自己的“成功”

---

## 11. 与现有代码的映射

### 11.1 `app/pipeline.py`

映射为：`lecture_compile program`

保留：

- 8 阶段主链
- `select_profile`
- `plan_chapters`
- `LectureIRBuilder`
- `Renderer`
- `db.upsert_summary`
- `index_lecture`

变化：

- 顶层 run 生命周期交给 runtime kernel
- compile path 进入 `S1 worker-capable` 模型
- `last_run_timing` 不再是唯一运行记录

### 11.2 `app/understand/agents.py`

映射为：`registered lecture agents`

保留：

- `StudyQuestionAgent`
- `CriticReviserAgent`
- patch / full revise
- usage / metrics / degrade 逻辑

变化：

- 接受 `RunContext + PolicySnapshot`
- 权限来自 runtime，不来自 agent 自身

### 11.3 `app/copilot/agent.py`

映射为：`interactive runtime program`

保留：

- LangGraph ReAct loop
- tool guidance
- `build_initial_state`
- `validate_anchors`

变化：

- 每次问答先创建 `copilot_answer run`
- tool loop、warning、anchor 校验进入统一 event / artifact / verdict 语言

### 11.4 `app/copilot/mcp_server.py`

映射为：`runtime ingress adapter`

保留：

- `ToolContext`
- 独立生命周期
- `summarize_video` 默认隐藏

变化：

- MCP request 先转为 runtime `RunRequest`
- MCP 只是 ingress，不是首版的中心

### 11.5 `app/config.py`

映射为：`policy defaults source`

保留：

- 现有模型、超时、Map-Reduce、Critic、Reviser、MCP、RAG 配置

变化：

- 区分全局 settings 与本次运行的 policy snapshot

---

## 12. 首版建议新增的 Runtime 文件

建议新增以下文件，不新增更多基础设施：

- `app/runtime/contracts.py`
- `app/runtime/kernel.py`
- `app/runtime/policy.py`
- `app/runtime/memory.py`
- `app/runtime/observe.py`
- `app/runtime/worker.py`

这些文件只负责 runtime 壳层，不重写现有业务模块。

---

## 13. 首版纳入范围

首版明确纳入：

1. 统一 `RunRequest -> RunResult` 契约
2. 统一 `run_id / trace_id`
3. 统一 event / artifact / verdict 语言
4. `lecture_compile` runtime program
5. `copilot_answer` runtime program
6. `mcp_readonly_request` runtime ingress
7. 混合沙盒 `S0 / S1 / forbidden`
8. 两层记忆：`run memory + lecture asset memory`
9. 统一 `accept / revise / block`

---

## 14. 首版不纳入范围

首版明确不纳入：

1. 用户长期学习画像
2. 跨 lecture 个性化推荐记忆
3. 容器级或集群级 agent sandbox
4. 通用任务总线平台
5. 自动 prompt 自进化
6. dashboard-first 产品层
7. 任意代码执行 agent
8. MCP 外部平台化
9. 所有 task 同步迁移成 worker
10. 重写 `Pipeline` 为新框架

---

## 15. 实施顺序建议

建议按以下顺序推进：

### Phase A：统一契约

产物：

- `contracts.py`
- `policy.py`
- `Verdict` 语言

目标：

- 先让系统会“说同一种运行时语言”

### Phase B：给 Pipeline 挂 Runtime Kernel

产物：

- `lecture_compile run`
- 统一 trace / event / artifact

目标：

- 让最重要的主链先进入统一生命周期

### Phase C：给 Copilot 挂 Runtime Context

产物：

- `copilot_answer run`
- anchor validation artifact
- question-level verdict

目标：

- 把交互链也纳入统一运行时

### Phase D：引入 Worker Sandbox

产物：

- `S1 worker` 边界
- compile 主链下沉

目标：

- 让高成本步骤具备可杀死、可限时、可单独观测的进程边界

### Phase E：整理 MCP Ingress

产物：

- `mcp_readonly_request run`
- MCP ingress adapter

目标：

- 把外部协议纳入统一 runtime，但不反客为主

---

## 16. 风险与权衡

### 16.1 风险：runtime 新增过重，反而变成重构

应对：

- 首版只新增 runtime 壳层
- 不改写现有业务模块
- 不在首版引入任务总线、容器编排、通用平台

### 16.2 风险：compile path 与 copilot path 被强行统一，导致双方都变笨

应对：

- 统一的是 `Run / Policy / Event / Verdict`
- 不统一具体业务 prompt 和 task graph

### 16.3 风险：memory 设计膨胀成用户学习平台

应对：

- 首版只承认 `run memory + lecture asset memory`
- 明确禁止 user long-term memory

### 16.4 风险：worker 下沉过早，导致交互调试复杂化

应对：

- 首版优先让 compile path 下沉
- Copilot 继续保持 `S0` 低延迟交互

---

## 17. 验收标准

本规格完成并实施后，应满足：

1. `pipeline`、`copilot`、`mcp` 三条入口都能生成统一 `run_id`。
2. 任意一次运行都能输出统一的 event / artifact / verdict 记录。
3. 讲义主链可以在 `S1 worker` 模式下运行，并保留完整 trace。
4. Copilot 问答可以在 `S0` 模式下运行，并输出 anchor validation artifact。
5. `Lecture Note IR / Evidence Index / RAG / summary_json` 被正式定义为 `lecture asset memory`。
6. verdict 不再只依赖异常，而是使用统一 `accept / revise / block` 语言。
7. 新增 runtime 层后，现有业务模块不需要被推倒重写。

---

## 18. 推荐优先级

优先做：

1. `contracts.py`
2. `policy.py`
3. `run_id / trace_id`
4. `lecture_compile run`
5. `copilot_answer run`
6. `verdict + artifact registry`

后做：

1. `worker sandbox`
2. `MCP ingress runtime 化`
3. 更细粒度的观测聚合

原因是前六项能最快把现有能力从“分散模块”收口成“统一 runtime harness”，后续项则是边界强化与扩展。
