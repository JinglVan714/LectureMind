# 证据索引、RAG 与 Copilot 消费

## 文档目的

这份文档聚焦 LectureMind 的下游消费层：  
已经生成的讲义和证据，如何被继续用于 Copilot 问答、RAG 检索和流式回答。  
这部分最适合用于面试中的“Agent 应用”相关提问。

## 一句话定义

LectureMind 的 Copilot 不是围绕原始 HTML 做问答，而是围绕讲义与证据索引做 note-first、evidence-second 的检索式消费，并通过 SSE 将回答流式返回给前端。

## 这部分在整个项目中的作用

如果没有下游消费层，前面的 IR、讲义和证据索引最终只会变成一个静态 HTML 页面。  
而这个项目更进一步，把前面生成的结构继续用在三个方向上：

- 页面内 Copilot 问答；
- RAG 检索；
- 回答中的证据锚点与可追溯引用。

这让项目从“内容生成工具”变成了“内容生成 + 内容消费”的完整 Agent 应用。

## 关键代码模块

这一层主要分布在：

- `app/copilot/api.py`
- `app/copilot/agent.py`
- `app/copilot/rag.py`
- `app/copilot/indexer.py`
- `app/copilot/tools.py`
- `app/copilot/sections.py`
- `app/copilot/prompts.py`
- `app/copilot/diagnose.py`

入口上最重要的是 `app/copilot/api.py` 和 `app/copilot/rag.py`。

## Copilot 的接口形态

从 `app/copilot/api.py` 看，Copilot 不是普通同步接口，而是通过 SSE 流式返回。  
接口核心能力包括：

- `POST /api/copilot/ask`
  流式返回问答结果。
- `GET /api/copilot/lectures`
  返回可供前端展示的讲义元数据。
- `POST /api/copilot/reindex`
  触发索引重建。
- `POST /api/copilot/taxonomy`
  允许手动修正分类。

`/ask` 这条接口里，返回的不只是文本 token，还包含：

- `token`
- `tool_call`
- `tool_result`
- `done`
- `error`

这说明前端和后端之间并不是一次性黑盒响应，而是保留了 Agent 调用过程的可观察性。

## Copilot 的消费逻辑

从 README、`app/copilot/api.py` 和相关 QA 文档看，这个项目明确采用了：

- **note-first**
- **evidence-second**

也就是回答优先建立在讲义内容和讲义结构上，再必要时回到更细粒度证据。  
这和“直接对 HTML 文本做向量检索”有本质区别。

这样做的好处是：

- 回答更接近已经整理好的知识结构；
- 引用不会完全碎片化；
- 更容易在回答里附带章节、时间点和关键帧锚点。

## RAG 层的结构

`app/copilot/rag.py` 展示了这个项目对 RAG 的实现不是一句“接入向量数据库”就结束，而是包含三个层面：

### 1. Chunking

系统会把讲义与证据拆成多种 chunk 类型，例如：

- `teaching_note`
- `quote`
- `pitfall`
- `knowledge_unit`
- `frame_ocr`
- `code_block`
- `formula_block`
- `study_question`

这说明它的检索单元不是随便切一段文本，而是与讲义语义结构绑定。

### 2. Embedding

RAG 层会通过 DashScope 的 embedding 能力生成向量，并把 embedding 维度信息写入 `kv` 表。  
这一点很工程化，因为它考虑了模型维度探测和后续一致性校验，而不只是简单“调一次 embedding 接口”。

### 3. Hybrid Retrieval

从模块说明看，系统不是只做向量检索，还结合了：

- 向量检索；
- FTS5 全文检索；
- RRF 融合；
- 再 hydrate 回结构化结果。

这是一种典型的混合检索路线，适合应对视频讲义这类同时存在术语、引用和长文本解释的场景。

## 为什么要把证据结构继续映射成 chunk

因为讲义和证据虽然已经是结构化的，但对检索系统来说还不够细。  
要让 Copilot 能回答：

- “这一概念在哪一章讲过？”
- “这句引用具体来自哪里？”
- “哪张关键帧解释了这个机制？”

就必须把讲义块、引用、关键帧 OCR、知识单元等进一步展开成可检索单元。

因此，Evidence Index 和 RAG chunk 之间并不是重复设计，而是上下游关系：

- Evidence Index 负责建模与锚点组织；
- chunk 负责检索粒度。

## 回答锚点与防幻觉处理

`app/copilot/api.py` 中一个很关键的点是：  
系统会对答案中的锚点做校验和修补。

这意味着系统考虑到了大模型生成过程中常见的问题：

- 时间锚点越界；
- 帧编号虚构；
- 章节引用不存在。

从 `docs/qa/lecturemind_copilot_qa.md` 看，这类校验不是纸面设计，而是被放进真实 QA 和 canary 验证中的一部分。

这部分很适合在面试时强调：  
**系统不是只会生成答案，还会校验答案中的证据引用是否合法。**

## 并发控制与稳定性

`app/copilot/api.py` 还体现了两个稳定性设计：

- 全局并发信号量，控制整体 Copilot 并发。
- 按 BV 维度的锁，避免同一讲义被并发问答时过度消耗资源。

这说明该项目不是只考虑模型效果，还考虑了实际运行时的资源竞争和限流。

## 为什么这部分像 Agent 应用

如果面试岗位是 Agent 应用开发，这部分尤其重要。  
因为它已经具备了 Agent 应用的几个关键特征：

- 有结构化工具调用而不是纯文本补全。
- 有工具上下文 `ToolContext`。
- 有检索与回溯链路。
- 有流式事件返回。
- 有错误分类、诊断与回退。

虽然它不是“通用自治 Agent”，但它已经是一个围绕讲义与证据进行工具化推理和回答的垂直 Agent 应用。

## 适合写进简历的亮点

- 基于讲义与证据索引实现了视频学习 Copilot，支持 SSE 流式问答、RAG 检索与工具调用事件回传。
- 设计了与讲义结构对齐的多类型 chunk 方案，结合向量检索、SQLite FTS5 与 RRF 实现混合检索。
- 为回答中的时间点、章节、关键帧锚点增加服务端校验与修补逻辑，降低幻觉式引用带来的错误感知。

## 适合面试展开的问题

### 1. 为什么不用 HTML 直接做检索

因为 HTML 是展示结果，不是事实源。  
直接对 HTML 检索会遇到：

- 内容粒度混乱；
- 语义块边界不清晰；
- 很难回到结构化证据对象。

因此项目选择基于讲义与证据索引做消费。

### 2. 为什么要做混合检索

因为这个场景既有自然语言解释，也有术语、公式、代码和引用。  
只用向量检索或只用全文检索都容易漏掉一部分信号，所以需要混合召回再融合。

### 3. 为什么要校验锚点

因为大模型即使回答总体正确，也可能在编号和时间点上出错。  
而这个项目的卖点之一就是“可追溯”，所以引用合法性本身就是质量目标的一部分。

### 4. 这一层和前面的 Evidence Index 是什么关系

可以这样回答：

- Evidence Index 负责表达“有哪些证据、它们怎么关联”。
- RAG 和 Copilot 负责“怎么把这些证据拿出来回答问题”。

## 当前文档中的事实与推断

### 事实

- Copilot 提供 `/api/copilot/*` 接口，核心问答接口采用 SSE。
- 系统会暴露 `token`、`tool_call`、`tool_result`、`done`、`error` 等事件。
- RAG 层包含 chunking、embedding 和 hybrid retrieval。
- chunk 类型不止一种，并与讲义结构相关。
- 存在锚点校验与修补逻辑。
- 存在全局并发与按 BV 锁的控制。

### 推断

- Copilot 的定位是“证据驱动的讲义问答层”，而不是普通聊天框。

## 在完整主线中的位置

这一篇对应主线中的第三段：  
**可阅读讲义 + 可追溯证据 -> Copilot 消费**

下一篇继续总结 MCP 与对外工具化消费，以及工程化交付与测试能力。
