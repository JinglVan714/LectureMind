# Copilot Agent 与 SSE 接口

## 文档定位

`Internship/03` 把 Copilot 描述为「note-first / evidence-second 检索式 Agent」，本篇展开 4 个模块：

- `app/copilot/api.py`（630 行）— FastAPI 路由 + SSE 流。
- `app/copilot/agent.py`（922 行）— LangGraph ReAct 图、锚点校验。
- `app/copilot/tools.py`（1679 行）— 11+ 个 Copilot/MCP 共享工具。
- `app/copilot/sections.py` + `taxonomy.py` + `prompts.py` — 答案结构化 + 归类规范化。

## 一句话定位

Copilot 是一个**LangGraph ReAct 图 + SSE 流 + 锚点校验**的垂直 Agent 应用。它**不**直接回答自由问题，而是先调讲义工具拿证据，再按强制段标签输出。整个回答过程对前端可见（token / tool_call / tool_result / done / error）。

## 接口形态

`@d:\Diet_Agent_NEW\app\copilot\api.py:51` 定义 router：

```python
router = APIRouter(prefix="/api/copilot", dependencies=[Depends(require_auth)])
```

4 个 endpoint：

- `POST /api/copilot/ask` — 流式问答，`text/event-stream`。
- `GET /api/copilot/lectures` — 讲义元数据列表（用于前端折叠树）。
- `POST /api/copilot/reindex` — 触发 RAG 索引重建。
- `POST /api/copilot/taxonomy` — 手动修正归类。

全部 Basic Auth 保护。`reindex` 长任务走 `BackgroundTasks`。

## 并发控制

`@d:\Diet_Agent_NEW\app\copilot\api.py:140-160`：

- **全局信号量** — `asyncio.Semaphore(settings.copilot_max_concurrent)`，跨整个进程。
- **按 BV 锁** — `state.copilot_bv_locks: dict[str, asyncio.Lock]`，懒创建，每 BV 一把锁。

`/ask` 协程内部 `async with sem: async with bv_lock:`，保证：

- 全局总并发不超过 N。
- 同一讲义同时只跑一个 `/ask`（防止同一 BV 反复消耗 embedding 配额）。

## /ask SSE 协议

### 5 种事件类型

`@d:\Diet_Agent_NEW\app\copilot\api.py:360-428` 的 `run_agent_sse` 是 SSE 主循环。它从 LangGraph 的 `astream_events(state, version="v2")` 里 yield 5 种事件：

| 事件 | 何时发 | data 形态 |
|------|--------|-----------|
| `token` | `on_chat_model_stream` 有文本 | `{"delta": "..."}` |
| `tool_call` | `on_tool_start` | `{"name": ..., "args": {...}}` |
| `tool_result` | `on_tool_end` | `_summarise_tool_output(name, output)` 压缩后的摘要 |
| `done` | 流结束（无异常） | `{"anchors_validated": True, "warnings": [...], "patched_answer": ...}` |
| `error` | 任意异常 | `{"code": ..., "message": ...}` |

### tool_result 故意压缩

`_summarise_tool_output(name, output)`（`@d:\Diet_Agent_NEW\app\copilot\api.py:264-333`）把工具完整返回压成精简摘要：

- `search_lecture` → `{ok, chunks_count}`
- `get_chapter` → `{ok, chapter_idx, title, primary_ref_kind, ...}`
- 出错 → `{ok: False, error}`

设计动机（docstring）：完整工具响应可能很长（一整章 + 帧 + 段落），前端只需要进度信号，不需要原始数据。完整数据通过 model 在下一轮 `messages` 里复用。

### thinking tag 剥离

`_strip_thinking_tags(text)` 删 `<think> / </think>` 标记，DeepSeek 在 thinking 模式下会输出这些。`_build_model` 在 `extra_body` 里强制把 thinking 关掉（不然 ReAct 第二轮会 400），但模型偶尔仍会自己加 thinking 标签，所以再做一层剥离。

## LangGraph ReAct 图

### 单节点 ReAct 循环

`@d:\Diet_Agent_NEW\app\copilot\agent.py:484-580` 的 `build_graph(ctx, ...)`：

```
        ┌────────────────────────────────────┐
        │            START                   │
        └──────────────┬─────────────────────┘
                       ▼
                 ┌──────────┐
                 │call_model│  ← 调 ChatOpenAI(DeepSeek)
                 └────┬─────┘
                      │
            ┌─────────┴───────────┐
   tool_calls?           !tool_calls?  + 上一条是 ToolMessage 但 content 空
   YES                    YES
   │                      │
   ▼                      ▼
 ┌──────┐         ┌────────────┐
 │tools │         │final_model │  ← 二次调模型补一个最终答案
 └──┬───┘         └─────┬──────┘
    │                   │
    └──── back to ──────┘
       _after_tools
                                       END
```

### 关键细节

- **每次请求一张新图**：`build_graph(ctx, ...)` 不缓存，让 tools 通过闭包持有当前请求的 `ctx`。这避免了「多个请求共享同一个 graph 时 ctx 错位」。
- **max_tool_calls 限制**：由 `settings.copilot_max_tool_calls` 控制。`_after_tools` 检查 `state.step_count`，超限直接 END，防止死循环。
- **final_model 补救**：当模型最后一轮没产文本只调了工具，进 `_call_final_model` 强制让它生成最终答案。

### FINAL_RESPONSE_PROMPT

`@d:\Diet_Agent_NEW\app\copilot\agent.py:68-81`：

```
请基于上面的工具结果直接输出最终答案。不要再请求工具，不要输出 <think>、</think> ...
最终答案第一行必须单独写段标签；除明显跑题时写 [[offtopic]] 外，第一行必须是 [[evidence]]，
即使只有一段也不能省略。最终答案按需分层，不要模板化凑段：
默认只输出 [[evidence]] 1 段；
只有问题确实需要前置知识、原理、应用、跨视频、联网或边界说明时，才增加最必要的补充段。
一般答案 1-2 段，复杂答案 3 段，4 段是硬上限；严禁固定输出 4 段...
[[evidence]] 只写当前讲义直接支持的内容，并优先为关键事实附上 [t=05:46] 或 [F7]；
[Ch3] 只能作为兼容补充，不应单独充当证据锚点；
回答组织应以讲义主线节点为先、以证据锚点为后，不要把章节编号当作主要结构；
...
```

这是一段精心调过的「输出格式约束」，要求每段回答以 `[[evidence]] / [[background]] / [[deep_dive]]` 等标签开头，并强制锚点优先级（time > frame > chapter）。

## 段标签：6 + 1 种

`@d:\Diet_Agent_NEW\app\copilot\sections.py`：

```python
SECTION_TYPES = (
    "evidence",      # 讲义证据
    "extension",     # 延伸理解
    "background",    # 背景补全
    "deep_dive",     # 原理深挖
    "application",   # 应用举例
    "boundary",      # 边界说明
    "offtopic",      # 问题引导（跑题时用）
)
```

`transform(raw, target)` 支持把 `[[evidence]]\n...\n[[background]]\n...` 转成 HTML / Markdown / 纯文本三种渲染目标。HTML 渲染时每段被包成 `<section class="cp-section cp-section-evidence">`，前端 CSS 给每种段不同样式。

## 锚点校验：validate_anchors

`agent.py::validate_anchors(answer, bv, db, rag)` 是 Copilot 防幻觉的最后一道关。`@d:\Diet_Agent_NEW\app\copilot\api.py:411-417` 在每个 SSE 流末尾调它：

```python
patched, warnings = await agent_mod.validate_anchors(answer, bv, db, ctx.rag)
```

### 三类锚点

`@d:\Diet_Agent_NEW\app\copilot\prompts.py::ANCHOR_PATTERNS`（**未直接读，但 agent.py 引用了这三种模式**）：

- `[t=MM:SS]` — 时间锚点。
- `[F<N>]` — 帧编号锚点。
- `[Ch<N>]` — 章节编号锚点。

校验做的事情：

- **时间锚点** — 检查 `MM:SS` 是否在 `lecture.duration` 范围内。
- **帧锚点** — 检查 `frame_id` 是否在该 BV 的真实帧列表里。
- **章节锚点** — 检查 `chapter_idx` 是否在 `lecture.chapters` 范围。

非法锚点会被替换或剥掉（patched_answer），并把警告塞进 `done` 事件的 `warnings` 字段，前端可以提示用户「答案中有 X 个无效锚点已被修复」。

## 工具集（详见 Detail/12 / Detail/10）

### Copilot 默认 5 + 4 + 1 工具

`@d:\Diet_Agent_NEW\app\copilot\agent.py:366-435` 的 `make_copilot_tools(ctx, ...)`：

**默认 5 个**（永远绑定）：

- `search_lecture` — 单 BV 向量 + FTS 混合检索。
- `get_note_unit` — 按 unit_id 读 TeachingUnit。
- `search_evidence` — 单 BV 在 EvidenceIndex 上搜证据。
- `get_evidence_object` — 按 evidence_id 读 EvidenceObject。
- `search_lectures` — 跨视频检索（用于「类似讲过的讲义在哪里」）。

**4 个 compatibility tools**（仅在问题含锚点 / 兼容关键词时绑定）：

- `get_chapter` — 按 chapter_idx 读章节。
- `get_frame` — 按 frame_id 读帧。
- `get_quote_context` — 找一句话的上下文。
- `explain_frame` — 解释一张帧的含义。

**可选 1 个**：`web_search`（`use_web=true` 时启用）。

### 工具引导描述（`_TOOL_GUIDANCE`）

`@d:\Diet_Agent_NEW\app\copilot\agent.py:121-132`：

```python
_TOOL_GUIDANCE = {
    "search_lecture": "Default first step: locate the relevant Lecture Note IR mainline before drilling down.",
    "get_note_unit": "Default second step: read the reader-source note unit ...",
    "search_evidence": "Use after the note path is clear to find Evidence Index proof for the claim.",
    "get_evidence_object": "Use after search_evidence when a specific evidence object needs exact quote/frame/source details.",
    "search_lectures": "Cross-lecture extension only; do not use for the current lecture default path.",
    ...
    "web_search": "External background only; never use as current-lecture evidence.",
}
```

这是 **note-first / evidence-second** 工作流的提示词级别约束——每个工具的 description 都明确告诉模型「这是第一步」「这是第二步」「不要作主路径用」。比起完全靠 system prompt 让模型自己决定调用顺序，工具级 hint 更稳定。

### `_should_include_compatibility_tools`

`@d:\Diet_Agent_NEW\app\copilot\agent.py:247-264`：

只有当 question / references / history 里含 `[t=...]` / `[F...]` / `[Ch...]` / `chapter N` / `frame N` 等锚点关键字时，才绑 compatibility tools。否则不给模型选择，强制走 note + evidence 工作流。

## 上下文注入：Lecture Note IR / Evidence Index 摘要

`build_initial_state(ctx, bv, question, references, history)` 内会调 `_render_note_evidence_context(lecture)`（`@d:\Diet_Agent_NEW\app\copilot\agent.py:223-235`），把当前讲义的「主线 + takeaways + 前 4 个 unit + 证据类型分布」拼成一段 system prompt 塞进 messages 开头。例：

```
当前讲义主线（Lecture Note IR，读者真源）：
- 一句话主张：xxx
- 核心要点：a；b；c
1. [teaching-unit-1] 入门：...
2. [teaching-unit-2] 实现：...
...

当前证据索引（Evidence Index，系统真源）：
- 证据对象 24 个；主要类型：chapter_evidence x6, quote_evidence x15, frame_evidence x3
- [chapter-1] -> teaching-unit-1：...
- [quote-2-1] -> teaching-unit-2-block-1：...
- ...
- 支撑关系 30 条，可继续下钻到具体证据对象。

回答策略：先按 Lecture Note IR 组织讲义主线，再用 Evidence Index / tool 调用下钻证据。
```

这是「note-first / evidence-second」在 prompt 层最直接的实现：模型一开始就看到讲义结构骨架和证据图谱概览，不需要靠工具调用先建立 mental model。

## 错误分类

`_classify_agent_error(exc)`（`@d:\Diet_Agent_NEW\app\copilot\api.py:243-261`）识别两类常见错误：

- **`quota_exhausted`** — `AllocationQuota.FreeTierOnly` / `free tier` / `quota` / `额度` 出现在异常文本里。
- **`upstream_forbidden`** — `error code: 403`。

其他归 `internal`。每类错误 emit `error` event 时带可读 message，前端能给用户具体的修复建议。

## Taxonomy：归类规范化

`@d:\Diet_Agent_NEW\app\copilot\taxonomy.py`：

```python
DOMAIN_WHITELIST = (
    "AI 技术", "编程开发", "数据科学", "硬件与系统",
    "数学", "物理", "化学生物", "医学", "烹饪", "健身运动",
    "金融投资", "人文社科", "艺术设计", "工程实务", "其他",
)
```

`normalize(taxonomy)` 做四件事：

- `domain` 必须在白名单里，否则强制 `"其他"` 并把 confidence 钳到 ≤ 0.3。
- `direction` 用 `_DIRECTION_SUFFIX_RE` 剥后缀（`教程` / `入门` / `详解` / ...），让「Rust 并发模型 教程」「Rust 并发模型 详解」都塌到「Rust 并发模型」。
- `tags` 大小写无关去重，限 5 个。
- `confidence` 钳到 `[0, 1]`。

`needs_review(taxonomy)` 在 4 种情况下返回 True：缺 taxonomy / 缺 direction / domain == "其他" / confidence < 0.4。首页就用这个判断把讲义放进「待归类」桶。

## /reindex 的两种模式

`@d:\Diet_Agent_NEW\app\copilot\api.py:520-609`：

- **`taxonomy_only=true`** — O(N) 列更新，inline 跑。遍历所有 summary，重新调 `normalize_taxonomy`，patch `domain / direction / domain_tags` 三列。
- **`all=true` / `bv=...`** — 走 `BackgroundTasks`，每 BV 重做一次 `chunk_lecture → embed → upsert`。

这是「便宜任务 inline，昂贵任务后台」的典型模式。

## 写简历可以怎么提炼

- 实现了基于 LangGraph 的 ReAct Copilot：单节点循环 + tool-bound LLM + final_model 补救节点，并通过 SSE 把 `token / tool_call / tool_result / done / error` 5 类事件实时回传前端。
- 设计了 note-first / evidence-second 的工具工作流，通过工具描述 hint + 兼容工具按需绑定 + 答案分段标签约束，把回答稳定收敛到「先讲义主线再证据下钻」的形态。
- 实现了答案锚点校验（`[t=MM:SS] / [F<N>] / [Ch<N>]`），把模型生成的非法锚点替换或剥掉，并通过 SSE `done` 事件透传 warnings 让前端提示用户。
- 设计了 6+1 种段类型（evidence / extension / background / deep_dive / application / boundary / offtopic）+ HTML / Markdown / 纯文本三种渲染目标的统一段标签机制。
- 通过全局信号量 + 按 BV 锁 + 错误分类（quota_exhausted / upstream_forbidden / internal）提升 Copilot 在生产环境下的可观察性与稳定性。

## 面试可展开点

### 1. 为什么用 LangGraph 而不是普通 ReAct 实现

LangGraph 的核心收益是「事件流可观察」——`astream_events(version="v2")` 把每一次 model 调用、工具调用都暴露成事件。这让 SSE 直接把这些事件透传给前端不需要自己维护事件 bus。

### 2. 为什么每次请求都重新 build_graph

工具通过 Python 闭包持有当前请求的 ctx（DB / RAG / 当前问题 / 当前 references）。如果共享 graph，多请求并发会让闭包变量错位。LangGraph 的图本身是无状态的，重建成本低（只是 wire up StateGraph），不是性能问题。

### 3. 为什么 compatibility tools 要按需绑定

不绑定时，模型不会知道 `get_chapter` 存在，所以一定先走 `search_lecture` / `get_note_unit` 这条主路径。这是「让 prompt 没法跑偏」的工程手段——比 system prompt 里写「不要用 get_chapter」要可靠得多。

### 4. 为什么强制段标签

如果让模型自由分段，每次回答的结构都不一样，前端没法做差异化样式。强制标签让 evidence 段醒目（绿色 + 锚点高亮）、background 段背景色变浅、boundary 段加 ⚠️ 等差异化呈现成为可能。

### 5. validate_anchors 为什么不在 prompt 里要求模型自己校验

模型自检不可靠，且会增加输出 token。`validate_anchors` 是确定性 Python 函数，O(chunk数) 时间，零幻觉风险。「凡是能确定性算的，就不交给模型」是项目反复体现的原则。

## 当前文档中的事实与推断

### 事实

- `/api/copilot/*` 4 个 endpoint 都过 Basic Auth。
- SSE 5 类事件：`token / tool_call / tool_result / done / error`。
- 默认 5 工具 + 按需 4 兼容工具 + 可选 web_search。
- 段类型 7 种，HTML / Markdown / 纯文本 3 种渲染目标。
- DOMAIN_WHITELIST 含 15 个领域，"其他" 是兜底。
- `/reindex` 两种模式：inline taxonomy_only，BackgroundTasks 全量。

### 推断

- Copilot 在这个项目里是一个「工具调用驱动的垂直 Agent」典型实现，比起自由聊天框，它的工程化深度（并发控制 / 错误分类 / 锚点校验 / 段约束）已经接近生产标准。
