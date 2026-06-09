# MCP 对外能力、部署与工程化质量保障

## 文档目的

这份文档聚焦两个方面：

- LectureMind 如何把讲义与证据能力对外暴露为 MCP 服务。
- 整个项目在部署、脚本化运维、测试和发布检查上具备哪些工程化能力。

对实习投递来说，这一篇的价值在于：  
它能把项目从“我做了个 AI 功能”提升为“我交付了一个可运行、可验证、可集成的系统”。

## 一句话定义

LectureMind 不仅提供 Web 页面内的 Copilot，还把讲义与证据检索能力封装成独立 MCP Server；同时通过 Docker、启动脚本、QA 脚本、发布前检查和测试体系，把项目做成了可部署、可回归的工程系统。

## MCP 这一层的定位

从 `app/copilot/mcp_server.py` 和 `scripts/run_mcp_server.py` 看，MCP 不是 Web API 的简单转发，而是一套独立的对外消费入口。

其核心特点包括：

- 独立构建 `Database` 和 `RAGStore` 上下文。
- 默认只暴露只读工具。
- `summarize_video` 默认隐藏，仅在显式开启时注册。
- 支持本地 `stdio` 和远程 `HTTP + SSE` 两种运行方式。

这说明项目不是只为自家前端做接口，而是考虑了外部 Agent 客户端的集成场景。

## MCP 暴露了什么能力

从 `app/copilot/mcp_server.py` 和 `tests/test_mcp_server.py` 看，默认暴露的工具围绕讲义与证据查询展开，例如：

- `search_lectures`
- `search_evidence`
- `get_note_unit`
- `get_evidence_object`
- `get_chapter`
- `get_frame`
- `get_quote_context`
- `explain_frame`
- `get_knowledge_units`
- `get_frame_description`
- `list_lectures`

这一组工具很有代表性，因为它们不是笼统的“聊天能力”，而是把讲义知识库拆成一组明确的、可组合的读取操作。

## 为什么 MCP 设计值得写进总结

因为它说明这个项目的产物不是只有页面。

如果只有页面，项目更像一个垂直应用。  
而有了 MCP 之后，项目就具备了“作为外部 Agent 的知识与工具后端”的能力。

这对 Agent 应用开发岗位尤其 relevant，原因在于：

- 你不仅在做大模型前台交互。
- 你还在设计可被其他 Agent 调用的工具面。
- 你考虑了 transport、权限边界、工具暴露范围和上下文初始化。

## MCP 的边界意识

MCP 这一层有两个很值得面试展开的边界设计：

### 1. 默认只读

从 `mcp_server.py` 的注释和 `run_mcp_server.py` 的启动逻辑看，项目默认只暴露读取能力。  
这是一种很实际的安全与成本边界：  
外部 Agent 可以查讲义和证据，但不会默认触发昂贵的视频总结流程。

### 2. summarize 需要显式开启

只有在显式配置或命令行参数开启时，才会注册 `summarize_video`。  
这意味着开发者清楚地区分了：

- 查询类能力；
- 高成本生成类能力。

这种显式开关思路非常适合在面试中展示你的工程意识。

## 部署与运行方式

根据 `README.md`、`docs/deploy.md`、`Dockerfile` 和 `docker-compose.yml`，项目至少支持三类运行形态：

- 本地 Python 环境直接启动。
- Docker Compose 一键部署。
- MCP Server 独立启动。

其中：

- `scripts/dev_up.ps1` / `scripts/dev_up.sh`
  负责本地开发环境一键启动。
- `scripts/init_db.py`
  初始化 SQLite schema。
- `scripts/run_pipeline.py`
  支持命令行直接处理单个视频。
- `scripts/run_mcp_server.py`
  启动 MCP 服务。
- `scripts/rerender_reports.py`
  在不重新调用 LLM 的前提下重渲染历史报告。

尤其是 `rerender_reports.py` 这一类脚本，说明系统已经把“内容生成”和“页面渲染”分离开了，这是一种很实用的工程化设计。

## 配置管理

`app/config.py` 是项目的集中配置入口。  
从中可以看到，系统把以下配置项统一放到环境变量中管理：

- 模型与 API Key；
- 并发控制；
- 数据目录；
- Bilibili cookie；
- Whisper 参数；
- 长视频 profile 阈值；
- VLM tiering 与 cache；
- Copilot 并发；
- MCP 开关与 Token；
- Basic Auth。

这说明项目不是散落式读取环境变量，而是有一个集中、类型化的配置层。  
对简历来说，这能支撑“具备可部署、可配置的服务化能力”。

## 测试与 QA 体系

从 `tests/` 目录和 `docs/qa/lecturemind_copilot_qa.md` 可以看到，这个项目并不缺测试证据。

测试覆盖至少涉及：

- `test_lecture_compiler.py`
  讲义编译与双投影。
- `test_copilot.py`
  taxonomy、RAG chunking、检索等。
- `test_mcp_server.py`
  MCP 工具与服务面。
- 其他 `test_ir_*`、`test_chapter_*`、`test_vlm_*`
  覆盖 IR、章节规划、VLM tiering、缓存等。

此外，仓库里还提供了：

- `scripts/qa_full.ps1` / `.sh`
- `scripts/preflight.ps1`
- `docs/release_checklist.md`
- `docs/qa/lecturemind_copilot_qa.md`

这说明项目不只是“有单元测试”，而是已经形成了：

- 自动化回归；
- 发布前检查；
- 手工 QA checklist；
- 真实 canary / smoke 记录。

## 为什么工程化部分很重要

很多 AI 项目总结容易只写：

- 用了什么模型；
- 做了什么效果；
- 写了什么页面。

但真正能让项目显得成熟的，往往是工程化部分：

- 能否反复部署；
- 能否快速启动；
- 能否重跑或重渲染；
- 能否在改动后验证不回归；
- 能否安全地对外暴露能力。

LectureMind 这一点相对完整，因此值得单独成篇。

## 适合写进简历的亮点

- 将讲义与证据检索能力封装为独立 MCP Server，支持 `stdio` 与 `HTTP + SSE` 两种接入方式，便于外部 Agent 复用。
- 构建了本地启动、Docker 部署、MCP 独立启动、报告重渲染、预检与 QA 脚本，提升系统可部署性与可维护性。
- 通过 SQLite schema、集中配置、自动化测试、发布清单和 smoke / canary 记录，形成从开发到交付的基本工程闭环。

## 适合面试展开的问题

### 1. 为什么要做 MCP

因为项目的核心产物不是页面，而是讲义知识与证据能力。  
把这些能力通过 MCP 暴露出来，可以被其他 Agent 客户端复用，扩大系统的消费面。

### 2. 为什么 summarize 默认不对外暴露

因为它和只读检索完全不是一类操作：

- 成本更高；
- 依赖更多；
- 风险更大；
- 对资源占用更重。

默认关闭是更稳妥的工程选择。

### 3. 你如何保证项目改动后不容易坏

可以从这些方面回答：

- 有单元和集成测试；
- 有 smoke 和 QA 脚本；
- 有发布前 checklist；
- 有脚本化重跑与重渲染能力。

### 4. 这个项目有什么“像后端系统”的地方

可以答：

- 有配置层；
- 有持久化层；
- 有任务状态与作业跟踪；
- 有服务接口；
- 有部署形态；
- 有测试和发布流程。

## 当前文档中的事实与推断

### 事实

- 项目提供独立 MCP Server。
- MCP 支持 `stdio` 和 `HTTP + SSE`。
- summarize 工具默认隐藏，需要显式开启。
- 仓库包含 Docker、Compose、开发启动、QA 和 preflight 脚本。
- 仓库存在多类测试和 QA / 发布文档。

### 推断

- 项目已经具备较完整的“面向外部 Agent 的知识服务后端”形态。

## 在完整主线中的位置

这一篇对应主线中的第四段：  
**可追溯证据 -> Copilot / MCP 消费 + 工程化交付**

至此，4 篇主线文档已经覆盖：

1. Bilibili 视频到结构化原材料。
2. 结构化讲义生成与双投影建模。
3. 证据索引、RAG 与 Copilot 消费。
4. MCP 对外能力、部署与工程化质量保障。
