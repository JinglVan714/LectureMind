# LectureMind

把一条 B 站视频链接变成一份**可阅读、可追溯、可对话**的结构化讲义，配合 Copilot 围绕讲义做问答，并通过 MCP 协议暴露给外部 Agent。

## 它解决什么问题

长视频的审阅/学习有一个根本矛盾：视频是线性的、不可跳读的、不可搜索的。一个 30 分钟的技术视频，你要花 30 分钟看完才知道它讲了什么。

LectureMind 的做法是把视频的**信息结构**提取出来，变成一份独立于视频的讲义。用户看完讲义就能判断"这段视频值不值得看"、"核心知识点是什么"、"关键证据在哪"。

和市面产品的本质区别：
- 市面产品：视频 → 摘要（maybe读完即弃）
- LectureMind：视频 → 讲义（可阅读、可追溯、可对话）

## 它能做什么

### 1. 视频 → 结构化讲义

输入一个 B 站视频链接或 BV 号，Pipeline 自动完成：

```
元数据 → 字幕 + 关键帧 + 封面（并行）→ VLM 视觉理解 → 多 Agent 协作生成 IR
→ 双投影编译（Note IR + Evidence Index）→ HTML 渲染 → RAG 索引
```

### 2. 多 Agent 协作

讲义生成不是单次 LLM 调用，而是 4 个 Agent 分工协作：

| Agent | 职责 | 失败策略 |
|-------|------|----------|
| StudyQuestionAgent | 生成学习问题驱动抽取 | 失败→跳过 |
| LectureIRBuilder | 结构化抽取（单次 / Map-Reduce） | 截断→自动降级 |
| Critic | 质量审查（覆盖度/引用/代码公式） | 失败→跳过 |
| Reviser | 定向修复（patch / full 两种模式） | 失败→保留原 IR |

### 3. 长度感知路由

按视频时长自动选择处理策略：

| Profile | 时长 | IR Build | Critic | Reviser |
|---------|------|----------|--------|---------|
| tiny | < 3min | 单次调用 | off | off |
| standard | 3-15min | 单次调用 | full | off |
| long | 15-60min | Map-Reduce | projected | patch |
| epic | 60min+ | Map-Reduce | projected | patch |

### 4. 讲义内 Copilot

基于 LangGraph ReAct 的讲义问答 Agent：
- RAG 混合检索（向量 + FTS5 → RRF 融合）
- 8 种 Chunk 类型（teaching_note / quote / pitfall / knowledge_unit / frame_ocr / code_block / formula_block / study_question）
- 锚点验证（自动校验时间戳/帧号/章节号有效性）
- 回答分段标签（`[[evidence]]` / `[[background]]` / `[[deep_dive]]`）

### 5. MCP Server

通过 MCP 协议暴露 11 个只读工具给外部 Agent（Cursor / Claude Desktop / ChatGPT Apps）：
- stdio 模式（本地）/ HTTP+SSE 模式（远程）
- `summarize_video` 默认隐藏，需显式开启

## 核心架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户入口                              │
│  Web UI · API · Copilot SSE · MCP                       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Pipeline 编排层 (app/pipeline.py)           │
│  8 阶段 · 并发控制 · 进度回调 · 阶段计时 · 容错降级     │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────────┐
        │                  │                      │
┌───────▼───────┐ ┌────────▼────────┐ ┌──────────▼──────────┐
│  Ingest       │ │  Understand     │ │  Render             │
│  字幕/帧/封面  │ │  IR/Agent/VLM   │ │  HTML + KaTeX       │
└───────────────┘ └─────────────────┘ └─────────────────────┘
                           │
        ┌──────────────────┼──────────────────────┐
        │                  │                      │
┌───────▼───────┐ ┌────────▼────────┐ ┌──────────▼──────────┐
│  Storage      │ │  Copilot        │ │  Runtime Harness    │
│  SQLite       │ │  Agent/RAG/MCP  │ │  Run/Verdict/Policy │
└───────────────┘ └─────────────────┘ └─────────────────────┘
```

### 双投影设计

LectureIR 编译为两个独立投影：
- **Lecture Note IR**：面向读者（teaching_units / content_blocks / visual_slots）
- **Evidence Index**：面向系统（evidence_objects / evidence_relations / anchor_map）
- 两者通过 `note_node_ids` 双向关联
- HTML 只是投影结果，不是数据源

### Runtime Harness

每次运行都有统一的运行契约：

```
RunContract（要做什么）
  + PolicySnapshot（允许什么）
  = RunRecord（运行记录）
    ├── artifacts[]（产出物）
    ├── warnings[]（警告）
    └── verdict（accept / revise / block + 原因 + 证据引用）
```

三种 RunType：
- `lecture_compile`（S1 可写）：Pipeline 完整讲义生成
- `copilot_answer`（S0 只读）：Copilot 问答
- `mcp_tool_request`（S0 只读）：MCP 工具调用

### Workplace Harness

仓库即规范——Agent 冷启动只需读根目录入口文件：

| 文件 | 作用 |
|------|------|
| `AGENTS.md` | Agent 入口：启动顺序、工作规则、完成定义 |
| `CLAUDE.md` | Claude Code 入口：项目概述、验证命令、架构约束 |
| `ARCHITECTURE.md` | 分层依赖方向、禁止的依赖、不变式 |
| `feature_list.json` | 机器可读功能清单（id / status / verification / evidence） |
| `claude-progress.md` | 当前进度、活跃任务、阻塞 |
| `clean-state-checklist.md` | Session 收官检查清单 |
| `init.ps1` / `init.sh` | 标准化启动脚本（环境检查 + 编译检查 + 基线测试） |

## 容错设计

系统有 10+ 层容错链，设计哲学是"假设 LLM 的每一次输出都可能出问题"：

| 层级 | 容错策略 |
|------|----------|
| Ingest | httpx→curl 回退、CC→Whisper 回退、场景检测→均匀采样 |
| IR Build | 单次截断→Map-Reduce 自动降级、章节失败→placeholder |
| Agent | StudyQuestion/Critic/Reviser 各自失败→no-op |
| Reviser | Patch 白名单 + Schema 校验 + 整批回滚 + _restore_dropped_content |
| Copilot | max_tool_calls 限制、锚点无效→`[⚠ t=MM:SS]` 标记 |
| Runtime | RunRecord 一致性校验、PolicySnapshot 能力约束 |

## 技术栈

| 层 | 技术 |
|----|------|
| 文本生成 + Copilot | DeepSeek V4（OpenAI 兼容） |
| 视觉理解 | DashScope / Qwen 3.5 Omni Plus |
| Embedding | DashScope / tongyi-embedding-vision-flash |
| Agent 框架 | LangGraph（ReAct） |
| MCP | FastMCP |
| RAG | sqlite-vec（向量）+ FTS5（全文）→ RRF 融合 |
| 存储 | SQLite + aiosqlite |
| Web | FastAPI + Uvicorn |
| 渲染 | Jinja2 + KaTeX + highlight.js |
| 关键帧 | PySceneDetect + ffmpeg |
| 字幕回退 | faster-whisper |

## 快速开始

### 环境要求

- Python 3.10+
- ffmpeg
- API Key：`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`

### Agent Quick Start

如果你是进入这个仓库的 coding agent：

```bash
# 1. 读 AGENTS.md 了解工作规则
# 2. 读 claude-progress.md 了解当前状态
# 3. 读 feature_list.json 了解功能清单
# 4. 运行标准化启动
./init.sh          # Linux/macOS/WSL
# 或
.\init.ps1         # Windows
```

### 本地启动

```bash
cp .env.example .env
# 编辑 .env 填写 API Key

# 方式一：一键启动
./scripts/dev_up.sh        # Linux/macOS/WSL
.\scripts\dev_up.ps1       # Windows

# 方式二：手动
pip install -e ".[whisper]"
python -m scripts.init_db
python -m app.main
```

启动后访问 `http://127.0.0.1:8000/`（HTTP Basic Auth 保护）。

### Docker 启动

```bash
cp .env.example .env
docker compose up -d --build
```

`./data` 挂载进容器，讲义和数据库持久化。健康检查：`GET /healthz`。

### 命令行处理单条视频

```bash
python -m scripts.run_pipeline https://www.bilibili.com/video/BVxxxxxxxxxx
python -m scripts.run_pipeline https://www.bilibili.com/video/BVxxxxxxxxxx --force  # 忽略缓存
```

### 启动 MCP Server

```bash
python -m scripts.run_mcp_server                          # stdio 模式
python -m scripts.run_mcp_server --sse --port 8111        # HTTP+SSE 模式
```

## 项目目录

```
app/
  main.py              FastAPI 入口
  api.py               Web API 路由
  pipeline.py          端到端编排（8 阶段）
  config.py            配置中心（200+ 配置项）
  auth.py              HTTP Basic Auth
  ingest/              原料采集（bilibili/subtitle/keyframe/cover）
  understand/          理解层（ir/agents/vlm/compiler/evidence_index）
  render/              渲染层（HTML + KaTeX）
  copilot/             消费层（agent/rag/tools/indexer/mcp_server）
  storage/             存储层（SQLite）
  runtime/             运行时 Harness（contracts/policy/observe）
  static/              前端资源（CSS/JS/KaTeX）
scripts/               CLI 工具（25+ 脚本）
tests/                 测试套件（28 文件）
docs/                  设计文档（36+ spec/plan）
```

## 配置

所有配置通过 `.env` 加载（见 `.env.example`），核心配置域：

| 域 | 关键配置 |
|----|----------|
| 模型栈 | `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `QWEN_VL_MODEL` |
| Agent | `LECTURE_QUESTION_DRIVEN` / `LECTURE_CRITIC_ENABLED` / `LECTURE_REVISER_MODE` |
| 路由 | `LECTURE_PROFILE_THRESHOLDS_SEC` / `LECTURE_CHAPTER_MAX_DURATION_SEC` |
| 容错 | `LECTURE_IR_MAX_TOKENS` / `LECTURE_IR_AUTO_FALLBACK_TO_MAP_REDUCE` |
| VLM | `VLM_TIERING_ENABLED` / `VLM_CACHE_ENABLED` |
| Copilot | `QWEN_COPILOT_MODEL` / `COPILOT_MAX_TOOL_CALLS` |
| MCP | `MCP_SERVER_TOKEN` / `MCP_EXPOSE_SUMMARIZE` |

## 验证

```bash
# 标准化启动检查
./init.sh

# 单测
pytest tests/test_smoke.py -v

# Harness 验证
pytest tests/test_harness_workspace.py -v
pytest tests/test_runtime_harness.py -v

# Copilot 验证
pytest tests/test_copilot.py -v
pytest tests/test_mcp_server.py -v

# 编译检查
python -m compileall app tests scripts
```

## 输出数据

```
data/
  app.db               SQLite 主数据库
  reports/             讲义 HTML
  keyframes/           关键帧
  subtitles/           字幕
  audio/               中间音频
  covers/              封面
  debug/               LectureIR 导出 + timing.json + RunRecord
  chapter_cache.sqlite 章节级缓存
  vlm_cache.sqlite     帧级缓存
```

## 边界说明

- 主要输入源是 B 站视频，不是通用视频平台聚合器
- 认证以 HTTP Basic Auth 为主，适合内网/自托管
- Copilot 依赖已处理好的讲义索引，不是即时联网研究系统
- 首版 Runtime Harness 用逻辑沙盒（S0/S1），没有进程级隔离

## License

[MIT](./LICENSE)
