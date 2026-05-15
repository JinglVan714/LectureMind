# LectureMind

面向长视频学习场景的“视频总结 + Copilot”系统。

LectureMind 不是把视频粗暴压缩成几条摘要，而是把一条 B 站课程或技术讲座链接，转换成一份可阅读、可追溯、可对话的学习讲义：你会得到一份独立 HTML 讲义、带证据锚点的内容结构，以及一个围绕讲义工作的 Copilot。

它适合这样的场景：

- 你想把长视频变成更适合复习、检索和二次学习的文字材料。
- 你不满足于“总结一下这视频讲了什么”，而是希望答案能回到原视频的时间点、关键帧和原文证据。
- 你希望把处理后的内容继续开放给外部 Agent，例如 Cursor、Claude Desktop 或其他支持 MCP 的客户端。

## 立意

大多数“视频总结”产品只解决了压缩问题，没有解决学习问题。

LectureMind 的目标不是生成一份好看的摘要，而是构建一个更适合学习和继续提问的知识表面：

- 对读者来说，核心产物是一份结构清楚的讲义。
- 对系统来说，核心产物是一套能追溯回原视频的证据索引。
- 对 Copilot 来说，回答应当优先基于讲义，再回到证据，而不是直接围绕 HTML 或零散片段胡乱检索。

这个项目的几个核心原则已经写进代码里：

- `Lecture Note IR` 是面向读者的事实来源。
- `Evidence Index` 是面向系统的事实来源。
- Copilot 采用 `note-first / evidence-second`。
- HTML 只是投影结果，不是源数据。

## 这个项目能做什么

### 1. 把 B 站视频变成结构化讲义

输入一个 B 站视频链接或 `BV` 号，系统会自动完成：

- 抓取视频元数据
- 抽取字幕
- 抽取关键帧和封面
- 用视觉模型描述关键帧内容
- 生成结构化讲义
- 渲染为可直接打开的独立 HTML

### 2. 保留证据链，而不是只给结论

生成结果不仅有总结文本，还会尽量保留这些证据入口：

- 时间戳
- 关键帧
- 字幕引用
- 章节边界
- 术语与知识单元

这让讲义不只是“读完即弃”的摘要，而是可以继续回溯、校验和提问的学习材料。

### 3. 提供讲义内 Copilot

项目内置了一个围绕讲义工作的 Copilot，支持：

- 基于讲义问答
- 基于证据补充说明
- 在答案中保留锚点与上下文
- 通过 SSE 流式返回结果

### 4. 对外暴露 MCP 服务

除了网页里的 Copilot，LectureMind 还提供独立的 MCP Server，让外部 Agent 能访问同一套讲义与证据数据。

默认提供的是只读知识访问能力，例如：

- 搜索讲义
- 搜索证据
- 读取章节
- 读取知识单元
- 读取关键帧说明

## 大致架构

项目当前主链路可以概括成：

```text
Raw Material Layer
  -> Lecture Compiler / IR Builder
  -> Lecture Note IR
  -> Evidence Index
  -> Consumers (HTML / Copilot / MCP)
```

展开一点看，大概是这样：

### 1. 原始材料层

对应 `app/ingest/`，负责把视频变成后续可处理的原料：

- `bilibili.py`：解析 B 站链接、抓取元数据
- `subtitle.py`：抽取字幕，必要时走 Whisper 回退
- `keyframe.py`：抽取关键帧
- `cover.py`：抓取封面
- `cookies.py`：处理 B 站抓取时可能需要的 cookie

### 2. 理解与编排层

对应 `app/pipeline.py` 和 `app/understand/`，负责把“原始材料”变成“讲义结构”。

这一层不是一次性硬拼 prompt，而是包含一组明确的处理阶段：

- `FrameDescriber`：给关键帧做视觉描述和 OCR
- `ChapterPlanner`：推断更合理的章节边界
- `Length-aware routing`：按视频时长选择不同生成策略
- `LectureIRBuilder`：构建讲义中间表示 `LectureIR`
- `Critic / Reviser`：对结构化结果做审查与修补

这里还有一个很重要的现实设计：模型是分栈使用的。

- 文本生成与 Copilot：默认走 DeepSeek 兼容接口
- 视觉理解与 Embedding：默认走 DashScope / Qwen

这样做的目的很直接：在成本、速度和视觉质量之间取一个更实用的平衡。

### 3. 读者视角与系统视角的双投影

生成出的中间结果不会直接等同于最终 HTML，而是继续被整理成两个方向：

- `Lecture Note IR`
  - 面向读者
  - 强调教学单元、阅读顺序、核心结论和讲义组织
- `Evidence Index`
  - 面向系统
  - 强调证据对象、锚点关系、RAG chunk 和可检索结构

这是这个项目和普通“总结网页”最不同的地方：它把“可读”和“可检索”拆开建模了。

### 4. 消费层

最终有 3 类主要消费方式：

- `app/render/`
  - 把结构化讲义渲染成独立 HTML
- `app/copilot/`
  - 提供讲义内 Copilot、RAG 检索和 SSE 接口
- `app/copilot/mcp_server.py`
  - 对外提供 MCP Server

### 5. 存储层

对应 `app/storage/`，默认使用本地 SQLite：

- `summaries`：讲义主记录
- `assets`：字幕、关键帧等资源
- `jobs`：异步任务状态
- `lecture_chunks` + `FTS`：Copilot 检索所需 chunk

运行期数据默认都落在本地 `data/` 下，便于自托管和迁移。

## 项目目录

```text
app/
  api.py              Web API
  main.py             FastAPI 入口
  pipeline.py         端到端处理主链路
  ingest/             视频抓取、字幕、关键帧、封面
  understand/         讲义理解、IR 构建、章节规划、视觉理解
  render/             HTML 渲染
  copilot/            Copilot、RAG、MCP Server
  storage/            SQLite 持久化
scripts/
  dev_up.ps1/.sh      本地一键启动
  run_pipeline.py     单条视频命令行处理
  run_mcp_server.py   MCP Server 启动
  rerender_reports.py 重新渲染历史讲义
tests/                单测、烟测、集成验证
docs/                 部署与设计文档
data/                 运行期数据目录
```

## 快速开始

### 环境要求

- Python `3.10` 或 `3.11`
- `ffmpeg`
- 默认模型栈所需的 API Key
  - `DEEPSEEK_API_KEY`
  - `DASHSCOPE_API_KEY`
- 可选：Docker `24+`
- 可选：Node `18+`（主要用于前端语法检查）

### 1. 准备配置

先复制环境变量模板：

```powershell
copy .env.example .env
```

或：

```bash
cp .env.example .env
```

至少建议填写这些项：

```env
DEEPSEEK_API_KEY=...
DASHSCOPE_API_KEY=...
BASIC_AUTH_USER=admin
BASIC_AUTH_PASSWORD=change-me
```

常见可选项：

- `BILIBILI_COOKIE_FILE`
  - 当公开抓取不稳定、需要登录态或需要规避 `403` 时使用
- `MCP_SERVER_TOKEN`
  - 只在 MCP 的 `SSE` 模式下需要
- `DATA_DIR`
  - 自定义运行数据目录

### 2. 本地启动

Windows PowerShell：

```powershell
.\scripts\dev_up.ps1
```

Linux / macOS / WSL：

```bash
./scripts/dev_up.sh
```

如果你想手动启动，也可以这样：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[whisper]"
python -m scripts.init_db
python -m app.main
```

启动后访问：

```text
http://127.0.0.1:8000/
```

首页受 HTTP Basic Auth 保护，使用 `.env` 中的 `BASIC_AUTH_USER` 和 `BASIC_AUTH_PASSWORD` 登录。

### 3. Docker 启动

```bash
cp .env.example .env
docker compose up -d --build
```

默认会把本地 `./data` 挂进容器内的 `/app/data`，所以讲义、数据库和缓存都会持久化。

健康检查接口：

```text
GET /healthz
```

## 怎么用

### Web 界面

最常见的使用方式就是网页入口：

1. 打开首页。
2. 提交一个 B 站视频链接或 `BV` 号。
3. 等待后台任务完成。
4. 打开生成的讲义 HTML。
5. 在讲义页面内继续用 Copilot 追问。

任务接口的核心入口大致是：

- `POST /api/summarize`
- `GET /api/jobs/{job_id}`
- `GET /api/summaries`

### 命令行单独跑一条视频

如果你不想经过网页，可以直接走 CLI：

```bash
python -m scripts.run_pipeline https://www.bilibili.com/video/BVxxxxxxxxxx
```

强制忽略缓存：

```bash
python -m scripts.run_pipeline https://www.bilibili.com/video/BVxxxxxxxxxx --force
```

### 启动 MCP Server

本地 `stdio` 模式：

```bash
python -m scripts.run_mcp_server
```

远程 `SSE` 模式：

```bash
python -m scripts.run_mcp_server --sse --host 0.0.0.0 --port 8111
```

如果使用 `SSE` 模式，请先设置：

```env
MCP_SERVER_TOKEN=your-token
```

默认 MCP 主要暴露只读检索能力；`summarize_video` 默认不会公开，除非显式打开 `MCP_EXPOSE_SUMMARIZE=true`。

## 输出与数据

项目运行后，核心数据通常会出现在 `data/` 下：

- `data/app.db`
  - SQLite 主数据库
- `data/reports/`
  - 生成后的讲义 HTML
- `data/keyframes/`
  - 抽取出的关键帧
- `data/subtitles/`
  - 字幕结果
- `data/audio/`
  - 中间音频文件
- `data/debug/`
  - 调试用 `LectureIR` 导出

这意味着你可以把它理解成一个本地知识仓库，而不只是一次性的“生成页面”。

## 常用维护命令

初始化数据库：

```bash
python -m scripts.init_db
```

重新渲染已有讲义：

```bash
python -m scripts.rerender_reports
```

启动 MCP Server：

```bash
python -m scripts.run_mcp_server
```

做一轮轻量验证：

```bash
python -m compileall app tests scripts
pytest tests/test_smoke.py -v
pytest tests/test_mcp_server.py -v
```

如果前端资源有改动，还可以补一条：

```bash
node --check app/static/copilot.js
```

## 当前边界

在当前代码形态下，README 里最值得提前说明的边界有 3 个：

- 主要输入源是 B 站视频，不是一个通用视频平台聚合器。
- 认证目前以 HTTP Basic Auth 为主，更适合内网、自托管或反向代理之后使用。
- Copilot 与 MCP 的效果依赖本地已处理好的讲义与索引，它不是“拿到任意 URL 就即时联网深度研究”的系统。

## 相关文档

- [README_develop.md](./README_develop.md)：开发补充说明
- [docs/deploy.md](./docs/deploy.md)：部署说明

## License

[MIT](./LICENSE)
