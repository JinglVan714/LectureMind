# LectureMind

> 把 B 站视频变成能点回原片、能吃透的交互式讲义。

[![CI](https://img.shields.io/github/actions/workflow/status/2772658778-ctrl/LectureMind/ci.yml?branch=main&label=CI)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](./pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#许可证)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1c3d5a)](https://langchain-ai.github.io/langgraph/)
[![MCP](https://img.shields.io/badge/MCP-fastmcp-blueviolet)](https://github.com/jlowin/fastmcp)

LectureMind 是一个面向「想吃透 B 站教学/科普视频但不想完整看完」场景的**自托管 Web 工具**（理论上能总结所有视频）。
丢一个 B 站链接进去，几分钟后你会得到一份自包含的单页 HTML 讲义：带封面、主线推进、术语速查、章节讲解、类型增强复盘、内嵌关键帧、可点击时间戳跳回原视频；每个论点旁边都贴了原字幕引文（反幻觉锚点）。

讲义页右下角还内置了 **Copilot 对话 Agent**：可以划词、引用章节/关键帧追问；外部 Agent（Cursor、Claude Desktop、ChatGPT Apps）通过 **MCP Server** 也能直接检索同一份知识库。

---

## ✨ 为什么不是另一个 BibiGPT

| 维度 | 摘要工具（BibiGPT / NoteGPT 类） | LectureMind |
| --- | --- | --- |
| 输出形态 | bullet 摘要 + mindmap | 学习地图 → 主线推进 → 章节讲解 → 类型复盘 → 学习闭环 |
| 反幻觉 | 文本生成结果，需用户自检 | **每个论点带 `[t=MM:SS]` / `[F7]` / `[Ch3]` 锚点**，越界自动标 ⚠️ |
| 视觉信息 | 关键帧仅作配图 | **VLM OCR + 画面类型 + 教学价值评分**，写进章节讲解 |
| 互动 | 一次性总结 | 内置 Copilot Agent，分层回答（讲义证据 / 背景补全 / 延伸 / 边界） |
| 外部集成 | 无 | **MCP Server**，Cursor / Claude Desktop 直接调用结构化检索 |
| 沉淀 | 文本笔记 | `LectureIR` + `knowledge_units` 中间层，已为 Graph RAG 跨视频综述铺路 |
| 部署 | SaaS | **本机/家庭服务器自托管**，缓存与数据 100% 在你硬盘上 |

---

## 📚 目录

- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [工作原理](#工作原理)
- [Copilot 对话 Agent](#copilot-对话-agent)
- [MCP Server](#mcp-server)
- [API 速查](#api-速查)
- [项目结构](#项目结构)
- [开发与测试](#开发与测试)
- [部署](#部署)
- [路线图](#路线图)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 核心特性

- **讲义式单页 HTML** — 自包含、关键帧 base64 内嵌，离线打开也能看；本地 KaTeX 0.16.11 + mhchem 渲染公式与化学式。
- **`LectureIR` 中间层** — 视频 → `LectureIR`（完整性、时间路径、视觉证据、知识单元） → `LectureJSON` → HTML，便于后续做 Graph RAG 跨视频主题综述。
- **类型自适应模板** — 自动识别 `technical_formula` / `procedural_tutorial` / `conceptual_talk` / `dense` 等类型，证据驱动展示对应的复盘模块（操作路线、技术要点、边界说明…），不强行凑模板。
- **首页两级折叠树** — `领域（15 类白名单）→ 方向（自由短语）` 自动归类；卡片支持悬停内联编辑 `domain` / `direction` / `tags`，「待归类」桶集中提示需要人工修正的讲义。
- **Copilot 学习型问答** — LangGraph ReAct Agent，按需输出 `[[evidence]]` / `[[background]]` / `[[extension]]` / `[[deep_dive]]` / `[[application]]` / `[[boundary]]` 分层回答；锚点越界自动标记 `⚠️`，不删原文。
- **MCP Server** — `fastmcp` 暴露 5 个只读工具（`list_lectures` / `search_lectures` / `get_chapter` / `get_knowledge_units` / `get_frame_description`），支持 stdio 与 HTTP+SSE 两种传输；`summarize_video` 默认隐藏，避免被当免费 GPU。
- **Hybrid RAG** — `sqlite-vec` 向量召回 + FTS5 关键词召回 + RRF 融合，全部在单文件 `app.db` 内，无需额外向量数据库。
- **Whisper** — `faster-whisper` 转录，结果落 `data/subtitles/{bv}.whisper.json` 缓存复用；支持本地 Hugging Face snapshot，避开网络抖动。
- **离线缓存** — 字幕、关键帧、VLM 描述、封面、调试 IR 全部命中 `data/` 缓存，重复跑同一 BV 几乎零成本。
- **生产就绪** — Dockerfile + docker-compose、GitHub Actions CI（py 3.10/3.11 矩阵 + Docker 构建）、`scripts/qa_full.*` 一键回归、`scripts/preflight.ps1` 发布前体检、Basic Auth 鉴权、`MAX_CONCURRENT_JOBS` / `COPILOT_MAX_CONCURRENT` 并发护栏。

---

## 快速开始

### 环境要求

- **Python** 3.10 或 3.11
- **ffmpeg** （系统级，关键帧抽取与音频转码）
- **DashScope API Key** （[阿里云百炼](https://dashscope.aliyuncs.com/)，用于 Qwen 文本/视觉/Embedding 模型），兼容OpenAI需额外操作
- **Node 18+** （可选，用于前端 `node --check`，CI 必备）
- **Docker 24+** （可选，用于一键容器化）

### 本地一键启动（推荐）

Windows PowerShell：

```powershell
git clone https://github.com/2772658778-ctrl/LectureMind.git
cd LectureMind
copy .env.example .env       # 然后填入 DASHSCOPE_API_KEY
./scripts/dev_up.ps1         # 默认使用 conda env 'myagent'，可加 -NoConda
```

Windows 双击启动：

```text
双击项目根目录的 Start-LectureMind.bat
```

`Start-LectureMind.bat` 会优先查找 `.venv` 或常见 Anaconda / Miniconda 路径下的 `myagent` 环境，例如 `D:\anaconda\envs\myagent\python.exe`；找到后会调用 `scripts/dev_up.ps1 -NoConda -PythonExe <python>`，避免双击时系统 `PATH python` 缺少依赖。

Linux / macOS / WSL：

```bash
git clone https://github.com/2772658778-ctrl/LectureMind.git
cd LectureMind
cp .env.example .env         # 然后填入 DASHSCOPE_API_KEY
./scripts/dev_up.sh
```

`dev_up` 脚本做的事：

1. 不存在 `.env` 时从 `.env.example` 复制并提示填写 `DASHSCOPE_API_KEY`。
2. 创建 `data/` 目录并执行 `python -m scripts.init_db` 初始化 SQLite。
3. 启动 `uvicorn app.main:app`，端口与 Host 由 `.env` 决定。

如端口已被占用，Windows 会看到 `Errno 10048`；这通常说明已有一个 `uvicorn` 实例正在运行，直接访问 `http://127.0.0.1:8000` 或先关闭旧进程再启动即可。

启动后访问 `http://127.0.0.1:8000`，使用 `BASIC_AUTH_USER` / `BASIC_AUTH_PASSWORD` 登录。

### 手动安装

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

pip install -e ".[whisper]"      
python -m scripts.init_db
python -m app.main                # → http://localhost:8000
```

### 单跑一次 Pipeline（CLI 调试）

```bash
python -m scripts.run_pipeline https://www.bilibili.com/video/BV1YM9gYdECb
```

### Docker 部署

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f lecturemind
```

容器持久化目录为 `./data`；健康检查：

```bash
curl -fsS -u admin:<password> http://localhost:8000/healthz
# {"status":"ok"}
```

---

## 配置说明

完整字段见 `.env.example`，常用项摘录：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | — | **必填**，阿里云百炼 API Key |
| `QWEN_TEXT_MODEL` | `qwen3.6-flash` | 讲义生成（`lecturize` / `LectureIR`）使用的文本模型 |
| `QWEN_VL_MODEL` | `qwen3.6-35b-a3b` | 关键帧描述 + OCR 使用的视觉模型 |
| `QWEN_COPILOT_MODEL` | `qwen3.6-plus` | Copilot 对话使用的文本模型，可与上面区分 |
| `QWEN_EMBEDDING_MODEL` | `tongyi-embedding-vision-flash-2026-03-06` | RAG 跨模态 Embedding |
| `QWEN_TEXT_ENABLE_THINKING` | `false` | 是否开启思维链；`max-preview` 模型默认必须关闭以遵循 JSON |
| `WHISPER_MODEL` | `base` | `faster-whisper` 模型；可填 HuggingFace ID 或本地 snapshot 路径 |
| `BASIC_AUTH_USER` / `BASIC_AUTH_PASSWORD` | `admin` / `change-me-please` | HTTP Basic Auth，**生产环境务必改** |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | 监听地址 |
| `DATA_DIR` | `./data` | 持久化目录（数据库、报告、缓存） |
| `MAX_CONCURRENT_JOBS` | `2` | 同时进行中的 lecturize 任务数 |
| `COPILOT_MAX_TOOL_CALLS` | `6` | Copilot 单次对话最多工具调用次数 |
| `COPILOT_MAX_CONCURRENT` | `4` | Copilot 全局并发数 |
| `BOCHA_API_KEY` | — | 可选，启用后 Copilot `🌐` 联网开关会调用 [博查 Web Search](https://bochaai.com) |
| `MCP_SERVER_TOKEN` | — | MCP SSE 模式必填的 Bearer Token；stdio 模式忽略 |
| `MCP_EXPOSE_SUMMARIZE` | `false` | 是否在 MCP 暴露 `summarize_video`，默认隐藏 |
| `BILIBILI_COOKIE_FILE` | `./data/cookies/bilibili.txt` | Netscape 格式的 cookie 文件路径；详见下文 |

> **生产建议：** 反向代理终止 TLS 后再加 IP 白名单；`COPILOT_MAX_CONCURRENT` 按 DashScope 配额调；`MAX_CONCURRENT_JOBS` 受机器内存与磁盘 I/O 限制。

### B 站 Cookie 配置（强烈建议）

未登录态访问 B 站会被严重削弱：

- `/x/web-interface/view` 返回精简元数据，**没有** `pages` / `cid`，导致后续抓不到字幕也抽不到关键帧。
- `/x/player/wbi/v2` 直接拒发 CC 字幕 URL，只能落到 Whisper 转写（慢且耗显存）。
- 大密度访问会触发 412 / 403 风控。

LectureMind 兼容 [yt-dlp](https://github.com/yt-dlp/yt-dlp) 的 **Netscape 格式 `cookies.txt`**，把它放在 `data/cookies/bilibili.txt`（路径可由 `BILIBILI_COOKIE_FILE` 改）即可。该目录被 `.gitignore` 整体挡住，不会进 commit。

**方法 A：浏览器插件（最省事）**

1. 装 Chrome / Edge 插件 [Cookie-Editor](https://cookie-editor.com/)（Chrome Web Store / Edge Add-ons 都有）。
2. 登录 `https://www.bilibili.com`，点插件图标 → 右下角 `Export` → 选 **Netscape**。
3. 把剪贴板里的内容粘贴到 `data/cookies/bilibili.txt`（首行应是 `# Netscape HTTP Cookie File`）。

**方法 B：yt-dlp 直接从浏览器导出**

```powershell
# Edge / Chrome / Firefox 任选一个；要求该浏览器已登录 b 站
yt-dlp --cookies-from-browser edge --cookies data/cookies/bilibili.txt `
       --skip-download "https://www.bilibili.com/video/BV1xx411c7mD"
```

**关键字段：** `SESSDATA`（必需）、`bili_jct`（CSRF）、`buvid3`、`DedeUserID` 这四个就够 LectureMind 用了；只少 `SESSDATA` 都会回退到匿名态。

**有效期：** B 站 `SESSDATA` 通常给一年左右；过期或换电脑后重新导一次即可，无需重启服务（每次 ingest 会重读文件）。

> **隐私提醒：** Cookie 等价于你的登录凭证。本仓库的 `.gitignore` 已把 `data/`、`.env`、`*.log` 全部排除；commit 前可以跑 `./scripts/preflight.ps1` 体检，确保不会误传敏感文件。

---

## 工作原理

```
B 站链接
   │
   ▼
[1] 解析 BV → 缓存命中？ ── 是 ── ▶ 直接返回 report_url
   │ 否
   ▼
[2] bilibili-api 取元数据 + 字幕（Whisper 结果会缓存）
   ▼
[3] yt-dlp 拉视频 → ffmpeg + PySceneDetect 提关键帧
   ▼
[4] Qwen-VL 批处理关键帧 → caption + OCR + 画面类型 + 教学价值评分（结果写入 VLM 缓存）
   ▼
[5] Qwen 文本模型综合字幕 + 帧描述 → LectureIR
       （完整性 / 时间路径 / 视觉证据 / 知识单元 / taxonomy）
   ▼
[6] LectureIR → LectureJSON（失败时回退 v1 路径并记录 generation_mode）
   ▼
[7] Jinja2 → 自包含单页 HTML
       主线推进 → 章节 → 类型复盘 → 术语速查 → 最终综合 → 复习问题 → 全局图解 → 闭环检查
   ▼
[8] SQLite 写元数据 + Copilot RAG 索引（chunk → embedding → sqlite-vec + FTS5）
   ▼
   返回 report_url
```

每一步都可独立缓存与重试：`data/subtitles/{bv}.whisper.json`、`data/keyframes/{bv}/*.jpg`、`data/vlm_cache/{bv}.json`、`data/debug/{bv}.lecture_ir.json`。

---

## Copilot 对话 Agent

讲义页右下角的圆形 **＋** 按钮就是 Copilot 入口。

- **三种提问方式**
  1. **直接问**：打开面板直接输入。
  2. **卡片引用**：章节 / 关键帧卡片上的 `📌 引用` 按钮一点即加；关键帧也支持点击 `F#` 或双击卡片快捷引用。
  3. **划词追问**：在讲义正文任意段落划词，页面浮出「📌 发给助手」。
- **图片放大**：点击讲义图片、关键帧或封面打开 lightbox；`Esc` 或点击遮罩关闭。
- **锚点交互**
  - 点击 `[t=05:21]`：在新标签页打开 B 站原片对应时间（`?t=321`）；这是回看「原视频此刻在讲什么」最直接的方式。
  - 点击 `[F7]` / `[Ch3]`：滚动到讲义内对应关键帧或章节 + 1.6 秒高亮。
  - **Shift / Alt + 点 `[t=05:21]`**：留在当前讲义页面，滚动到该时间所属章节（无对应章节时定位到最近的章节）。
  - 越界锚点会被服务端标成 `[⚠ t=…]`，不删原文，也不响应点击。
- **学习向分层回答** —— 默认 1 段 `[[evidence]]`；当问题需要前置知识、原理、应用、跨视频或边界说明时，按需追加 `[[background]]` / `[[extension]]` / `[[deep_dive]]` / `[[application]]` / `[[boundary]]`，硬上限 4 段。明显跑题的问题会用 `[[offtopic]]` 礼貌引导。
- **联网开关** —— 前端 `🌐` 切换打开后请求带 `use_web=true`，后端才注册 `web_search` 工具（需要配 `BOCHA_API_KEY`）。
- **成本护栏** —— 单次对话默认 ≤ 6 次工具调用（`COPILOT_MAX_TOOL_CALLS`），全局并发由 `COPILOT_MAX_CONCURRENT` + 每 BV 单深度锁兜住。

---

## MCP Server

外部 Agent（Cursor / Claude Desktop / ChatGPT Apps）可以通过 MCP 协议直接调用 LectureMind 的结构化检索能力，不经浏览器。

### 启动

```bash
# stdio transport（本机，默认；给 Cursor / Claude Desktop 用）
python -m scripts.run_mcp_server

# HTTP+SSE transport（远程部署）
export MCP_SERVER_TOKEN=your-long-random-token
python -m scripts.run_mcp_server --sse --host 0.0.0.0 --port 8111
```

### 工具列表

| 工具 | 说明 |
| --- | --- |
| `list_lectures` | 列出全部已处理讲义元信息（含 `domain` / `direction` / `tags`） |
| `search_lectures` | 跨讲义混合检索（向量 + FTS5）；可选 `domain` / `direction` 过滤 |
| `get_chapter` | 按 `bv` + `chapter_idx` 拉取完整章节 JSON |
| `get_knowledge_units` | 按 `bv` 拉取结构化知识单元；可选 `kind` 过滤 |
| `get_frame_description` | 按全局 `[F7]` 取关键帧详情（OCR / insight / caption） |

`summarize_video` 默认**不暴露**。要对外开启需设置环境变量 `MCP_EXPOSE_SUMMARIZE=true` 或 CLI 加 `--enable-summarize`，开启后会拉起完整 Pipeline 实例。

### Claude Desktop 配置示例

编辑 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "lecturemind": {
      "command": "python",
      "args": ["-m", "scripts.run_mcp_server"],
      "cwd": "D:/path/to/lecturemind",
      "env": {
        "DASHSCOPE_API_KEY": "sk-...",
        "DATA_DIR": "D:/path/to/lecturemind/data"
      }
    }
  }
}
```

重启 Claude Desktop 后在对话里输入 `@lecturemind 列出所有讲义` 即可看到返回。

---

## API 速查

所有 `/api/*`、`/`、`/reports/*`、`/keyframes/*` 都需要 HTTP Basic Auth；`/healthz` 同样受保护，监控端需带账号。

### 讲义生成 / Lecture Generation

| 方法 (Method) | 路径 (Path) | 说明 (Description) |
| --- | --- | --- |
| `POST` | `/api/summarize` | `{url, force_refresh?}` → `{job_id, bv_id, cached, report_url?}` |
| `GET` | `/api/jobs/{job_id}` | 轮询任务状态（`pending` / `running` / `done` / `failed`） |
| `GET` | `/api/summaries` | 已完成讲义列表，分页参数 `limit` / `offset` |
| `GET` | `/reports/{bv}.html` | 单篇讲义自包含 HTML |
| `GET` | `/keyframes/{bv}/{filename}` | 关键帧原图 |
| `GET` | `/` | 首页（讲义库两级折叠树） |

### Copilot

| 方法 (Method) | 路径 (Path) | 说明 (Description) |
| --- | --- | --- |
| `POST` | `/api/copilot/ask` | SSE 流式响应；事件类型 `token` / `tool_call` / `tool_result` / `done` / `error` |
| `GET` | `/api/copilot/lectures` | 紧凑型讲义元数据（首页折叠树消费） |
| `POST` | `/api/copilot/reindex` | 重建 RAG 索引；body 支持 `{bv}` / `{all: true}` / `{taxonomy_only: true}` |
| `POST` | `/api/copilot/taxonomy` | 手动改 `domain` / `direction` / `tags`，仅写 `summaries` 三列，不动 `summary_json` |

### 请求示例

```bash
# 触发新讲义
curl -u admin:xxx -X POST http://localhost:8000/api/summarize \
     -H 'Content-Type: application/json' \
     -d '{"url":"https://www.bilibili.com/video/BV1xx411c7mD"}'

# 轮询任务
curl -u admin:xxx http://localhost:8000/api/jobs/<job_id>

# Copilot SSE
curl -u admin:xxx -N http://localhost:8000/api/copilot/ask \
     -H 'Content-Type: application/json' \
     -d '{"bv":"BV1xx411c7mD","question":"这一章的核心问题是什么？"}'
```

---

## 项目结构

```
lecturemind/
├── app/
│   ├── main.py              # FastAPI 入口 + lifespan 装配
│   ├── api.py               # /api/summarize, /api/jobs, /api/summaries, /
│   ├── auth.py              # HTTP Basic Auth 依赖
│   ├── config.py            # pydantic-settings 配置中心
│   ├── pipeline.py          # 端到端流水线（ingest → understand → render → store）
│   ├── ingest/              # bilibili / subtitle / keyframe / cover / cookies
│   ├── understand/          # ir / ir_builder / lecturize / vlm / prompts / schema
│   ├── render/              # Jinja2 模板 + KaTeX 资源复制
│   ├── copilot/             # agent / api / tools / rag / sections / taxonomy / mcp_server
│   ├── storage/             # SQLite DAO + 迁移
│   └── static/              # copilot.css/js + lecture.css + vendor (KaTeX)
├── scripts/
│   ├── dev_up.ps1 / dev_up.sh        # 一键启动
│   ├── qa_full.ps1 / qa_full.sh      # 一键回归
│   ├── preflight.ps1                  # 发布前体检
│   ├── init_db.py                     # 初始化 SQLite schema
│   ├── run_pipeline.py                # 单视频 CLI
│   ├── run_qa_matrix.py               # 多领域 QA 矩阵 runner
│   ├── rerender_reports.py            # HTML-only 全量重渲染（不调 LLM）
│   ├── reindex_copilot.py             # 重建 RAG 索引
│   └── run_mcp_server.py              # 启动 MCP Server（stdio/SSE）
├── tests/
│   ├── test_smoke.py                  # 离线 smoke
│   ├── test_copilot.py                # Copilot 单元/集成测试
│   └── test_mcp_server.py             # MCP 工具表测试
├── docs/
│   ├── deploy.md                      # 部署手册
│   ├── release_checklist.md           # 发布前清单
│   └── qa/                            # QA 矩阵与跟踪
├── data/                              # 运行时（已 gitignore）
│   ├── app.db                         # SQLite + sqlite-vec
│   ├── reports/{bv}.html
│   ├── keyframes/{bv}/*.jpg
│   ├── subtitles/, audio/, vlm_cache/, covers/, debug/
├── .github/workflows/ci.yml           # py 3.10/3.11 矩阵 + Docker 构建
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 开发与测试

### 一键回归

```powershell
# Windows
./scripts/qa_full.ps1                     # 默认 conda env 'myagent'
./scripts/qa_full.ps1 -CondaEnv foo       # 切换到其他 conda 环境
./scripts/qa_full.ps1 -NoConda            # 用 PATH 上的 python
```

```bash
# Linux / macOS
./scripts/qa_full.sh
CONDA_ENV=myagent ./scripts/qa_full.sh
```

`qa_full` 会依次跑：

1. `node --check app/static/copilot.js` — 前端语法
2. `python -m compileall app tests scripts` — Python 字节码体检
3. `python -m pytest tests/ -v` — 全量回归

### 仅跑离线 smoke

```bash
pytest tests/test_smoke.py -v        # 不调用网络 / API
```

### 模板/CSS 改动后重渲历史报告

```bash
python -m scripts.rerender_reports   # 复用 LectureIR / SQLite，不再调 LLM
```

### 多领域 QA 矩阵

```bash
python -m scripts.run_qa_matrix --only BV1xx411c7mD     # 单条
python -m scripts.run_qa_matrix --force                  # 强制重跑全部
```

### 发布前体检

```powershell
./scripts/preflight.ps1               # 检查 .env / .gitignore / .env.example 卫生
./scripts/preflight.ps1 -SkipTests    # 跳过 pytest
```

---

## 部署

完整部署细节见 [`docs/deploy.md`](./docs/deploy.md)。三种典型路径：

1. **本地 Python** — Windows 双击 `Start-LectureMind.bat`，或命令行运行 `./scripts/dev_up.ps1` / `dev_up.sh`，最快启动。
2. **Docker Compose** — `docker compose up -d --build`，挂载 `./data` 持久化。
3. **MCP Server** — `python -m scripts.run_mcp_server`（stdio）或 `--sse`（远程，必填 `MCP_SERVER_TOKEN`）。

发布到 GitHub 前请按 [`docs/release_checklist.md`](./docs/release_checklist.md) 走一遍清单（`.env` 卫生、`.gitignore` 覆盖、Docker 构建、`scripts/preflight.ps1` 体检）。

---

## 路线图

- [ ] 完整 4 BV × 3 题真实 SSE 矩阵在浏览器 + Cursor / Claude Desktop 真机复测
- [ ] 全库 Embedding 重建（剩余 10 BV）
- [ ] 长视频分片 lecturize / LectureIR 生成（>30 min 拆段）
- [ ] 主题封面策略与图鉴库
- [ ] Active Recall 题目 + Anki 导出
- [ ] FTS5 全文检索接入首页
- [ ] **Graph RAG 跨视频主题综述** — 基于 `LectureIR.knowledge_units` 做主题聚合
- [ ] 输入扩展：YouTube / 本地 mp4 / 抖音 / 视频号 /小红书


---

## 常见问题

| 现象 | 排查 |
| --- | --- |
| `401 Unauthorized` | 浏览器没附 Basic Auth；或 `.env` 中用户名/密码与请求不匹配 |
| `/healthz` 也 401 | 这是预期行为；监控端要带账号 |
| Copilot SSE 立即断开 | 检查 `DASHSCOPE_API_KEY` 与 `COPILOT_MAX_CONCURRENT` 是否打满 |
| Whisper 报 HF SSL | 已加本地 snapshot 回退；首次离线运行需提前下好 `Systran/faster-whisper-base` |
| Bilibili 抓取 SSL EOF / 403 | 在 `data/cookies/bilibili.txt` 放 Netscape 格式 cookie；脚本会自动重试 + UA fallback |
| 关键帧描述空白 | 没有 DashScope VL 配额或视觉模型不可用；不影响主流程，但全局视觉证据会减少 |
| 老报告 UI 缺新功能 | `python -m scripts.rerender_reports` 会重渲染所有 `data/reports/BV*.html` |
| 任务卡在 75–78% | 可能是 `QWEN_TEXT_MODEL` 配额耗尽或代理拦截；查看 `data/debug/{bv}.lecture_ir.*.txt`，必要时切到其他模型如 `qwen3.6-max-preview` 并设 `QWEN_TEXT_ENABLE_THINKING=false` |

---

## 致谢

LectureMind 站在以下开源项目之上：
- [wdkns](https://github.com/wdkns/wdkns-skills/blob/main/skills/bilibili-render-pdf/SKILL.md) — 视频总结灵感
- [FastAPI](https://fastapi.tiangolo.com/) — Python Web 框架
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent 图编排
- [fastmcp](https://github.com/jlowin/fastmcp) — Python MCP Server SDK
- [DashScope / 通义千问](https://dashscope.aliyuncs.com/) — Qwen 文本/视觉/Embedding 模型
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper 推理优化
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频/字幕下载
- [PySceneDetect](https://www.scenedetect.com/) — 关键帧检测
- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — B 站元数据
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — SQLite 向量索引
- [KaTeX](https://katex.org/) — 数学公式渲染



## 许可证

[MIT](./LICENSE) © LectureMind contributors

> 本项目仅作个人学习/研究用途；请遵守 B 站、阿里云百炼及任何第三方平台的服务条款，自行评估合规风险。
