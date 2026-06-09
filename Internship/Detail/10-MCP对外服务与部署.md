# MCP 对外服务与部署

## 文档定位

`Internship/04` 把 MCP 描述为「项目从单一应用升级为知识服务后端」，本篇细化 MCP 这一层的实现和部署形态：

- `app/copilot/mcp_server.py` — FastMCP server 构造器。
- `scripts/run_mcp_server.py` — 启动 CLI。
- `Dockerfile` / `docker-compose.yml` / `docs/deploy.md` — 部署形态。
- 「summarize_video 默认隐藏」的安全边界。

## 一句话定位

LectureMind 的 MCP 不是 Web API 的薄转发，而是一个**独立的 ToolContext + FastMCP server + 显式 lifespan + 工具按需注册**的进程。读取类工具默认全开，昂贵的 `summarize_video` 默认隐藏，让外部 Agent 在最小权限上就能消费讲义知识。

## 模块边界

### app/copilot/mcp_server.py（核心）

`@d:\Diet_Agent_NEW\app\copilot\mcp_server.py:1-29` 的 docstring 列了 4 个关键决策：

- **独立 ToolContext** —— `build_default_context(with_pipeline=False)` 自己开 DB + RAGStore，**不依赖 FastAPI 进程**。
- **summarize_video 默认隐藏** —— `MCP_EXPOSE_SUMMARIZE=true` 或 `--enable-summarize` 才注册。注册时需要 `pipeline=Pipeline(db)`，会拉满整套 ingest 栈。
- **错误以 `ValueError` 上抛** —— FastMCP 把 `ValueError` 序列化成 MCP 标准 error 包，Agent 客户端能解析；内部 `ToolError` 被翻译为 `[code] message` 字符串前缀。
- **Lifespan 绑定生命周期** —— `build_mcp(...)` 注册 `lifespan` hook，启动时初始化 DB+RAG，关闭时清理。

### build_default_context

`@d:\Diet_Agent_NEW\app\copilot\mcp_server.py:54-84`：

```python
async def build_default_context(*, with_pipeline=False):
    settings = get_settings()
    db = Database(settings.db_path)
    await db.init()
    rag = RAGStore(settings.db_path)
    try:
        await rag.init()
    except Exception as exc:
        logger.warning("RAGStore init failed in MCP server (%s); search tools will return empty.", exc)
    pipeline = None
    if with_pipeline:
        from app.pipeline import Pipeline as _Pipeline  # 懒导入避免拉 ingest 依赖
        pipeline = _Pipeline(db)
    return ToolContext(db=db, rag=rag, pipeline=pipeline), db, rag
```

`Pipeline` 是**懒导入**的：不开 summarize 就不会引入 Whisper / yt-dlp / ffmpeg-python 这些重依赖，stdio 模式启动更快。

### build_mcp 构造器

`@d:\Diet_Agent_NEW\app\copilot\mcp_server.py:122-358` 的 `build_mcp(ctx, expose_summarize, name, instructions)`：

- 如果传入 `ctx`，说明调用方自己管理 DB/RAG（测试场景）。
- 如果 `ctx=None`，**owned_ctx=True**，由 lifespan 在启动时调 `build_default_context` 创建并在 stop 时关闭。
- 创建 `FastMCP(name, instructions, lifespan)`。
- 注册 11 个常规工具（每个工具按规则注册）。
- `expose_summarize=True` 时额外注册 `summarize_video`。

## 默认暴露的 11 个工具

`@d:\Diet_Agent_NEW\app\copilot\mcp_server.py:186-339`：

| 工具 | 入参 | 含义 |
|------|------|------|
| `search_lectures` | `query, top_k, domain?, direction?` | 跨视频检索（结合 taxonomy 过滤） |
| `search_evidence` | `bv, query, top_k` | 单 BV 在 EvidenceIndex 上做证据级检索 |
| `get_note_unit` | `bv, unit_id` | 按 unit_id 读 TeachingUnit |
| `get_evidence_object` | `bv, evidence_id` | 按 evidence_id 读 EvidenceObject |
| `get_chapter` | `bv, chapter_idx` | 按章节读完整内容 |
| `get_frame` | `bv, frame_id` | 按 frame_id 读关键帧 |
| `get_quote_context` | `bv, quote` | 给一段文本找上下文 |
| `explain_frame` | `bv, frame_id, context_radius_seconds=60` | 解释一张帧的含义（含上下文） |
| `get_knowledge_units` | `bv, kind?` | 读知识单元列表 |
| `get_frame_description` | `bv, frame_id` | 只取帧的 VLM 描述 |
| `list_lectures` | `limit, domain?, direction?` | 讲义列表 |

每个 wrapper 都是一致的：

```python
@mcp.tool(name="search_lectures", description=_describe(tool_registry.search_lectures))
async def search_lectures(query, top_k=5, domain=None, direction=None) -> dict[str, Any]:
    try:
        out = await tool_registry.search_lectures(_ctx(), query=query, top_k=top_k, ...)
    except ToolError as exc:
        _handle_tool_error(exc)
    return _dump(out)
```

`_describe(fn)` 取 docstring 第一行作为 MCP 工具描述。`_dump(obj)` 把 Pydantic model 翻译成 JSON dict。

## 隐藏的 summarize_video

`@d:\Diet_Agent_NEW\app\copilot\mcp_server.py:341-356`：

```python
if expose_summarize:
    @mcp.tool(name="summarize_video", description=_describe(tool_registry.summarize_video))
    async def summarize_video(url: str, force_refresh: bool = False) -> dict[str, Any]:
        try:
            out = await tool_registry.summarize_video(_ctx(), url=url, force_refresh=force_refresh)
        except ToolError as exc:
            _handle_tool_error(exc)
        return _dump(out)
```

注意三件事：

- **缩进位于 `expose_summarize` 分支内**，工具不会被注册到 MCP 列表里。Agent 客户端调 `list_tools` 也看不到。
- summarize_video 在 `tools.py` 里通过 `ctx.pipeline` 调用 `Pipeline.run(url, force_refresh)`，没有 pipeline 直接抛 `ToolError("unsupported")`。
- 启动时只有 `MCP_EXPOSE_SUMMARIZE=true` 或 `--enable-summarize` 才会让 `build_default_context(with_pipeline=True)` 真的造 Pipeline 实例。

## 启动 CLI：scripts/run_mcp_server.py

`@d:\Diet_Agent_NEW\scripts\run_mcp_server.py:30-95`：

```bash
# 本地 stdio（Cursor / Claude Desktop / 其他本地 MCP 客户端默认）
python scripts/run_mcp_server.py

# 远程 HTTP + SSE（外部 Agent / ChatGPT Apps）
python scripts/run_mcp_server.py --sse --host 0.0.0.0 --port 8111

# 显式开启 summarize_video
python scripts/run_mcp_server.py --enable-summarize
```

### 关键边界检查

`@d:\Diet_Agent_NEW\scripts\run_mcp_server.py:64-69`：

```python
if args.sse and not os.environ.get("MCP_SERVER_TOKEN") and not settings.mcp_server_token:
    print("ERROR: --sse requires MCP_SERVER_TOKEN (bearer token) to be set in env.", file=sys.stderr)
    return 2
```

SSE 模式**强制要求** `MCP_SERVER_TOKEN`，避免裸暴露 HTTP 端点。stdio 模式不需要（本地进程通信，无网络面）。

### 两种 transport 启动

```python
if args.sse:
    asyncio.run(mcp.run_http_async(host=args.host, port=args.port))
else:
    asyncio.run(mcp.run_stdio_async())
```

`mcp.run_stdio_async()` 走 stdin/stdout JSON-RPC（MCP 协议默认）；`mcp.run_http_async()` 起一个 HTTP 服务器，事件流走 SSE。

## 与主进程的关系

MCP server 是**独立进程**——

- 不读 FastAPI 的 `app.state`。
- 不抢 FastAPI 进程的 `data/app.db`（SQLite WAL 允许多进程并发只读访问，写入由 FastAPI 进程做）。
- 启动顺序：先用 FastAPI 跑过几次视频生成 → 数据沉淀到 `data/app.db` 和 `data/reports/` → 启动 MCP server，外部 Agent 就能看到这些讲义。

也就是说 MCP 不参与「生成」，只参与「消费」。这种「读写分离 + 进程隔离」的结构让两种使用形态可以独立扩缩容。

## Docker / Compose 部署

### Dockerfile

`@d:\Diet_Agent_NEW\Dockerfile` 38 行，关键点：

- 基础镜像 `python:3.11-slim`。
- 系统依赖装 `ffmpeg / git / curl / ca-certificates / build-essential`。
- `pip install -e ".[whisper]"` 装项目和可选 whisper 依赖。
- `DATA_DIR=/app/data` + `EXPOSE 8000`。
- 内置 `HEALTHCHECK` 每 30s 调 `/healthz`。
- 默认 `CMD ["python", "-m", "app.main"]`（启 FastAPI）。

### docker-compose.yml

挂载 `./data:/app/data` 持久化，端口 8000:8000。

### docs/deploy.md 的三种运行形态

`@d:\Diet_Agent_NEW\docs\deploy.md`：

- **本地 Python** — `./scripts/dev_up.ps1` 一键启动。
- **Docker Compose** — `docker compose up -d --build`。
- **MCP Server 单独** — `python -m scripts.run_mcp_server`。

`docs/deploy.md` 还列了 5 个生产建议（反向代理 + TLS / 持久化 / 并发参数 / 冷启动 / 重渲染）和 7 个常见问题排查。

## 启动脚本生态

`@d:\Diet_Agent_NEW\scripts\` 12 个脚本，运维相关的 6 个：

- `dev_up.ps1 / dev_up.sh` — 一键启动 FastAPI。
- `init_db.py` — 初始化 SQLite schema。
- `run_pipeline.py` — CLI 单视频处理。
- `run_mcp_server.py` — MCP 启动。
- `rerender_reports.py` — 不重调 LLM 重渲染所有历史 HTML。
- `reindex_copilot.py` — 重建 RAG 索引（支持 `--reset-vectors` 维度切换）。

### rerender_reports 的工程意义

「重渲染」是「内容生成」与「页面渲染」分离的直接收益。改动 CSS / Jinja 模板 / 渲染逻辑后，不用重调 LLM 也能让所有历史报告刷新——因为 `summary_json` 已经完整持久化在 DB 里。

## 配置与权限

### MCP 相关配置项

- `MCP_EXPOSE_SUMMARIZE=true|false` — 是否注册 summarize_video。默认 false。
- `MCP_SERVER_TOKEN` — SSE 模式必填，Bearer Token。
- `MCP_BASE_URL` — 客户端连接地址（若需要）。

### Web API 相关

- `BASIC_AUTH_USER` / `BASIC_AUTH_PASSWORD` — 整个 FastAPI 应用的 Basic Auth。
- `APP_HOST=0.0.0.0` / `APP_PORT=8000`。
- `COPILOT_MAX_CONCURRENT` — Copilot 全局并发上限。

## Healthz 与 Basic Auth

`/healthz` 同样受 Basic Auth 保护（**这是 deploy.md 显式标注的「预期行为」**）。监控探针必须带凭据。

## 测试覆盖

`tests/test_mcp_server.py` 17733 字节，10 个测试用例覆盖：

- 默认启动只暴露读取工具。
- `expose_summarize=True` 才出现 summarize_video。
- 工具调用走 Pydantic schema 严格校验。
- ToolError → ValueError 翻译。
- Lifespan 在 owned_ctx 下正确清理 DB/RAG 连接。

具体见 `Detail/14-测试-QA-工程化.md`。

## 「读默认开，写默认关」的工程哲学

这条原则在 MCP 这一层尤其明显：

- 全部读取工具默认开 — 因为外部 Agent 接入的预期就是「来读知识」。
- summarize_video 默认关 — 因为它会触发：
  - 长时网络下载（yt-dlp 视频 + 字幕）。
  - 重 CPU（PySceneDetect + ffmpeg）。
  - 昂贵 LLM 调用（VLM + 文本）。
  - 写库（覆盖已有 summary）。

让默认安全是工程交付的标杆做法。要打开就必须显式说「我懂这意味着什么」。

## 写简历可以怎么提炼

- 把讲义知识能力封装为独立 FastMCP server，支持 stdio（本地 Cursor / Claude Desktop）和 HTTP + SSE（远端 Agent）两种 transport，并对 SSE 模式强制要求 Bearer Token。
- 实现了「读默认开 / 写默认关」的权限边界：11 个读取工具默认注册，`summarize_video` 只有在 `MCP_EXPOSE_SUMMARIZE=true` 或 `--enable-summarize` 时才暴露并惰性初始化 Pipeline，避免外部 Agent 误触重 IO / 重模型操作。
- 通过独立 `ToolContext` + lifespan hook 让 MCP server 完全脱离 FastAPI 进程运行，与主应用共享同一份 `data/app.db` 但读写分离，支持独立扩缩容。
- 完成了 Dockerfile + docker-compose + dev_up 脚本 + reindex / rerender 维护脚本，形成从开发到部署的完整运维闭环。

## 面试可展开点

### 1. 为什么不直接把 FastAPI 的 Copilot 路由复用给外部 Agent

两个原因：

- 协议不同——FastAPI 是 HTTP/JSON，MCP 是 stdio/SSE + JSON-RPC + 工具 schema 自描述。
- 安全边界不同——FastAPI 假设走 Basic Auth + 反向代理；MCP 假设外部 Agent 拿 token 直连，权限模型要重新设计。

MCP server 是独立产物，与 FastAPI 共享业务逻辑（`tools.py`）但不共享接口形态。

### 2. summarize_video 为什么不内嵌在默认工具集

它是唯一一个**写操作 + 重 IO + 高成本**的工具。如果默认暴露，外部 Agent 一调就触发一次完整流水线，可能用掉用户几十块 DashScope 额度。把它放显式开关后面，是把「合理默认」放到正确的位置。

### 3. 为什么 MCP server 要重新开 DB / RAG 而不是复用 FastAPI 的实例

进程隔离。MCP 经常被 Cursor / Claude Desktop 这类 IDE 客户端 spawn 成子进程，启动时不知道 FastAPI 是否在跑。让它自己开 DB/RAG 让两种使用场景独立。WAL 模式保证多进程读不冲突。

### 4. stdio vs SSE 的取舍

- **stdio** 适合本地 IDE 集成（Cursor / Claude Desktop），无网络面，无需鉴权，启动最快。
- **SSE** 适合远端调用（Web UI / 其他服务的 Agent），需要 token 防滥用，可负载均衡。

两种 transport 共享同一份 `build_mcp` 输出，只是启动方式不同。

### 5. 为什么 ToolError 翻译为 ValueError

FastMCP v3 把抛出的标准 Exception 序列化为 MCP 标准 error envelope。`ValueError("[code] message")` 让远端 Agent 既能拿到结构化 error type，又能从消息开头解析机器可读 code。这种「在协议边界做翻译」是 server-side 跨协议互操作的常用模式。

## 当前文档中的事实与推断

### 事实

- `build_mcp` 注册 11 个常规工具，summarize_video 在 `expose_summarize=True` 时额外注册。
- `--sse` 模式强制需要 `MCP_SERVER_TOKEN`，stdio 模式不需要。
- MCP server 独立进程，共享 `data/app.db` 但不依赖 FastAPI 运行。
- Dockerfile 安装 ffmpeg / git / curl 三个系统依赖。
- `rerender_reports.py` 不调 LLM 就能刷新所有历史报告。

### 推断

- MCP 这一层是项目「内容生成产品 → 知识服务后端」转型最有代表性的部分。
