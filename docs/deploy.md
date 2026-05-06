# LectureMind 部署指南

本指南覆盖三种最常见的运行方式：**本地 Python**、**Docker Compose**、**生产/共享场景**。本文档不替代 README，仅记录运维细节。

---

## 0. 前置条件

| 依赖              | 用途                                  | 备注                                              |
| ----------------- | ------------------------------------- | ------------------------------------------------- |
| Python 3.10 / 3.11 | 后端运行时                           | conda env `myagent` 已验证                        |
| ffmpeg            | 关键帧抽取、音频转码                  | 系统级安装（Windows: `winget install ffmpeg`）   |
| Node 18+          | 仅用于 `node --check` 前端语法检查    | 可选；CI 必备                                     |
| Docker 24+        | 一键容器化运行                        | 仅 Docker Compose 路径需要                        |
| DashScope API Key | 调用 Qwen 文本/视觉/Embedding 模型    | 写入 `.env` 的 `DASHSCOPE_API_KEY`               |

---

## 1. 本地 Python 一键启动

```powershell
# Windows PowerShell（已激活 conda）
./scripts/dev_up.ps1                  # 默认使用 conda env 'myagent'
./scripts/dev_up.ps1 -CondaEnv foo    # 切换 conda 环境
./scripts/dev_up.ps1 -NoConda         # 使用 PATH 上的 python
./scripts/dev_up.ps1 -Reload          # uvicorn --reload，开发用
```

```bash
# Linux / macOS / WSL
./scripts/dev_up.sh                   # 直接用系统 python
CONDA_ENV=myagent ./scripts/dev_up.sh # conda 环境
RELOAD=1 ./scripts/dev_up.sh          # 开发热重载
```

脚本做的事情：

1. 不存在 `.env` 时从 `.env.example` 复制一份并提示填写 `DASHSCOPE_API_KEY`
2. 创建 `data/` 目录并通过 `python -m scripts.init_db` 初始化 SQLite schema
3. 启动 `uvicorn app.main:app`（端口 / Host 由 `.env` 控制）

启动后访问 `http://127.0.0.1:8000/`，使用 `.env` 中的 `BASIC_AUTH_USER` / `BASIC_AUTH_PASSWORD` 登录。

---

## 2. Docker Compose 一键部署

```bash
cp .env.example .env       # 修改 DASHSCOPE_API_KEY、BASIC_AUTH_PASSWORD 等
docker compose up -d --build
docker compose logs -f lecturemind
```

镜像构建上下文受 `.dockerignore` 限制，不包含 `data/` / `.env` / `tests/` / `docs/` 等内容；运行时通过 `volumes: ./data:/app/data` 持久化报告与缓存。

健康检查：`curl -fsS -u admin:xxx http://localhost:8000/healthz` 应返回 `{"status":"ok"}`。

升级镜像：

```bash
docker compose pull          # 远程镜像
docker compose up -d --build # 本地源码改动
```

---

## 3. MCP Server（可选）

```bash
# stdio（Cursor / Claude Desktop 默认调用）
python -m scripts.run_mcp_server --transport stdio

# HTTP SSE（远端共享）
python -m scripts.run_mcp_server --transport sse --host 0.0.0.0 --port 8765
```

SSE 模式必须设置 `MCP_SERVER_TOKEN`；客户端请求需带 `Authorization: Bearer <token>`。

---

## 4. 生产环境建议

- **反向代理**：Nginx / Caddy 终止 TLS，将 `:443` 转发到 `app:8000`，并强制 Basic Auth 之上再加 IP 白名单
- **持久化**：必须挂载 `./data`，包含 `app.db`、`reports/`、`keyframes/`、`vlm_cache/`、`subtitles/`
- **进程数**：`MAX_CONCURRENT_JOBS` 控制 lecturize 并发；`COPILOT_MAX_CONCURRENT` 控制 Copilot 并发，按 DashScope quota 调
- **冷启动**：第一次跑某个 BV 会下载视频/音频与抽帧，耗时较高；之后命中 `data/` 缓存
- **重渲染**：模板/CSS/JS 改动后跑 `python -m scripts.rerender_reports` 即可刷新所有历史报告，无需重新调用 LLM

---

## 5. 常见问题

| 现象                                  | 排查                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------- |
| `401 Unauthorized`                    | 浏览器没附 Basic Auth；或 `.env` 中用户名/密码与请求不匹配                            |
| `healthz` 永远 401                    | 这是预期：`/healthz` 也受 Basic Auth 保护，监控端要带账号                             |
| Copilot SSE 立即断开                  | 检查 `DASHSCOPE_API_KEY`；`COPILOT_MAX_CONCURRENT` 是否被打满                         |
| Whisper 报错 `HF SSL`                 | 已加本地 snapshot 回退；首次离线运行需提前下好 `Systran/faster-whisper-base`           |
| Bilibili 抓取 `SSL EOF` / `403`       | 在 `data/cookies/bilibili.txt` 放 Netscape 格式 cookie；脚本会自动重试 + UA fallback |
| `frame description` 空白              | 没有 DashScope VL 配额或视觉模型不可用；不影响主流程，但全局视觉证据会减少           |
| 老报告 UI 缺失新功能                   | `python -m scripts.rerender_reports` 会重渲染所有 `data/reports/BV*.html`             |

---

## 6. 升级路径

1. 拉取新代码
2. `./scripts/qa_full.ps1`（或 `.sh`）确认自动化绿灯
3. `python -m scripts.rerender_reports`（如改了模板/CSS）
4. 重启服务（`docker compose up -d --build` 或重启 uvicorn）
