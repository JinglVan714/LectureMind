# LectureMind 发布前检查清单

> 用于每次发布候选（Release Candidate）和上传 GitHub 之前的硬性体检。
> 全部勾完才能 push 到公开仓库。

## 1. 代码与回归

- [ ] `node --check app/static/copilot.js` 通过
- [ ] `python -m compileall app tests scripts` 通过
- [ ] `python -m pytest tests/ -v` 全绿（当前基线 137 / 137）
- [ ] 手动跑过一次：`./scripts/qa_full.ps1`（或 `.sh`）

## 2. 模板与静态资源

- [ ] 修过 `app/render/templates/*.j2`、`app/static/*.css`、`app/static/*.js` 后跑过：
      `python -m scripts.rerender_reports`
- [ ] 抽查至少一个 `data/reports/BV*.html`：包含 `cp-panel`、`mainline-steps`、`assets/katex`，不含旧的 `学习地图` / `递进关系`

## 3. 安全 / 隐私

- [ ] `.gitignore` 包含 `.env`、`.env.*`、`data/`、各类缓存（已默认配置）
- [ ] `.dockerignore` 排除 `.env`、`data/`、`tests/`、`docs/`、`.git/`
- [ ] `.env.example` 中所有秘钥都是占位符（如 `sk-replace-me`、`change-me-please`）
- [ ] `.env` 未被 git 跟踪：`git ls-files --error-unmatch .env` 应返回非 0
- [ ] `data/` 未被 git 跟踪：`git ls-files --error-unmatch data` 应返回非 0
- [ ] `data/cookies/bilibili.txt`、`data/debug/*.json` 等敏感缓存不在提交里

> 一键体检：`./scripts/preflight.ps1`

## 4. 配置项 / 文档同步

- [ ] 新增 / 修改的环境变量，已同步到 `app/config.py` + `.env.example` + `docs/deploy.md`
- [ ] `docs/deploy.md` 中的命令仍可直接复制粘贴运行
- [ ] `pyproject.toml` 的 `package-data` 覆盖所有 `app/render/templates/*` 与 `app/static/{*.css,*.js,vendor/**}`

## 5. Docker 构建（可选但推荐）

- [ ] `docker build -t lecturemind:rc .` 本地能成功
- [ ] `docker compose up -d --build` 后 `curl -u admin:xxx http://localhost:8000/healthz` 返回 `ok`
- [ ] CI（`.github/workflows/ci.yml`）`docker build` job 通过

## 6. 手动 QA 抽查

按 `docs/qa/lecturemind_copilot_qa.md` 抽测：

- [ ] 首页搜索/筛选（标题、BV、author、domain、direction、tags、未归类）
- [ ] 方向治理：fragment hint + datalist 建议出现，可保存
- [ ] Copilot 本地持久化：刷新后历史/草稿/面板状态保留；`Clear` 后清空
- [ ] 锚点跳转：`[t=mm:ss]` 滚动到对应章节，`[F#]` 跳到关键帧
- [ ] 引用 chips：发送后被快照保留；frame 缩略图渲染正常
- [ ] 移动端布局（最低优先级，但建议过一遍）

## 7. GitHub 上传前最后一步

- [ ] 仓库初始化（如尚未）：`git init`，确认 `git status` 不显示 `.env` / `data/`
- [ ] 写好 `LICENSE`（用户决定，README 之外的封装不擅自添加）
- [ ] `git add . && git commit -m "release: vX.Y.Z"`
- [ ] `git remote add origin <url>` → `git push -u origin main`
- [ ] 在 GitHub 上确认：未泄漏 `.env`、未上传 `data/`、Actions 第一次绿灯
