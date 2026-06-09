# Claude Progress

## 当前已验证基线
- `scripts/qa_full.ps1` 是当前全量自动回归入口。
- `scripts/preflight.ps1` 是当前推送前卫生检查入口。
- `tests/test_smoke.py` 是当前最小 smoke 基线。
- `app/pipeline.py` 与 `docs/qa/` 构成当前项目的运行时验证主链。
- `D:\anaconda\envs\myagent\python.exe -m pytest tests/test_harness_workspace.py -v` 已通过 10/10。
- `powershell -ExecutionPolicy Bypass -File .\init.ps1` 已通过：`compileall` 通过，`tests/test_harness_workspace.py` 10 passed，`tests/test_smoke.py` 107 passed。
- `wsl.exe bash -lc 'cd /mnt/d/Diet_Agent_NEW && ./init.sh'` 已通过：`compileall` 通过，`tests/test_harness_workspace.py` 10 passed，`tests/test_smoke.py` 106 passed。

## 当前唯一 Active Task
- No active implementation task. `harness-004` 已完成验证并进入可交接状态。
- Git 安全基线：当前执行分支为 `runtime-harness-v1`；最近可回到的 2026-05-14 前基线已标记为 `runtime-baseline-pre-2026-05-14` -> `aab3437`（仓库内无 2026-05-14 提交）。

## 最近完成
- 新增 runtime harness 设计规格：`docs/superpowers/specs/2026-06-09-lecturemind-runtime-harness-design-v2.md`
- 新增 runtime harness 实现计划：`docs/superpowers/plans/2026-06-09-lecturemind-runtime-harness-implementation-plan.md`
- 新增 `app/runtime/` 最小壳层：`contracts.py`、`policy.py`、`observe.py`
- `app/pipeline.py` 已接入 `lecture_compile` runtime metadata，并暴露 `last_run_record`
- `app/copilot/api.py` 已给 `copilot_answer` SSE `done` payload 补齐 `trace_id / run_contract / policy_snapshot / verdict`
- `app/copilot/tools.py` 与 `app/copilot/mcp_server.py` 已补最小 tool runtime metadata 和 `mcp_tool_request` debug run record
- `tests/test_runtime_harness.py` 已新增并通过 12/12
- `D:\anaconda\envs\myagent\python.exe -m pytest tests/test_smoke.py tests/test_copilot.py tests/test_mcp_server.py tests/test_runtime_harness.py -q` 已通过 295 passed
- `powershell -ExecutionPolicy Bypass -File .\init.ps1` 已继续通过：`compileall` 通过，`tests/test_harness_workspace.py` 10 passed，`tests/test_smoke.py` 107 passed
- 新增开发环境 harness 设计规格：`docs/superpowers/specs/2026-06-09-lecturemind-agent-workspace-harness-design.md`
- 新增项目运行时 harness 设计规格：`docs/superpowers/specs/2026-06-09-lecturemind-project-runtime-harness-design.md`
- 新增开发环境 harness 实施计划：`docs/superpowers/plans/2026-06-09-lecturemind-agent-workspace-harness-upgrade-plan.md`
- 对 `带教文档/09-Harness升级方案-借鉴learn-harness-engineering.md` 补充了基于仓库真实状态的勘误。
- 新增根目录入口层：`AGENTS.md`、`CLAUDE.md`、`ARCHITECTURE.md`
- 新增状态真源：`feature_list.json`、`claude-progress.md`、`clean-state-checklist.md`
- 新增启动自检脚本：`init.ps1`、`init.sh`
- `README.md` 已新增 `Agent Quick Start`
- `tests/test_harness_workspace.py` 已新增并通过 10/10。

## 下一步
1. 如需继续推进，可在 runtime harness v1 之上补更细的失败语义与恢复策略，而不是重写入口协议。
2. 如需扩展多 Agent 协作，可优先从现有 staged workflow 的 role contract 和 artifact contract 开始，而不是引入泛化 agent platform。
3. 若准备发布或合并，再补跑 `scripts/preflight.ps1 -SkipTests` 做发布前卫生检查。

## 阻塞与注意事项
- `docs/*` 与 `handoff*.md` 目前主要通过 `.gitignore` 做私有化；第一阶段不放开整个深文档区。
- `ruff` 当前未安装，开发环境 harness 第一阶段不把它列入 `init` 必跑项。
- `init.*` 必须只做自检和最小验证，不得启动 `uvicorn`。

## 深度参考
- 当前主交接：`handoff.md`
- 收尾与发布卫生：`handoff_closeout.md`
- 设计真源：`docs/superpowers/specs/2026-06-09-lecturemind-agent-workspace-harness-design.md`
- 实施计划：`docs/superpowers/plans/2026-06-09-lecturemind-agent-workspace-harness-upgrade-plan.md`
