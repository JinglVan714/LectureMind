# Internship/Detail 阅读地图

这一组文档是对 `Internship/01~05` 四篇总览文档的**模块级深入展开**，共 16 篇 markdown，按主链路顺序编排。每一篇都对应代码里一个明确的模块或一组紧耦合的模块，做到「论断 → 代码位置 → 数据形态 → 设计动机 → 写简历点 → 面试问题」一站式覆盖。

## 文档清单

| 编号 | 标题 | 主要覆盖模块 |
|------|------|--------------|
| `00` | 项目总览：分层架构与主链路 | 5 层架构 / 模型栈 / 双 SQLite 缓存 / 长度路由全景 |
| `01` | Ingest 层：视频原料采集深入 | `app/ingest/{bilibili,subtitle,keyframe,cover,cookies}.py` |
| `02` | 视觉理解：VLM 分级与缓存 | `app/understand/{vlm,frame_ranker,vlm_cache}.py` |
| `03` | 长度路由与章节规划 | `app/understand/{profile,length_adapt,chapter_planner,chapter_cache}.py` |
| `04` | LectureIR：中间表示与 Schema | `app/understand/{ir,schema,ir_sanity}.py` |
| `05` | 多 Agent 构建：StudyQuestion / Critic / Reviser | `app/understand/{agents,critic_context,ir_patches}.py` + `ir_builder` |
| `06` | Map-Reduce 长视频处理 | `app/understand/ir_map_reduce.py` |
| `07` | 双投影编译：Note 与 Evidence | `app/understand/{lecture_compiler,lecture_blueprint,evidence_index,composition}.py` |
| `08` | Copilot RAG：混合检索 | `app/copilot/rag.py`（chunking / embedding / sqlite-vec + FTS5 + RRF） |
| `09` | Copilot Agent 与 SSE 接口 | `app/copilot/{api,agent,tools,prompts,sections,taxonomy}.py` |
| `10` | MCP 对外服务与部署 | `app/copilot/mcp_server.py` + `scripts/run_mcp_server.py` + Docker |
| `11` | Storage：SQLite 与缓存物理布局 | `app/storage/db.py` + 3 个 SQLite 文件 |
| `12` | Config：配置体系与降级矩阵 | `app/config.py` 全部 ~50 个字段 + 24+ 处降级 |
| `13` | Pipeline：端到端编排 | `app/pipeline.py` 8 阶段 + telemetry |
| `14` | 测试、QA 与工程化质量保障 | `tests/` 25 个文件 + `scripts/_verify_*` + `qa_full` + `preflight` + release_checklist |
| `15` | 面试讲述与预期问答 | 项目介绍模板 + 21 个预期问题 + 白板架构图 + 避坑指南 |

## 推荐阅读路径

### 想最短路径理解主线

`00 → 01 → 04 → 05 → 07 → 13`

6 篇，能覆盖「视频怎么变成讲义」的全链路。

### 关注 Agent 应用 / MCP 设计

`00 → 05 → 08 → 09 → 10`

5 篇，覆盖三角色 Agent + RAG + Copilot SSE + MCP。

### 关注工程化交付能力

`00 → 11 → 12 → 13 → 14`

5 篇，覆盖存储 / 配置 / 编排 / 测试 QA。

### 关注长视频处理优化

`00 → 02 → 03 → 06 → 05`

5 篇，覆盖 VLM 分级 + 长度路由 + Map-Reduce + Critic-Reviser。

### 想准备「项目里最有挑战的部分」面试问题

`04 → 06 → 07`：双投影 + Map-Reduce + 数据建模

或

`03 → 05 → 12`：长度路由 + Agent 协作 + 降级矩阵

## 文档约定

- **代码引用**全部用 `@d:\Diet_Agent_NEW\<file>:<line>` 格式，可点击跳转。
- **配置项**直接列 `LECTURE_REVISER_MODE=off` 这种 env 名而不是 Python 字段名。
- **降级路径**显式标注「失败时降级到什么」「由什么开关控制」。
- 每篇结尾都有「写简历可以怎么提炼」+「面试可展开点」+「事实与推断」三段。

## 与 Internship/01~05 的关系

- `Internship/01~04` 是按主链路 4 段（采集 / 讲义生成 / Copilot / MCP）做的项目总览。
- `Internship/05` 是简历与面试提纲。
- `Internship/Detail/` 在前 4 篇基础上**深入到代码**，去掉笼统描述，加入具体类 / 函数 / 配置项 / 降级路径。

如果只读 5 篇总览，能对外讲清楚项目立意。读完 16 篇 Detail，能对内讲清楚每一处实现选择。**面试前重点看 `15-面试讲述与预期问答`**。
