# 多 Agent 构建：StudyQuestion / Critic / Reviser

## 文档定位

`Internship/02` 提到了「多 Agent 协作生成讲义」，但没说清三个 Agent 各自做什么、怎么衔接、Reviser patch 模式具体是什么。本篇拆到模块：

- `app/understand/agents.py`（56905 字节）— 三个 Agent 类。
- `app/understand/critic_context.py`（9179 字节）— 全 / 投影两种 Critic 输入打包。
- `app/understand/ir_patches.py`（11754 字节）— 白名单结构化补丁。
- `app/understand/ir_builder.py`（65157 字节）— 主调度，负责把三个 Agent 串起来。

## 一句话定位

LectureIR 不是一次 LLM 调用直出来的。它经过「Question-Driven 预生成 → 初始 IR 构建 → Critic 审查 → Reviser 打补丁」四步循环，每一步失败都有降级路径，每一步产物都可被 telemetry 看到。

## 三角色分工

| Agent | 输入 | 输出 | 默认模式 | 失败影响 |
|-------|------|------|----------|---------|
| **StudyQuestionAgent** | 标题 + 多窗口字幕 + 多窗口关键帧 | 5-10 个学习问题 | 默认开 | 降级为空列表，不阻断 |
| **Critic** | LectureIR + 字幕 + 帧 + study_questions | `CritiqueResult{verdict, issues, metrics}` | 默认开 | 降级为「audit skipped」字符串，不阻断 |
| **Reviser** | Critic.issues + LectureIR | `IRPatch[]` 或全 IR 重写 | 默认 patch 模式 | 降级为 noop，IR 不动 |

主调度入口是 `LectureIRBuilder.build_with_agents(ctx, profile, chapter_plan, chapter_cache)`，定义在 `@d:\Diet_Agent_NEW\app\understand\ir_builder.py` 较深处（文件 65k 字节，是项目最复杂的单文件）。

## StudyQuestionAgent：Question-Driven Extraction

### 为什么要预生成问题

`@d:\Diet_Agent_NEW\app\understand\agents.py:1-17` 的 docstring 解释：直接让 IR 提取模型「读完字幕自己列要点」容易塌缩到「介绍背景 / 讲解核心 / 总结」这种通用框架。**先让一个轻量 Agent 生成 5-10 个有针对性的问题**，再把这些问题作为 IR builder 的 prompt 软约束，能强制结构化抽取关注「视频真正回答了什么」。

### 多窗口采样（plan §10 N4）

`@d:\Diet_Agent_NEW\app\understand\agents.py:149-223` 的 `_study_question_windows(duration, multi_window=True)` 按时长选窗口：

- **< 10 min** 或 `multi_window=False`：单一头部窗口（兼容 legacy）。
- **10-30 min**：开头 4min + 中段 3min + 尾声 2min。
- **30-60 min**：开头 4min + 中段-1 3min + 中段-2 3min + 尾声 3min。
- **>= 60 min**：开头 4min + 三个中段各 2min + 尾声 3min。

每个窗口低于 30 秒会被丢弃，避免出现空窗口。这一段策略由 `LECTURE_STUDY_QUESTION_MULTI_WINDOW=true|false` 控制；profile.study_question_mode 是 `head_only` 时强制单窗口。

### 关键帧加权采样

`@d:\Diet_Agent_NEW\app\understand\agents.py:255-306`：每窗口最多 2 帧，选择策略：

- 优先级 1：`visual_type ∈ {code, formula, diagram}`（`_CRITIC_HIGH_VALUE_VISUAL_TYPES`）。
- 优先级 2：`importance_score` 降序。
- 优先级 3：`timestamp` 升序（同 score 时按时间）。
- 跨窗口全局去重（按 `path`）。

这是「让 prompt 在有限 token 内拿到最有信息量的帧」的典型实现。

### 后置覆盖度自检

`@d:\Diet_Agent_NEW\app\understand\agents.py:335-376` 的 `_study_question_coverage_warnings` 给问题集做覆盖度检查：

- 只对 ≥ 10 分钟视频做。
- 把字幕拆成两半，看每个 question 的关键词是否在「后半字幕」里出现过。
- 匹配率 < 30% → emit `second_half_keyword_ratio_low ratio=X.XX` 警告。
- **不**改 question 列表，纯诊断。

这条警告会进 `StudyQuestionsResult.warnings`，最终落到 `pipeline_stats`，矩阵报告能识别「这次跑出来的问题都集中在开头，没覆盖后半视频」。

### `_question_match_terms`：中英混合关键词抽取

`@d:\Diet_Agent_NEW\app\understand\agents.py:309-332`：

- 拉丁字符 + 数字 ≥ 3 字符 → 整段保留（`QKV` / `softmax`）。
- CJK 连续段 ≤ 3 字符 → 整段保留。
- CJK 连续段 > 3 字符 → 用 3 字符滑窗（`正则化` / `则化在` / ...）。

不追求完美分词，专门给上面的覆盖度自检用。

## Critic：审查 + 验证

### Critic 的职责（**不**重写 IR）

Critic 只输出一份审查结果：

```python
@dataclass
class CritiqueResult:
    verdict: str            # "pass" / "needs_revision" / "fail"
    issues: list[CritiqueIssue]
    metrics: dict           # 各种统计
    usage: dict             # token 计数
    raw: dict               # 原始 LLM 输出
```

每个 `CritiqueIssue` 含 `severity / kind / location / description`。`location` 通常是 `"chapters[3].points[2]"` 这种 path-like 字符串，下游 Reviser 用它定位补丁路径。

### CriticContext：full vs projected 两种打包

`@d:\Diet_Agent_NEW\app\understand\critic_context.py:196-243` 的 `build_critic_context(profile, ir, segments, frames, study_questions, chapter_issue_digest)` 根据 profile 选两种模式：

- **`full` 模式**（standard profile）：完整 IR + 全字幕 + 全帧 + study_questions，`skip_quote_validation=False`。Critic 能逐字检查 `point.quote` 是否真在字幕里。
- **`projected` 模式**（long / epic profile）：只投影 IR 的 `lecture_summary / mainline / glossary / cross_references` 和每章的 `title / start_sec / end_sec / summary / points_count / code_blocks_count / unverified`。**字幕和帧整体丢掉**，`skip_quote_validation=True`。

`_project_chapter` 把 `points` 替换成 `points_count`，是为了**禁止 `point.quote` 字符串泄漏到 projected prompt**（一旦泄漏，模型会在 `skip_quote_validation=True` 的语义下幻觉 false-positive）。

### 调用 Critic

`CriticReviserAgent.audit(context: CriticContext)` 根据 `context.mode` 选 prompt 模板：

- `LECTURE_CRITIC_FULL_USER_TEMPLATE` — 含字幕摘录 + 全帧 + 完整 IR。
- `LECTURE_CRITIC_PROJECTED_USER_TEMPLATE` — 只含投影 IR。
- system prompt 共享 `LECTURE_CRITIC_SYSTEM`。

Critic 超时单独 600s（`LECTURE_REVISER_TIMEOUT=600`，复用相同的「贵 LLM 调用」timeout）。

## Reviser：白名单结构化补丁

### 为什么不再做「全 IR 重写」

`@d:\Diet_Agent_NEW\app\understand\ir_patches.py:1-23` 的 module docstring 列了三个问题：

- **时延** — full Reviser 在长视频上常常 200-600s，即使 Critic 只要求改一行 quote。
- **静默回归** — full Reviser 偶尔会清空 `code_blocks` 这种重要数组，旧代码靠 `_restore_dropped_content` 兜底。
- **审计性** — full Reviser 是黑盒重写，没法把成本归因到具体修改。

M2.P6 引入 patch Reviser 解决这三个问题。

### 5 种允许的 op

`@d:\Diet_Agent_NEW\app\understand\ir_patches.py:40-50`：

```python
PatchOp = Literal[
    "replace_array",
    "append",
    "set_quote",
    "set_text",
    "set_explanation",
]
```

### 正则白名单（**路径必须在表里**）

`@d:\Diet_Agent_NEW\app\understand\ir_patches.py:119-164`：

- `replace_array` 允许的路径：
  - `/chapters/{i}/{code_blocks|formula_blocks|pitfalls|key_takeaways|process_steps}`
  - `/{knowledge_units|study_questions|review_questions|mainline}`
- `append` 是同样的路径加 `/-` 后缀。
- `set_quote` 只允许 `/chapters/{i}/points/{j}/quote` 和 `/knowledge_units/{j}/quote`。
- `set_text` 只允许 `/chapters/{i}/points/{j}/text` 和 `/knowledge_units/{j}/title`。
- `set_explanation` 只允许 `/chapters/{i}/code_blocks/{j}/explanation` / `formula_blocks/{j}/explanation` / `knowledge_units/{j}/explanation`。

**任何 spec §3.6 禁区（profile、taxonomy、chapter start_sec、point ts...）都不在白名单里**，所以 LLM 即便提议也会被自动拒收。

### IRPatch + RejectedPatch

```python
@dataclass(frozen=True)
class IRPatch:
    op: str
    path: str
    value: Any | None = None       # 给 replace_array / append
    quote: str | None = None        # 给 set_quote
    text: str | None = None         # 给 set_text / set_explanation

@dataclass(frozen=True)
class RejectedPatch:
    patch: IRPatch | None
    reason: str
```

`value / quote / text` 分离成不同字段而不是单一 polymorphic 字段，是因为「LLM 看 prompt 例子时知道该填哪个 slot」更明确。

### apply_patches：原子语义 + 整批回滚

`@d:\Diet_Agent_NEW\app\understand\ir_patches.py:279-328`：

```python
def apply_patches(ir, patches) -> tuple[LectureIR, list[RejectedPatch]]:
    rejected = []
    payload = copy.deepcopy(ir.model_dump())
    for patch in patches:
        try:
            _validate_op_and_path(patch)   # 白名单
            _apply_one(payload, patch)     # 走 JSON-Pointer 解析 + 修改
        except PatchRejected as exc:
            rejected.append(RejectedPatch(patch=patch, reason=str(exc)))

    try:
        revised = LectureIR.model_validate(payload)
    except ValidationError as exc:
        # schema 违规 → 整批回滚到原 IR
        rejected.append(RejectedPatch(patch=None, reason=f"schema_violation:{...}"))
        return ir, rejected
    return revised, rejected
```

两个关键设计：

- **逐补丁拒收，但不影响其他补丁继续应用**。
- **最终 Pydantic 校验失败 → 整批回滚**。设计理由（docstring 显式说明）：LLM 把多个补丁视为一个连贯的修复计划，只应用一半可能让 IR 状态比原始还差（例如清空了 `code_blocks` 但保留了对它的引用）。

### `_apply_one`：JSON-Pointer 风格路径解析

`@d:\Diet_Agent_NEW\app\understand\ir_patches.py:186-271`：

- `/chapters/0/code_blocks` → tokens `["chapters", "0", "code_blocks"]`。
- 路径中遇到 list → 把 token 转 int，越界就抛。
- 路径中遇到 dict → 找 key，缺失就抛。
- 末端 `/-` 表示 append 到 list 尾。

错误类型用机器可读 tag：`unknown_op:set_severity` / `path_not_in_whitelist:/profile/primary_type` / `path_index_oob:5` / `path_traversal_failed_at:summary`。

### `revise_patch` vs `revise`（legacy）

`agents.py` 里同时保留：

- `CriticReviserAgent.revise_patch(*, context, issues)` — M2.P6 主推路径。
- `CriticReviserAgent.revise(...)` — legacy 全 IR 重写，由 `LECTURE_REVISER_MODE=full` 触发。

Profile 表里所有非 tiny 档位的 `reviser_mode` 都是 `"patch"`，默认就用 patch 路径。

## ir_builder 的主循环

`LectureIRBuilder.build_with_agents` 把这三个 Agent 串起来：

1. **StudyQuestion 阶段** — 调 `StudyQuestionAgent.generate(ctx, profile)`，失败降级为空列表。结果作为后续 prompt 的 hint。
2. **Initial IR 阶段** — 调 `_build_initial_ir`，调用 LLM 一次产出 LectureIR JSON。失败传播或回退到 v1（取决于 `LECTURE_STRICT_AGENTS`）。
3. **Critic / Reviser Loop** — `_run_critic_loop(ir, ctx, profile, ...)`：
   - 若 `profile.critic_mode == "off"` → 跳过。
   - 否则 `build_critic_context(profile, ...)` → `CriticReviserAgent.audit(context)`。
   - Critic 返回 `verdict="pass"` → 结束。
   - `verdict="needs_revision"` 且 `profile.reviser_mode == "patch"`：
     - 调 `CriticReviserAgent.revise_patch(context, issues)` 拿 `RevisePatchResult`。
     - `apply_patches(ir, result.patches)` 拿新 IR + rejected 列表。
     - 把 Critic / Reviser 信息塞 `pipeline_stats`。
   - `reviser_mode == "full"`：调旧 `revise()`，重写整份 IR。
   - `reviser_mode == "off"`：noop。

整个循环不做多轮（spec 决策 `max_rounds = 0/1`）。一次 Critic + 一次 Reviser 已经覆盖 90% 实际问题，多跑只会增加成本。

## 失败传播规则

每个 Agent 都遵循同一个降级模式：

- **`LECTURE_STRICT_AGENTS=true`**（默认 false）：任何 Agent 抛异常都重新抛出，让 Pipeline 决定是否走 v1 fallback。
- **`LECTURE_STRICT_AGENTS=false`**：StudyQuestion 失败 → 空列表；Critic 失败 → `verdict="fail"` 但不抛；Reviser 失败 → IR 不动。

这一层让「单个 LLM 调用挂掉」不会传染整条流水线。

## Telemetry 出口

`pipeline_stats` 在 `_run_critic_loop` 里被这样填充（`@d:\Diet_Agent_NEW\app\pipeline.py:261-282`）：

```python
if isinstance(stats, dict):
    if "map_reduce" in stats:
        timing["map_reduce"] = stats["map_reduce"]
    if "reviser" in stats:
        timing["reviser"] = stats["reviser"]
    if isinstance(stats.get("critique"), dict):
        timing["critic"] = {
            "verdict": stats["critique"].get("verdict"),
            "issue_count": len(stats["critique"].get("issues") or []),
            "metrics": stats["critique"].get("metrics") or {},
        }
```

`reviser` 的内容长这样：

```python
{
  "mode": "patch",
  "proposed": 6,        # LLM 返回了多少 patch
  "applied": 4,         # apply_patches 留下多少
  "rejected": 2,        # 拒收数（按 reason 分类的细节也在）
  "unfixable_issues": ["..."],
  "tokens": {"input": ..., "output": ...},
}
```

矩阵脚本可以直接读这些指标，回答「Reviser 真的修了多少东西」。

## 写简历可以怎么提炼

- 设计了 Question-Driven extraction：用一个轻量 StudyQuestionAgent 预生成 5-10 个学习问题（多窗口采样 + 关键帧加权），再把它们作为后续 IR 构建的 prompt 软约束，避免长视频 IR 塌缩到通用框架。
- 实现了 Critic 的 full / projected 两种输入打包：标准视频提供完整 IR + 字幕 + 帧让其逐字校验引用；长视频投影到「lecture summary + chapter shell」节省 token，并强制 `skip_quote_validation` 不让模型在缺源的语境下幻觉。
- 用一组正则白名单 + 5 种 op（`replace_array / append / set_quote / set_text / set_explanation`）把 Reviser 从「全 IR 重写」改造成「结构化补丁」，并通过 `apply_patches` 原子语义 + 整批回滚保证修补后的 IR 必然通过 Pydantic 校验。
- 引入了完整的 telemetry（`critic.verdict / metrics / issue_count`、`reviser.proposed / applied / rejected / unfixable_issues`），让 Critic-Reviser 链路的成本与收益可量化、可回归。

## 面试可展开点

### 1. 为什么 Critic 不直接改 IR

Critic 是审查角色，Reviser 是修补角色，**职责分离让 Critic 的回答更短、更稳定**。Critic 只输出「这里有问题」，不必同时思考「怎么改」。让一个模型既审又改，往往会让审查变得「自我辩护」，错漏问题。

### 2. 为什么 long / epic 用 projected 模式

完整 IR + 全字幕 + 全帧塞进 prompt 后，对 60 分钟视频常常超过 100k token，单次 Critic 调用可能跑 120-180 秒，触发 LLM 的 context 上限。投影到「lecture-level shell」后，prompt 通常 < 20k token，调用稳定 30 秒内。代价是不能逐字校验 quote，但这本来在长视频上就靠 IR builder 内部的 fuzzy match 兜底了。

### 3. 为什么 Reviser 要做白名单

Reviser 修补只想动一小块（quote / explanation / 单个 chapter 的 code_blocks）。如果允许任意 path，LLM 会偶尔「顺手」改 chapter timing / profile / taxonomy，导致下游静默回归。白名单把语义边界固化在代码里，**LLM 提议什么不重要，能落地的只有白名单内**。

### 4. apply_patches 为什么整批回滚而不是丢失败的那条

LLM 给的多个补丁是一个连贯的修复计划。例如 Critic 说「chapter 3 的 code_blocks 缺了 explanation 而且 set_quote 引用了错的话」，Reviser 同时给「set_explanation」和「set_quote」两条补丁。如果只回滚其中一条，最终 IR 可能比原始版本还混乱。整批回滚保证「要么整套补丁通过 Pydantic + 上线，要么一字不动」。

### 5. 为什么 `_apply_one` 的错误信息要写成 `unknown_op:set_severity` 这种格式

机器可读 tag 让 telemetry 能按 `reason.split(":")[0]` 分组统计。运维报告会显示「这一周拒收的补丁 60% 是 unknown_op，30% 是 path_not_in_whitelist」，比一句自由文本更可量化。

## 当前文档中的事实与推断

### 事实

- 三个 Agent 类都定义在 `app/understand/agents.py`。
- `CriticContext` 有 `full / projected` 两种 mode，`projected` 强制 `skip_quote_validation=True`。
- Patch Reviser 白名单是 5 种 op + 一组正则路径（`@d:\Diet_Agent_NEW\app\understand\ir_patches.py:119-164`）。
- `apply_patches` 在 Pydantic 校验失败时整批回滚。
- 失败时降级语义由 `LECTURE_STRICT_AGENTS` 控制。

### 推断

- 这一层是项目「让 LLM 修自己写的东西」最成熟的实现——Critic / Reviser 两阶段 + 白名单 + 投影 + telemetry，是任何「LLM 输出再加工」场景都可以借鉴的模式。
