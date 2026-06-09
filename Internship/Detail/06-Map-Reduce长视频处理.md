# Map-Reduce 长视频处理

## 文档定位

`Internship/02` 提到了「长视频走 map-reduce」，本篇展开 `app/understand/ir_map_reduce.py` 全文件，覆盖：

- 为什么 60+ 分钟视频不能一次 LLM 调用？
- map / reduce_local / reduce_global 各做什么？
- 缓存命中怎么和章节绑定？
- 单 chapter 失败、global 失败分别怎么处理？

主文件 `app/understand/ir_map_reduce.py` 共 1150 行，是项目第二复杂的模块。

## 一句话定位

`MapReduceIRBuilder` 是长 / epic profile 的 IR 构建器。**每个章节一次 LLM 调用 → 纯 Python 局部合并 → 一次轻量全局 pass**，三阶段共同产出一份合法的 `LectureIR`。

## 为什么需要 Map-Reduce

`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:1-31` 的 docstring 给出动机：

- 单次 IR 构建在 30 分钟视频以下都能塞进 prompt，但 60+ 分钟视频会撞 LLM context 窗口。
- 实际证据：23 分钟代码密集视频 BV1ypdgBCE9B 已经触发 DeepSeek `finish_reason=length` 截断。配置 `LECTURE_PROFILE_THRESHOLDS_SEC` 因此把 standard 上界从 25 分钟降到 15 分钟，让 15+ 分钟直接走 Map-Reduce。

设计目标三条不可妥协：

- **可缓存** — 每个 map 调用按 `compute_prompt_hash` 寻址，二跑时几乎免费。
- **resilient** — 单章失败用占位符（`unverified=True`）顶替，不阻塞整条流水线；占位符**不写缓存**让下次还能重试。
- **fail-fast on global** — 全局 pass 失败抛 `ReduceGlobalError`，**整次构建放弃**，但章节缓存保留让重跑只重做全局阶段。

## 三阶段总览

`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:297-384` 的 `build()` 协程串起来：

1. `_collect_chapter_dicts(chapter_plan, segments, frames, meta, stats)` —— **Map 阶段**。
2. `reduce_local(chapter_dicts, meta)` —— **Reduce-Local 阶段**，纯 Python。
3. `reduce_global_pass(local_ir, glossary_conflicts, meta, study_questions, stats)` —— **Reduce-Global 阶段**，一次轻量 LLM 调用。
4. 顶层 metadata 注入 + `sanitize_ir_data` + `LectureIR.model_validate`。

返回 `(LectureIR, MapReduceStats)`，stats 含 `map_calls / cache_hits / cache_writes / map_failures / map_total_sec / reduce_local_sec / reduce_global_sec / reduce_global_attempts / ir_sanity`。

## Map 阶段：每章一次 LLM 调用

### 章节切片

`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:390-469` 的 `_collect_chapter_dicts`：

- 按 `start_sec` 排序 anchors，保证章节顺序确定。
- 对每个 anchor，用 `_segments_in_window(segments, start, end)` 和 `_frames_in_window(frames, start, end)` 切窗口。
- 调 `compute_prompt_hash(...)` 算 SHA-256，命中缓存就直接用，未命中就跑 `_map_chapter_with_retry`。
- 占位符（`failure != None` 的 payload）**不**写缓存。
- 成功 payload 通过 `chapter_cache.put(...)` 写回。

`_stamp_chapter(payload, anchor, chapter_index)` 把 LLM 输出的章节内容戳上 `chapter_index / start / end`，让 LLM 不能改这三个字段。

### 单章 prompt：`LECTURE_IR_MAP_CHAPTER_*`

- system prompt `LECTURE_IR_MAP_CHAPTER_SYSTEM` — 强约束「只填本章字段，不要写 mainline / final_synthesis / knowledge_units 这些全局字段」。
- user prompt `LECTURE_IR_MAP_CHAPTER_USER_TEMPLATE` — 含 `{meta_block}{chapter_window}{segments_block}{frames_block}`。

输出 JSON 只含一个章节的字段：`title / summary / learning_goal / teaching_notes / process_steps / points / code_blocks / formula_blocks / pitfalls / key_takeaways` 等。

### 重试策略：`_map_chapter_with_retry`

`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:471-560` 实现「最多 `max_retries + 1` 次尝试」：

- 调用时显式传 `max_tokens=LECTURE_MAP_CHAPTER_MAX_TOKENS`，避免落到 DeepSeek 默认 4096 导致输出截断。
- 解析失败 / 网络失败 → 按指数退避重试。
- 检测到 `finish_reason == 'length'` → 抛 tagged `RuntimeError`，**短路重试**（再试也会截断），直接占位符。
- 占位符 payload 含 `_lm_placeholder` 标记和「本章生成失败」标题，summary 会被改写成可操作建议（「降低 LECTURE_CHAPTER_MAX_DURATION_SEC」）。

### MapFailure：M2.2 telemetry 升级

`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:84-131`：

```python
@dataclass(frozen=True)
class MapFailure:
    chapter_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    subtitle_count: int       # 输入字幕段数
    frame_count: int          # 输入帧数
    attempts: int             # 实际尝试次数
    finish_reason: str        # 'length' / 'stop' / ''
    error_class: str          # TimeoutError / ValueError / ...
    error_excerpt: str        # 前 240 字
```

设计动机（docstring 显式）：M2.2 之前 `map_failures` 只是 `tuple[int]`，知道章节 index 但不知道为什么挂。运维拿到 telemetry 也只能猜「是不是网络？是不是 prompt？是不是 token 截断？」。新结构带上完整诊断上下文，可以直接从矩阵报告里分类原因。

`MapReduceStats.failure_indices` 保留 legacy `tuple[int]` 形态，让老的日志解析器 / dashboard 还能用。

## Reduce-Local：纯 Python 局部合并

`reduce_local(chapter_dicts, meta)` 不调 LLM，逻辑全部确定性：

- 把 map 阶段产出的每章 dict 直接放进 `local_ir["chapters"]`。
- 各章节的 `knowledge_unit_ids` 暂时收集但**不去重 / 不合并**——这一步留给 global pass。
- 收集 glossary 冲突：同一术语在不同章节有不同 explanation → 加入 `glossary_conflicts` 列表，喂给全局 pass。
- 输出 `local_ir` 这个**半成品 dict**（不是 LectureIR）：缺少 `mainline / final_synthesis / cross_references` 等全局字段，等下一步填。

## Reduce-Global：一次轻量 LLM pass

`reduce_global_pass(local_ir, glossary_conflicts, meta, study_questions, stats)` 是整条 map-reduce 链路里**唯一调用全局上下文的 LLM 调用**：

- prompt system `LECTURE_IR_REDUCE_GLOBAL_SYSTEM` — 「不要改任何章节内容；只填 mainline / final_synthesis / knowledge_units / cross_references / completeness / timeline 这几个全局字段」。
- prompt user 含已经存在的 chapters（仅 `title / summary / key_takeaways`）+ glossary_conflicts + study_questions。

### 重试 + Fail-fast 设计

如果全局 pass 失败 → 重试 `LECTURE_REDUCE_GLOBAL_MAX_RETRIES` 次（默认 2-3 次）→ 全部失败 → 抛 `ReduceGlobalError`，**整次 build 放弃**。

设计理由（`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:176-183`）：

> 章节缓存保留 → 重跑只重做全局 pass，~30 秒在 epic 视频上不大。
> 强行降级（例如用第一章 summary 拼一个 mainline）会让产物质量不可控，与「fail-fast」的工程原则相悖。

## ChapterCache 怎么挂进来

```python
def __init__(self, *, client, settings, profile, chapter_cache: ChapterCache):
    ...
    self._cache = chapter_cache
    self._model = getattr(settings, "qwen_text_model", "")
```

- 构造时**注入**一个 `ChapterCache`，让测试可以传 mock cache。
- 缓存 key 的 `model_id` 字段就是 `self._model`，**模型升级自动废老缓存**。
- 缓存 key 的 `prompt_version` 字段取 `settings.lecture_map_prompt_version`（默认 `"m2-map-v1"`），**prompt 模板升级时手动 bump 这个版本号**，立刻让历史缓存全部 miss。

## 顶层兜底：sanitize + fallback mainline

`@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:355-372` 在 `LectureIR.model_validate` 前还有两道防御：

```python
sanity = sanitize_ir_data(local_ir)
if not (local_ir.get("mainline") or []):
    local_ir["mainline"] = self._fallback_mainline_from_chapters(local_ir.get("chapters") or [])
    if local_ir["mainline"]:
        local_ir["completeness"]["mainline_closed"] = True
```

- `sanitize_ir_data` 与单次构建路径完全一致（Q1-Q4 修复）。
- 全局 pass 漏写 `mainline` 时，自动用每章 title 拼一个简易 mainline，避免 IR 缺失关键字段。

## Stats 数据形态

成功跑完的 `MapReduceStats`：

```python
MapReduceStats(
  map_calls=8,                  # 8 章 → 8 次潜在调用
  cache_hits=6,                 # 二跑命中了 6 章
  cache_writes=2,                # 新写入 2 章
  map_failures=(),               # 没有失败
  map_total_sec=82.4,            # 8 章累积 wall-clock
  reduce_local_sec=0.05,         # 纯 Python，毫秒级
  reduce_global_sec=29.8,        # 全局 LLM 一次
  reduce_global_attempts=1,      # 一次过
  ir_sanity={...}                # Q1-Q4 修复报告
)
```

Pipeline 在 `@d:\Diet_Agent_NEW\app\pipeline.py:267-271` 把这个 dict 一字不差挂到 `timing.map_reduce`，矩阵脚本 / 验证脚本能直接读。

## 一个具体的成本对比（epic 视频）

以一段 90 分钟视频为例（12 章）：

- 首跑：12 次 map 调用（~10s 每次） + 1 次 reduce_global（~30s） + 0 次 local（毫秒）= **~150s LLM 时延**。
- 二跑（仅 prompt_version 没变，模型也没变）：0 次 map + 1 次 reduce_global = **~30s**。
- 三跑（任一帧的 OCR 改了）：受影响的 chapter 重做 + reduce_global = **~40s**。

这是为什么二跑要做到「重做全局 pass 后基本就完事」的精确动机。

## Pipeline 集成（P7）

主调度在 `@d:\Diet_Agent_NEW\app\pipeline.py:227-244`：

```python
use_agents = bool(settings.lecture_question_driven or settings.lecture_critic_enabled)
if use_agents:
    lecture_ir, stats = await self.ir_builder.build_with_agents(
        ctx,
        profile=profile,
        chapter_plan=chapter_plan,
        chapter_cache=chapter_cache_obj,
    )
```

`build_with_agents` 内部按 `profile.use_map_reduce` 分流：

- `False`（tiny / standard）→ 走单次 IR 构建 + Critic / Reviser。
- `True`（long / epic）→ 走 `MapReduceIRBuilder(...).build(...)` + Critic / Reviser。

详细分流逻辑在 `app/understand/ir_builder.py` 的 `build_with_agents` 主循环（文件 65k，本篇不展开，主要参见 `Detail/05`）。

## Auto-fallback：单次构建截断 → Map-Reduce

`_TRUNCATION_ERROR_TAG = "LLM truncated LectureIR JSON"`（`@d:\Diet_Agent_NEW\app\understand\ir_builder.py:37`）是一个特殊标记。当单次 IR 构建因 `finish_reason=length` 失败时，`build_with_agents` 的外层 catch 检测到这个 tag 会**自动 fallback 到 Map-Reduce 路径**，避免「我以为 15 分钟够 standard 但实际太密集」这种边界情况让用户直接看到失败。

这是「profile 路由 + 实际 truncation 兜底」的双重保险。

## 写简历可以怎么提炼

- 实现了长 / epic profile 的 Map-Reduce LectureIR 构建器：每章一次 LLM 调用、纯 Python 局部合并、一次轻量全局 pass，把 60+ 分钟视频的单次 prompt 大小压回 LLM context 窗口内。
- 通过 `compute_prompt_hash`（含 `model_id × prompt_version × 规范化 chapter 输入`）+ `ChapterCache` SQLite，让长视频重跑只需要重做全局 pass（~30s），首跑后接近免费。
- 设计了「单章失败用占位符 + 不入缓存 + Map-failure telemetry，全局失败直接抛 ReduceGlobalError」的混合策略，做到「局部容错」与「全局 fail-fast」并存。
- 为单次 IR 截断设计了基于 `_TRUNCATION_ERROR_TAG` 的自动 fallback，让用户即便在 profile 边界情况下也不会看到失败。

## 面试可展开点

### 1. 为什么 reduce_local 完全不调 LLM

Map 阶段已经把每章内容生成好了，local reduce 只做「拼接 + 收集冲突」。让 LLM 干这种纯结构操作是浪费 token 和时间，还会引入幻觉。

### 2. 为什么 reduce_global 不和 reduce_local 合并

Reduce-local 处理 chapter-level dict 列表（拼接）；reduce-global 用 LLM 做 lecture-level 推理（mainline / final_synthesis / glossary 合并）。两阶段语义不同，且 reduce-global 是唯一会失败的步骤——分开能让 reduce-local 的成本完全可预测。

### 3. 为什么不缓存 reduce_global 的输出

Reduce-global 的输入包含**所有 chapter map 的输出**——只要任一章节变了，全局结果也会变。要缓存就得算「所有章节输出的联合 hash」，这等价于一次完整 build 输入的 hash，缓存粒度太粗，命中率低。30 秒的成本不如直接每次重跑。

### 4. 占位符为什么不写缓存

占位符代表「LLM 失败」，而不是「这一章应该长这样」。如果写缓存，二跑会直接拿到占位符，永远走不出失败状态。**不缓存失败结果**是缓存系统的通用原则。

### 5. truncation tag 为什么要做特殊处理

Truncation 不是「网络抖动」也不是「prompt 写错」，重试再多遍也只会再次截断。把它显式识别出来短路重试链，能避免「3 次重试 × 60s 每次」的浪费。截断信号要么改 max_tokens 要么改章节宽度，与重试无关。

## 当前文档中的事实与推断

### 事实

- `MapReduceIRBuilder.build()` 三阶段定义在 `@d:\Diet_Agent_NEW\app\understand\ir_map_reduce.py:297-384`。
- `MapFailure` 在 M2.2 升级，含 10 个诊断字段。
- 单章失败占位符不写缓存（spec §3.3.1）。
- 全局 pass 失败抛 `ReduceGlobalError`，整次 build 失败但 chapter cache 保留。
- `LECTURE_MAP_PROMPT_VERSION` 变化会废所有 map 缓存；模型变化也会。
- 单次构建截断有 auto-fallback 到 Map-Reduce 的路径。

### 推断

- 这一层是项目里「Agent 应用工程化」最有亮点的部分：缓存粒度、失败传播、telemetry 三个维度的设计都比常见的 toy demo 严肃得多。
