# Pipeline：端到端编排

## 文档定位

`Internship/01-04` 各自讲了某一层，本篇把 `app/pipeline.py`（388 行）作为「主链路指挥中心」做一次端到端复盘。其他 Detail 文档讲的是模块本身，本篇讲的是这些模块怎么**被一份代码串起来**、telemetry 怎么落、降级怎么传播。

## 一句话定位

`Pipeline._run_inner` 是 LectureMind 从 URL 到 HTML 的唯一主路径。它把 ingest / vlm / profile / chapter_plan / ir_builder / renderer / db / rag 8 个阶段串联起来，每一步都写 `last_run_timing` + `last_run_stats`，并且**任何阶段都不会被强迫执行**——一切是按配置和返回值决定的。

## 类结构

`@d:\Diet_Agent_NEW\app\pipeline.py:38-58`：

```python
class Pipeline:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.settings = get_settings()
        self.bili = BilibiliIngest()
        self.covers = CoverCache()
        self.subtitles = SubtitleExtractor()
        self.keyframes = KeyframeExtractor()
        self.vlm = FrameDescriber()
        self.ir_builder = LectureIRBuilder()
        self.lecturizer = Lecturizer()         # v1 fallback
        self.renderer = Renderer()
        self._concurrency = asyncio.Semaphore(self.settings.max_concurrent_jobs)
        self.rag = RAGStore(db.db_path)
        self.last_run_timing: dict[str, Any] = {}
        self.last_run_stats: dict[str, Any] = {}
        self.last_run_rag_task: asyncio.Task[Any] | None = None
```

构造一次，所有 ingest/render/rag 实例都准备好。Pipeline 跨多个 `/api/summarize` 共享，但每次 `run()` 都会重置 `last_run_*`。

## 入口：run（外） + _run_inner（内）

`@d:\Diet_Agent_NEW\app\pipeline.py:60-69`：

```python
async def run(self, url, *, force_refresh=False, progress_cb=None) -> Path:
    async with self._concurrency:
        return await self._run_inner(url, force_refresh=force_refresh, progress_cb=progress_cb)
```

`_concurrency` semaphore 把全局 pipeline 并发限制在 `MAX_CONCURRENT_JOBS`（默认 2）。`progress_cb(percent, msg)` 是可选回调，FastAPI 用它向前端推送 SSE 进度。

## 主流程：8 个阶段

### 阶段 0：进度回调 + telemetry 重置

`@d:\Diet_Agent_NEW\app\pipeline.py:72-84`：

```python
async def _progress(p: int, msg: str = "") -> None:
    logger.info("[%d%%] %s", p, msg)
    if progress_cb:
        await progress_cb(p, msg)

timing: dict[str, Any] = {}
self.last_run_timing = timing
self.last_run_stats = {}
self.last_run_rag_task = None
```

每次 run 都拿一个新的 `timing` dict，挂在 `self.last_run_timing` 上让验证脚本读。**verifier 直接读这个 dict，不用解析日志**。

### 阶段 1：metadata（5%）

`@d:\Diet_Agent_NEW\app\pipeline.py:86-90`：

```python
await _progress(5, "fetching metadata")
t_meta = time.perf_counter()
meta: VideoMeta = await self.bili.fetch_meta(url)
timing["metadata_sec"] = round(time.perf_counter() - t_meta, 3)
```

只这一步是必须 sequential 的——后面所有阶段都依赖 `meta`。

### 阶段 2：缓存命中检查

`@d:\Diet_Agent_NEW\app\pipeline.py:93-100`：

```python
if not force_refresh:
    existing = await self.db.get_summary(meta.bv_id)
    if existing and existing.status == "done":
        report_path = self.settings.data_dir / existing.report_path
        if report_path.exists():
            timing["cache_hit"] = True
            await _progress(100, "cache hit")
            return report_path
```

`status="done"` 且报告文件还在 → 直接返回，**跳过所有后续阶段**。这是 Pipeline 最大的快路径。`force_refresh=True` 可绕过。

### 阶段 3：subtitle + keyframes + cover 并行（15%）

`@d:\Diet_Agent_NEW\app\pipeline.py:102-133`：

```python
async def _timed_subtitle() -> tuple[Any, float]:
    t = time.perf_counter()
    res = await self.subtitles.extract(meta.bv_id, meta.aid, meta.primary_cid)
    return res, time.perf_counter() - t

# 类似 _timed_keyframes / _timed_cover

(sub_result, subtitle_sec), (frames, keyframes_sec), (cover_path, cover_sec) = (
    await asyncio.gather(_timed_subtitle(), _timed_keyframes(), _timed_cover())
)
timing["subtitle_sec"] = round(subtitle_sec, 3)
timing["keyframes_sec"] = round(keyframes_sec, 3)
timing["cover_sec"] = round(cover_sec, 3)
timing["ingest_parallel_sec"] = round(time.perf_counter() - t_ingest, 3)
timing["subtitle_source"] = sub_result.source  # "cc" or "whisper"
timing["keyframes_count"] = len(frames)
```

三件事并行，wall-clock 时间 ≈ max(三者)，详见 `Detail/01-Ingest`。

之后顺序写 assets：

```python
await self.db.add_asset(meta.bv_id, "subtitle", str(sub_result.raw_path), ...)
for f in frames:
    await self.db.add_asset(meta.bv_id, "keyframe", str(f.path), meta={"ts": f.timestamp})
```

### 阶段 4：VLM 帧描述（55%）

`@d:\Diet_Agent_NEW\app\pipeline.py:143-150`：

```python
await _progress(55, "describing frames with VLM")
t_vlm = time.perf_counter()
frame_descs: list[FrameDescription] = await self.vlm.describe_all(frames)
timing["vlm_sec"] = round(time.perf_counter() - t_vlm, 3)
timing["vlm_concurrency"] = getattr(self.vlm, "_concurrency", None)
timing["vlm"] = self.vlm.last_telemetry  # M3 tiering / cache 全部信息
```

`self.vlm.last_telemetry` 是一份完整的分级 / 缓存命中 telemetry，详见 `Detail/02-VLM`。

### 阶段 5：profile + chapter_plan + chapter_cache（75% 前）

`@d:\Diet_Agent_NEW\app\pipeline.py:152-225`：

```python
ctx = LecturizeContext(
    bv_id=meta.bv_id, url=..., title=..., author=..., duration=..., cover_url=...,
    segments=sub_result.segments,
    frame_descs=frame_descs,
)

profile = select_profile(meta.duration, self.settings)
timing["profile"] = {
    "name": profile.name,
    "duration_sec": float(profile.duration_sec),
    "use_chapter_planner": profile.use_chapter_planner,
    "use_map_reduce": profile.use_map_reduce,
    "use_chapter_cache": profile.use_chapter_cache,
    "critic_mode": profile.critic_mode,
    "reviser_mode": profile.reviser_mode,
    "study_question_mode": profile.study_question_mode,
    "chapter_planner_mode": profile.chapter_planner_mode,
}
```

profile 全量字段都进 telemetry，验证脚本能精确知道这次 run 走了哪条路径。

接下来如果 `profile.use_chapter_planner`：

```python
chapter_plan = []
chapter_plan_sec = 0.0
if profile.use_chapter_planner:
    t_plan = time.perf_counter()
    try:
        ir_budget = LengthBudget.for_duration(meta.duration, ...)
        chapter_plan = list(plan_chapters(...))
    except Exception as exc:
        logger.warning("ChapterPlanner failed (degrading to empty): %s", exc)
        chapter_plan = []
    chapter_plan_sec = time.perf_counter() - t_plan
timing["chapter_plan"] = {
    "anchors": len(chapter_plan),
    "mode": profile.chapter_planner_mode if profile.use_chapter_planner else None,
    "sec": round(chapter_plan_sec, 3),
}
```

失败降级到空列表，**不阻断主链路**。

`chapter_cache_obj` 同理懒构造：

```python
chapter_cache_obj = None
if profile.use_chapter_cache:
    try:
        chapter_cache_obj = make_chapter_cache_from_settings(self.settings)
    except Exception as exc:
        chapter_cache_obj = None
```

### 阶段 6：LectureIR 构建（75%）

`@d:\Diet_Agent_NEW\app\pipeline.py:227-283`：

```python
try:
    use_agents = bool(settings.lecture_question_driven or settings.lecture_critic_enabled)
    if use_agents:
        lecture_ir, stats = await self.ir_builder.build_with_agents(
            ctx, profile=profile, chapter_plan=chapter_plan, chapter_cache=chapter_cache_obj,
        )
    else:
        lecture_ir, stats = await self.ir_builder.build(ctx)
    self._dump_ir(meta.bv_id, lecture_ir.model_dump(mode="json"))
    lecture = lecture_ir_to_lecture_json(lecture_ir)
    transcript = " ".join(s.text for s in sub_result.segments)
    marked = lecture.mark_unverified_points(transcript)
    stats["marked_unverified_points"] = marked
except Exception as exc:
    if settings.lecture_strict_agents:
        raise
    logger.warning("LectureIR v2 failed; falling back to v1: %s", exc)
    lecture, stats = await self.lecturizer.lecturize(ctx)
    lecture.generation_mode = "v1_fallback"
    lecture.completeness.notes = "LectureIR 分析失败，已回退到 v1 讲义生成路径。"
    lecture_ir = None
```

两条路径：

- **v2 路径**：`build_with_agents` 内部按 `profile.use_map_reduce` 进一步分流到单次或 Map-Reduce。
- **v1 fallback**：`Lecturizer.lecturize(ctx)` 老路径，单次构建。

`mark_unverified_points(transcript)` 给 quote 找不到原文的 point 加 ⚠️ 前缀（详见 `Detail/04-LectureIR`）。

`_dump_ir(...)` 把 IR JSON 写到 `data/debug/<BV>.lecture_ir.json` 供离线分析。

### 阶段 6.5：telemetry 注入

`@d:\Diet_Agent_NEW\app\pipeline.py:260-282`：

```python
timing["lecture_ir_sec"] = round(time.perf_counter() - t_ir, 3)
if isinstance(stats, dict) and stats.get("agent_timing"):
    timing["agent_timing"] = stats.get("agent_timing")
timing["revise_rounds"] = stats.get("revise_rounds") if isinstance(stats, dict) else 0
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
if chapter_cache_obj is not None:
    try:
        timing["chapter_cache"] = chapter_cache_obj.stats()
    except Exception:
        timing["chapter_cache"] = {}
```

所有阶段子统计在这里集中挂到 `timing`，验证脚本一字不差读得到。

### 阶段 7：taxonomy 修复 + 帧分配

`@d:\Diet_Agent_NEW\app\pipeline.py:285-312`：

```python
lecture.taxonomy = normalize_taxonomy(lecture.taxonomy)
if lecture.taxonomy and lecture.taxonomy.tags:
    lecture.domain_tags = list(lecture.taxonomy.tags)

# 如果 LLM 没把 frames 分到具体章节，按 ts 自动分配
if frame_descs and not any(ch.frames for ch in lecture.chapters):
    for fd in frame_descs:
        target = next(
            (ch for ch in lecture.chapters if ch.start <= fd.timestamp <= ch.end),
            None,
        )
        if target is None:
            continue
        target.frames.append(LectureFrame(ts=..., path=..., caption=..., ocr_text=..., insight=...))
```

`normalize_taxonomy` 在白名单外的 domain 强制 `"其他"` 并钳 confidence。详见 `Detail/09-Copilot-Agent`。

### 阶段 8：render（90%）

`@d:\Diet_Agent_NEW\app\pipeline.py:314-320`：

```python
await _progress(90, "rendering HTML")
t_render = time.perf_counter()
css_inline = self.renderer.load_inline_css()
report_path = self.renderer.render_lecture(lecture, css_inline, cover_path=cover_path)
timing["render_sec"] = round(time.perf_counter() - t_render, 3)
```

`Renderer.render_lecture(...)` 输出 HTML 到 `data/reports/<BV>.html`，内联 CSS + base64 封面。

### 阶段 9：persist（95% 前）

`@d:\Diet_Agent_NEW\app\pipeline.py:322-345`：

```python
t_persist = time.perf_counter()
rel_report = report_path.relative_to(self.settings.data_dir).as_posix()
await self.db.upsert_summary(
    bv_id=meta.bv_id,
    url=ctx.url,
    title=lecture.title,
    author=lecture.author,
    duration=int(lecture.duration),
    cover_url=lecture.cover_url,
    category=lecture.category,
    domain_tags=lecture.domain_tags,
    summary_json=lecture.model_dump(mode="json"),       # ← 完整 LectureJSON
    report_path=rel_report,
    model_used=stats.get("model"),
    token_cost=stats.get("tokens", {}).get("total_tokens", 0) or 0,
    status="done",
    domain=domain or None,
    direction=direction or None,
)
timing["persist_sec"] = round(time.perf_counter() - t_persist, 3)
```

完整 LectureJSON（含双投影）落到 `summaries.summary_json` 一列。

### 阶段 10：RAG 索引（95%-100%）

`@d:\Diet_Agent_NEW\app\pipeline.py:347-378`：

```python
wait_rag = bool(getattr(self.settings, "pipeline_wait_rag", True))
timing["wait_rag"] = wait_rag

async def _do_rag_index() -> int | None:
    try:
        n = await index_lecture(self.rag, meta.bv_id, lecture, lecture_ir)
        logger.info("Copilot RAG indexed %d chunks for %s", n, meta.bv_id)
        return n
    except Exception as exc:
        logger.warning("Copilot RAG hook failed for %s: %s", meta.bv_id, exc)
        return None

if wait_rag:
    await _progress(95, "indexing for Copilot RAG")
    t_rag = time.perf_counter()
    await _do_rag_index()
    timing["rag_index_sec"] = round(time.perf_counter() - t_rag, 3)
else:
    await _progress(95, "scheduling RAG indexing in background")
    timing["rag_index_sec"] = 0.0
    self.last_run_rag_task = asyncio.create_task(_bg_rag())
```

`PIPELINE_WAIT_RAG=true` 时阻塞等索引；false 时丢到后台，让 `report_path` 更快返回。后台任务挂在 `last_run_rag_task` 上，验证脚本可以 `await pipeline.last_run_rag_task` 等它完成。

## 阶段对应的进度数

| 阶段 | 进度 |
|------|------|
| metadata | 5% |
| ingest (subtitle / keyframes / cover) | 15% |
| VLM | 55% |
| LectureIR build | 75% |
| LectureIR done | 78%（v1 fallback 时） |
| render | 90% |
| RAG | 95% |
| done | 100% |

这些 milestone 让前端 SSE 能给用户「具体在做什么」的反馈。

## telemetry 全表（最终 `timing` dict）

成功跑完一个 long 视频的 timing 大概长这样：

```python
{
  "metadata_sec": 0.4,
  "subtitle_sec": 12.3, "keyframes_sec": 38.1, "cover_sec": 0.2,
  "ingest_parallel_sec": 38.2, "subtitle_source": "cc", "keyframes_count": 42,
  "vlm_sec": 156.4, "vlm_concurrency": 4,
  "vlm": {"total_frames": 42, "junk_filtered": 6, "high_tier": 22, "low_tier": 20,
          "cache_hits": 30, "vlm_calls_made": 12, "estimated_token_savings_pct": 41.2,
          "tiering_enabled": True, "cache_enabled": True},
  "profile": {"name": "long", "duration_sec": 2200.0,
              "use_chapter_planner": True, "use_map_reduce": True, ...},
  "chapter_plan": {"anchors": 8, "mode": "structural", "sec": 0.04},
  "lecture_ir_sec": 145.2,
  "agent_timing": {"study_question_sec": 22.1, "initial_ir_sec": 0.0,
                   "critic_sec": 28.4, "reviser_sec": 12.0},
  "revise_rounds": 1,
  "map_reduce": {"map_calls": 8, "cache_hits": 6, "cache_writes": 2,
                 "map_failures": [], "map_total_sec": 84.7,
                 "reduce_local_sec": 0.05, "reduce_global_sec": 28.9,
                 "reduce_global_attempts": 1, "ir_sanity": {...}},
  "reviser": {"mode": "patch", "proposed": 6, "applied": 4, "rejected": 2,
              "unfixable_issues": [], "tokens": {...}},
  "critic": {"verdict": "needs_revision", "issue_count": 5, "metrics": {...}},
  "chapter_cache": {"hits": 6, "misses": 2, "writes": 2, "rows_total": 16},
  "render_sec": 0.7,
  "persist_sec": 0.2,
  "wait_rag": True,
  "rag_index_sec": 18.5,
}
```

整段 telemetry 由 `scripts/_verify_lecture_v2_real_e2e.py` 落到 `data/debug/<BV>.timing.json`，再被 `scripts/_summarize_lecture_matrix.py` 聚合成矩阵报告。详见 `Detail/14-测试-QA-工程化`。

## 失败传播路径

| 阶段 | 失败行为 |
|------|---------|
| metadata | 抛异常 → 整次 run 失败 |
| 缓存命中检查 | DB 异常 → 抛 → 整次失败 |
| ingest 并行（任一） | 抛异常 → `asyncio.gather` 抛 → 整次失败 |
| VLM | 单帧失败局部回填空描述；整批失败抛 → 整次失败 |
| ChapterPlanner | 异常 → 降级到空列表，**继续** |
| ChapterCache | 初始化失败 → 降级 None，**继续** |
| LectureIR v2 | 异常 → 走 v1 fallback（除非 strict） |
| Map-Reduce reduce_global | 抛 `ReduceGlobalError` → 整次失败 |
| Critic | 失败 → noop（除非 strict） |
| Reviser | 失败 → IR 不动（除非 strict） |
| render | 异常 → 整次失败 |
| persist | 异常 → 整次失败 |
| RAG | 异常 → logger.warning，**继续** |

主路径上有 4 个「失败可继续」的阶段（ChapterPlanner / ChapterCache / Agents / RAG），其他都是 fail-fast。这套传播策略让「核心质量保证」和「锦上添花的可观察 / 检索能力」清晰分层。

## 写简历可以怎么提炼

- 在 `Pipeline._run_inner` 中编排了 8 阶段端到端流水线（metadata / ingest parallel / VLM / profile router / chapter plan / IR build / render / persist / RAG），把 ingest / understand / render / copilot 4 个子系统统一调度。
- 实现了细粒度 telemetry（`metadata_sec` / `subtitle_sec` / `keyframes_sec` / `vlm_sec` + `vlm` 子 dict / `chapter_plan` / `map_reduce` / `critic` / `reviser` / `chapter_cache` / `render_sec` / `persist_sec` / `rag_index_sec`），让每次 run 的成本结构都可量化。
- 设计了「核心 fail-fast / 可观察阶段降级」的混合失败传播：ingest / IR build / persist 失败直接抛；ChapterPlanner / ChapterCache / Critic / Reviser / RAG 任何异常都降级为 noop，不影响主链路返回。
- 通过 `PIPELINE_WAIT_RAG=false` 把 RAG 索引拆成后台任务，让首页响应延迟与索引完整性可以独立调优。

## 面试可展开点

### 1. 为什么 ingest 三件事并行而后续阶段串行

ingest 三件事都是 I/O 密集且只依赖 `meta`，事件循环切换收益明显，并行能把 wall-clock 从 Σ 降到 max。VLM 之后阶段都涉及大 LLM 调用且互相依赖（chapter_plan 依赖 frames，IR build 依赖 chapter_plan...），强行并行只会增加复杂度。

### 2. 为什么用 self.last_run_timing 这种「状态变量」而不是返回 dict

Pipeline 的返回值是 `Path`（HTML 报告路径），让验证脚本拿到「成功 / 失败 + HTML 路径」。详细 telemetry 是辅助信息，挂在实例变量上让脚本能 `pipeline.last_run_timing` 拿到，不污染主返回签名。

### 3. 缓存命中检查为什么这么早

在 metadata 之后立即检查，避免**白做后面所有事**——尤其是 keyframes 阶段那个 yt-dlp 下载视频，几十兆带宽 + 几分钟时间，错过 cache 就白浪费。早检查 = 早返回。

### 4. 为什么强制 sequential 写 assets 而不是并行

SQLite 单 writer。让 N 个 `add_asset` 并行只会触发 BUSY error。Sequential 是「与数据库语义对齐」。

### 5. RAG 后台模式的取舍

`PIPELINE_WAIT_RAG=true` 默认能保证「report 返回时就能问答」，对 UI 一致性好；但长视频 RAG 要等 30-60 秒，用户感知慢。`PIPELINE_WAIT_RAG=false` 让 report 更快返回，但用户立刻问的话可能 RAG 还没好。两种语义都合法，配置化让用户自己选。

## 当前文档中的事实与推断

### 事实

- 主流程 `_run_inner` 共 8 个阶段，进度从 5% 推到 100%。
- `last_run_timing` 字段约 20+ 个，覆盖所有阶段耗时和子统计。
- ingest 三件事用 `asyncio.gather` 并行。
- 缓存命中走快路径直接 return。
- 4 个阶段失败时降级到 noop / 空列表（ChapterPlanner / ChapterCache / Agents / RAG）。

### 推断

- Pipeline 是「编排层薄、子模块厚」的典型 — 388 行代码统筹了项目里几乎所有「贵的」操作，每个子模块自己负责自己的复杂度。这种设计让「重构子模块」对主链路影响最小。
