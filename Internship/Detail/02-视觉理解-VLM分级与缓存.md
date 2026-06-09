# 视觉理解：VLM 分级与缓存

## 文档定位

`Internship/01` 提到了「关键帧需要视觉描述」，但并没有展开「为什么不是每帧都用昂贵 prompt」「为什么需要 SQLite 缓存」。本篇把视觉理解层完整拆开，回答：

- 关键帧怎么被分级？
- HIGH / LOW 两种 prompt 各有什么差异？
- 为什么单独再建一个 SQLite 缓存？
- 这一层失败时怎么不污染主链路？

## 三件套总览

视觉理解层由三个模块组成，分工很干净：

- **`@d:\Diet_Agent_NEW\app\understand\frame_ranker.py`** — 纯 CPU 分级器，决定每帧走 HIGH 还是 LOW prompt。
- **`@d:\Diet_Agent_NEW\app\understand\vlm.py`** — `FrameDescriber` 主类，编排 ranker / 缓存 / VLM 调用。
- **`@d:\Diet_Agent_NEW\app\understand\vlm_cache.py`** — SQLite 缓存，按 `(sha256, model, tier)` 寻址。

入口是 `Pipeline.vlm.describe_all(frames)`（`@d:\Diet_Agent_NEW\app\pipeline.py:144-150`），输出 `list[FrameDescription]`。

## 为什么要分级：成本与质量的折中

`@d:\Diet_Agent_NEW\app\understand\frame_ranker.py:1-37` 的 module docstring 把动机讲得很直白：

- 旧实现给每帧都跑「caption + OCR + visual_type + 评分」prompt，约 1k 输入 + 200 输出 token。
- 对一个 44 分钟的课程，关键帧自适应放到 ~50-60 帧时，光 VLM 成本就成了瓶颈。
- 但**大部分帧不需要这么贵**：talking head、连续幻灯片、黑屏过场，给一个 OCR-only 200-token 的 LOW prompt 就够了。

LOW / HIGH 的区别在 `@d:\Diet_Agent_NEW\app\understand\vlm.py:268-296`：

| Tier | system prompt | user prompt | extra kwargs |
|------|--------------|-------------|-------------|
| HIGH | `VLM_FRAME_DESCRIBE_SYSTEM` | `VLM_FRAME_DESCRIBE_USER`（要求 JSON 含 caption / ocr_text / visual_type / 三个分数 / why_useful） | `temperature=0.2` |
| LOW | 无 system | `VLM_FRAME_OCR_ONLY_USER` | `max_tokens=200, temperature=0.0` |

## KeyframeRanker：三信号纯 CPU 分级

`KeyframeRanker` 完全跑在本地，**不调任何模型**。它对每帧计算三个 Pillow 信号：

- **`similarity_to_prev`** — 与前一帧的 dHash 相似度。同一张幻灯片连续出现 → 接近 1.0。`@d:\Diet_Agent_NEW\app\understand\frame_ranker.py:216-231`。
- **`entropy`** — 灰度直方图 Shannon 熵，范围 `[0, 8]`。黑屏 / 纯白 → 接近 0。`@d:\Diet_Agent_NEW\app\understand\frame_ranker.py:234-243`。
- **`edge_density`** — `ImageFilter.FIND_EDGES` 后高于阈值 50 的像素比。代码 / 公式幻灯片 → > 0.05。`@d:\Diet_Agent_NEW\app\understand\frame_ranker.py:246-251`。

### 分级策略（混合策略 D3）

`@d:\Diet_Agent_NEW\app\understand\frame_ranker.py:167-210` 的 `classify(frames)` 流程：

1. **绝对垃圾过滤**：`similarity ≥ 0.93` 或 `entropy < 2.0` → 直接强制 LOW，不进 HIGH 候选。
2. **HIGH 楼层保护**：在幸存者中按 `score = 0.5(1-sim) + 0.3 entropy + 0.2 edge_density` 排序，强制保留至少 `ceil(survivors * 0.5)` 个为 HIGH。
3. **极端情况**：如果全部被标垃圾，依然按楼层把得分最高的几帧提升到 HIGH（移除它们的 junk 标记），避免输出 0 张 captioned frame。

阈值都在 `Settings`：

- `VLM_JUNK_SIM_THRESHOLD=0.93`
- `VLM_JUNK_ENTROPY_THRESHOLD=2.0`
- `VLM_TIERING_HIGH_FLOOR_RATIO=0.5`
- `VLM_TIERING_ENABLED=true`

`VLM_TIERING_ENABLED=false` 时整个分级器变 no-op，每帧都是 HIGH（回到旧行为）。

## VLMCache：内容寻址 + 跨 BV 复用

### 关键设计点

`@d:\Diet_Agent_NEW\app\understand\vlm_cache.py:1-29` 的 module docstring 列了三件事：

- **键是 `(sha256(image_bytes), model, tier)`** —— 跨 BV 复用。同一张幻灯片在两个视频里出现，第二次直接命中。
- **WAL + 每次新连接** —— 简单 thread / process 模型，无需 ORM。
- **`enabled=False` 时全部 no-op** —— `batch_lookup` 返回全 miss 字典，`batch_insert` 静默丢。

### 表结构

`@d:\Diet_Agent_NEW\app\understand\vlm_cache.py:64-77`：

```sql
CREATE TABLE IF NOT EXISTS vlm_cache (
    image_sha256 TEXT NOT NULL,
    model        TEXT NOT NULL,
    tier         TEXT NOT NULL CHECK (tier IN ('high','low')),
    payload      TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (image_sha256, model, tier)
);
CREATE INDEX idx_vlm_cache_created ON vlm_cache(created_at);
```

`payload` 是 JSON 字符串，字段：`caption / ocr_text / visual_type / importance_score / ocr_density / novelty_score / why_useful`（即 `_payload_of(desc)` 的输出）。

### Lazy 迁移历史 JSON 缓存

旧版本把缓存写在 `data/vlm_cache/<model>/<bv>.json`，是按 BV 整文件的。`FrameDescriber.describe_all` 在 SQLite 全 miss 时会自动调 `VLMCache.import_legacy_bv_json` 把旧文件作为 HIGH-tier 行导入（`@d:\Diet_Agent_NEW\app\understand\vlm.py:138-152`），让用户可以无缝升级而不丢历史。

## FrameDescriber.describe_all：完整调度

`@d:\Diet_Agent_NEW\app\understand\vlm.py:120-221` 是这一层的核心，全流程 8 步：

1. **分级** — `KeyframeRanker.classify(frames)` 返回 `tiers / signals / is_junk`。
2. **算 hash** — 每帧 SHA-256，失败时退回 `f"NOHASH::{path}"` 作为稳定 key（`@d:\Diet_Agent_NEW\app\understand\vlm.py:237-246`）。
3. **批量 lookup** — `VLMCache.batch_lookup` 用一条 `WHERE (sha, model, tier) IN (...)` 查询拿回所有 hits。
4. **必要时迁移 legacy 缓存** — 见上一节。
5. **找出 miss 列表** — `todo: list[(idx, frame, tier, sha)]`。
6. **并发 VLM 调用** — `asyncio.Semaphore(_concurrency)` 限流（默认 4，由 `QWEN_VL_CONCURRENCY` 控制）。每个调用走 `_describe_one(frame, tier=tier)`。
7. **写回缓存** — `_is_failed_response(desc, tier)` 过滤掉 HIGH-tier 全空响应（避免毒化缓存），其他成功响应批量 `batch_insert`。
8. **合并结果** — 把 cache hits 和 fresh results 按原始 frame 顺序拼回。

`_is_failed_response` 的判断很务实（`@d:\Diet_Agent_NEW\app\understand\vlm.py:368-384`）：只对 HIGH-tier 判失败，且要求 caption / ocr_text / why_useful 都空且 importance==0 才认定。LOW-tier 允许 ocr_text 为空（无可见文字是合法情况）。

### 重试链

`_describe_one` 用 `tenacity` 三次指数退避（`@d:\Diet_Agent_NEW\app\understand\vlm.py:263-267`），失败时返回 `_empty_parse()`，不抛——交给 `_is_failed_response` 决定是否进缓存。这是「失败局部隔离」的典型做法：单帧失败不会让整个链路报错。

## Telemetry：M3 最重要的可观察性产物

`@d:\Diet_Agent_NEW\app\understand\vlm.py:208-220` 写 `_telemetry`：

```python
self._telemetry = {
    "total_frames": total,
    "junk_filtered": junk_count,
    "high_tier": ..., "low_tier": ...,
    "cache_hits": cache_hit_count,
    "vlm_calls_made": len(todo),
    "vlm_calls_high": ..., "vlm_calls_low": ...,
    "estimated_token_savings_pct": round(savings_pct, 2),
    "tiering_enabled": ..., "cache_enabled": ...,
}
```

主链路在 `@d:\Diet_Agent_NEW\app\pipeline.py:150` 把它一字不差挂到 `timing.vlm`，验证脚本 / 矩阵报告直接消费。

`estimated_token_savings_pct` 的计算用的是经验权重 `HIGH=1.0 / LOW=0.4`（`@d:\Diet_Agent_NEW\app\understand\vlm.py:66-67`），目的是给一个相对量级，等真实 DashScope token 计数接入后可以重新校准而不动语义。

## 响应解析：宽松 JSON 抽取

DashScope 返回的有时是裸 JSON，有时被 markdown fence 包住，有时前面带一段散文。`_parse_vlm_response` 用 `_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)` 抓所有可能的 JSON 对象候选，逐个 `json.loads`，第一个成功的就用（`@d:\Diet_Agent_NEW\app\understand\vlm.py:390-419`）。

`_coerce_score` 把任何可转 float 的值钳到 `[0, 1]`，非法返回 0.0，保证 `FrameDescription.importance_score` 这种字段永远在合法范围。

## 失败时的退路

VLM 这一层在主链路里**不应该让整条流水线挂掉**：

- 单帧 VLM 失败 → `_empty_parse()` 返回空 `FrameDescription`，不进缓存，下次重跑还有机会。
- 整批 VLM 全失败 → `describe_all` 还是返回 `[FrameDescription(empty)] * N`，主链路继续往下走。`LectureIRBuilder` 只把它当作「视觉信号弱」的输入。
- 缓存数据库挂了 → `VLMCache.__init__` 的 try-except 把 `_enabled` 翻成 False，整个缓存变 no-op，主链路不感知。

## 写简历可以怎么提炼

- 对关键帧实现了两级 VLM prompt 分级（HIGH 全要素 / LOW OCR-only），通过纯 CPU 的 dHash + entropy + edge density 三信号在调用模型前完成路由，并设计了 HIGH 楼层比例保护，避免分级误伤视觉密集型视频。
- 用 SQLite WAL 实现了内容寻址（`sha256 × model × tier`）的 VLM 描述缓存，支持跨视频复用同一张幻灯片的视觉描述结果；同时提供 legacy per-BV JSON → SQLite 的懒迁移机制。
- 引入了完整的 telemetry（`vlm_calls_high / low / cache_hits / estimated_token_savings_pct`），让长视频的 VLM 成本下降可量化可回归。

## 面试可展开点

### 1. 为什么要在 VLM 之前做纯 CPU 分级，而不是让模型自己判断

模型判断「这张图值不值得详细描述」需要先把图送进去，省不了一次 input token。CPU 端用 dHash / entropy / edge density 几乎免费，能在调模型之前就剔掉重复幻灯片和黑屏帧。这是典型的「便宜的过滤先做」。

### 2. 为什么 HIGH 楼层是「至少 50% 幸存者」而不是固定 N 张

固定 N 张对短视频和长视频都不合适。50% 比例既能让长视频得到比例上合理的 HIGH 帧数，又能让纯口播视频不会被压成全 LOW。

### 3. 为什么缓存键里要包含 `model`

DashScope 升级模型（qwen3.5 → qwen3.6 → ...）时返回的 caption 风格、visual_type 取值都会变。如果不区分模型，缓存里的旧描述会被新模型的 prompt 误用。

### 4. 为什么不直接复用主库 `data/app.db` 来存这些缓存

VLM 缓存是「内容寻址」的横切数据，跟「按 BV 实体寻址」的 `summaries` / `assets` 表语义完全不同。把它放独立 SQLite 让数据迁移、磁盘清理、备份策略都更简单——一个 `rm data/vlm_cache.sqlite` 就能干净重跑。

### 5. _is_failed_response 为什么要这么写

`tenacity` 重试链有时是「网络成功了但返回空 JSON」。如果把空响应也写缓存，下次永远命中错的空结果。判定空响应不入缓存 + 不阻断主链路，是「让缓存自愈」的关键。

## 当前文档中的事实与推断

### 事实

- `FrameDescriber` / `KeyframeRanker` / `VLMCache` 三个类分别在三个模块。
- 分级器三信号 + 楼层比例的策略写在 `@d:\Diet_Agent_NEW\app\understand\frame_ranker.py:185-210`。
- 缓存键是 `(sha256, model, tier)`，表 schema 在 `@d:\Diet_Agent_NEW\app\understand\vlm_cache.py:64-77`。
- HIGH / LOW prompt 差异写在 `@d:\Diet_Agent_NEW\app\understand\vlm.py:268-296`。
- Telemetry 落点 `timing.vlm`，与验证脚本对齐。

### 推断

- 这一层是 M3 阶段对长视频成本的主要优化点；其他模块（IR builder / Critic / Reviser）的优化是文本侧的，与之互补。
