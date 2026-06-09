# Ingest 层：视频原料采集深入

## 文档定位

`Internship/01` 已经讲了「视频先变成原料，再交给理解层」这件事。本篇把这一层拆到每一个文件、每一种降级路径，回答四个问题：

- 元数据怎么来？
- 字幕怎么来？怎么降级？
- 关键帧怎么来？为什么和 VLM 缓存命中率挂钩？
- 这些任务是怎么并行又互不污染的？

## 模块边界（5 个文件）

`app/ingest/` 一共 5 个 Python 文件，职责完全不重叠：

- `@d:\Diet_Agent_NEW\app\ingest\bilibili.py` — 解析链接、抓取视频元数据。
- `@d:\Diet_Agent_NEW\app\ingest\subtitle.py` — CC 字幕优先，Whisper 兜底。
- `@d:\Diet_Agent_NEW\app\ingest\keyframe.py` — PySceneDetect + ffmpeg 抽关键帧。
- `@d:\Diet_Agent_NEW\app\ingest\cover.py` — 封面下载与本地缓存。
- `@d:\Diet_Agent_NEW\app\ingest\cookies.py` — 处理 B 站 cookie 文件。

主链路 `@d:\Diet_Agent_NEW\app\pipeline.py:107-131` 通过 `asyncio.gather` 把 `subtitle / keyframes / cover` 三个 I/O 任务并行起来，元数据则在它们之前同步获取。

## BilibiliIngest：从 URL 到 VideoMeta

入口类 `BilibiliIngest` 定义在 `@d:\Diet_Agent_NEW\app\ingest\bilibili.py:45-148`。它故意只用一个轻量接口 `https://api.bilibili.com/x/web-interface/view`，而不是引入 `bilibili-api-python`，原因写在代码注释里：**减少依赖面，规避 API 漂移风险**。

### 关键调用链

- `BilibiliIngest.parse_bv(url_or_bv)` — 用正则 `BV[0-9A-Za-z]{10}` 提取 BV 号（`@d:\Diet_Agent_NEW\app\ingest\bilibili.py:67-73`）。
- `BilibiliIngest.fetch_meta(url_or_bv)` — 先尝试 `httpx`，失败时退回 `curl.exe` 进程（`@d:\Diet_Agent_NEW\app\ingest\bilibili.py:75-82`）。
- `_fetch_payload_httpx` — 带 `tenacity` 三次指数退避（`@d:\Diet_Agent_NEW\app\ingest\bilibili.py:84-96`）。
- `_payload_to_meta` — 把 payload 解析为 `VideoMeta` dataclass（`@d:\Diet_Agent_NEW\app\ingest\bilibili.py:123-148`）。

### `VideoMeta` 数据形态

```python
@dataclass
class VideoMeta:
    bv_id: str
    aid: int
    title: str
    author: str
    duration: int       # 秒
    cover_url: str
    description: str
    pages: list[dict]   # [{cid, page, part, duration}]
```

`primary_cid` 是只读属性，等于 `pages[0]["cid"]`，多 P 视频默认取第一个 P 的 cid。

### 鉴权与抗封锁

- 默认 `User-Agent` 模拟 Chrome 124，`Referer` 固定 `https://www.bilibili.com/`。
- 通过 `load_cookie_jar(settings.bilibili_cookie_file)` 加载本地 cookie，由 `app/ingest/cookies.py` 负责把 Netscape 格式 cookie.txt 转成 `{name: value}` dict。
- 当 `httpx` 因网络层错误失败时，会通过 `asyncio.to_thread` 走 `curl.exe`。这个 fallback 对国内办公网络的 SSL 中断 / 代理抖动很关键。

## SubtitleExtractor：CC 优先 + Whisper 兜底

字幕模块定义在 `@d:\Diet_Agent_NEW\app\ingest\subtitle.py:57-303`。两条核心路径，互为兜底：

### 路径一：官方 CC 字幕（`_fetch_cc`）

`@d:\Diet_Agent_NEW\app\ingest\subtitle.py:102-151`：

1. 调 `https://api.bilibili.com/x/player/wbi/v2?aid=...&cid=...&bvid=...`。
2. 从 `data.subtitle.subtitles[]` 选轨道，优先 `lan` 含 `zh`，否则取第一条。
3. 拼好 `subtitle_url`（处理 `//` 开头的 protocol-relative URL），二次拉取 JSON。
4. 转换为统一的 `SubtitleSegment(start, end, text)` 列表。
5. 原始 JSON 落到 `data/subtitles/<BV>.cc.json` 便于后续审计。

返回 `SubtitleResult(source="cc", language, segments, raw_path)`。

### 路径二：Whisper 兜底（`_whisper_fallback`）

`@d:\Diet_Agent_NEW\app\ingest\subtitle.py:184-236`：

1. 先用 `yt-dlp -f bestaudio` 抓 m4a 音频到 `data/audio/<BV>.m4a`，重试 + 备用 cmd（`@d:\Diet_Agent_NEW\app\ingest\subtitle.py:238-302`）。
2. 懒导入 `faster_whisper`（`WhisperModel`），让没有装 whisper 的用户依然能用 CC 路径。
3. 优先用本地 HuggingFace snapshot（`_local_faster_whisper_base_snapshot`）避免每次启动都联网。
4. 模型推理在 `asyncio.to_thread` 里跑，避免阻塞事件循环。
5. VAD 滤波（`vad_filter=True`）后产出 `SubtitleSegment[]`，并落到 `data/subtitles/<BV>.whisper.json`。

### 路径三：复用已缓存 Whisper（`_load_cached_whisper`）

`@d:\Diet_Agent_NEW\app\ingest\subtitle.py:155-182`：

如果 `<BV>.whisper.json` 已经存在，直接读，不重跑模型。这对长视频反复调试至关重要——一次 Whisper 转录可能要 5-10 分钟，缓存一次就够。

### 失败传播规则

`extract(bv_id, aid, cid)` 主入口（`@d:\Diet_Agent_NEW\app\ingest\subtitle.py:83-98`）的逻辑是：

- CC 抛任何异常 → `logger.warning` 后尝试 Whisper。
- Whisper 兜底有缓存 → 直接复用。
- 都没有 → 真跑 Whisper；如果连 `faster-whisper` 都没装，**抛 RuntimeError**，让 Pipeline 决定是否进入 v1 fallback。

这种「逐级降级、最后才报错」的设计是 Ingest 层的标准模式，整个 `app/ingest/` 都遵守它。

## KeyframeExtractor：场景检测 + 长度自适应

关键帧模块定义在 `@d:\Diet_Agent_NEW\app\ingest\keyframe.py:29-262`。

### 主入口与缓存复用

`@d:\Diet_Agent_NEW\app\ingest\keyframe.py:36-62`：

1. 输出目录 `data/keyframes/<BV>/`。
2. **若已有 `*.jpg`，直接通过 `_frame_from_path` 重建 `Keyframe` 对象返回**——不重抽。
3. 否则用 `yt-dlp -f 'bv*[height<=480]+ba/...'` 下低质量视频到 `data/keyframes/<BV>.mp4`。
4. 跑 PySceneDetect → ffmpeg 出图。
5. 抽完即 `unlink` 掉源视频，节省磁盘。

### 场景检测（`_detect_scene_timestamps`）

`@d:\Diet_Agent_NEW\app\ingest\keyframe.py:135-200`：

- 默认 `KEYFRAME_LENGTH_ADAPT=true` 时，调用 `app.understand.length_adapt.keyframe_window(duration, base_min, base_max)` 得到长度自适应的 `(kf_min, kf_max)`。
- `KEYFRAME_MIN/MAX` 在这里被解释为**下界**而非绝对界，44 分钟的课程能拿到 ~50-60 帧而不是死锁在 20。
- PySceneDetect `ContentDetector(threshold=27.0)` 出场景列表，每个场景取中点。
- 超出上界则按 `len/kf_max` 步长稀释；少于下界则用均匀采样填充。
- 边界保护：至少返回一帧（视频中点）。

### 帧抽取与时间戳量化（关键修复点）

`@d:\Diet_Agent_NEW\app\ingest\keyframe.py:204-262` 包含一处**精心设计的 bug 修复**：

```python
ts_ms = int(ts * 1000)
ts_quant = ts_ms / 1000.0
fname = f"{ts_ms:08d}.jpg"
# 写入 Keyframe(timestamp=ts_quant, path=fpath)
```

### 为什么要量化时间戳

历史上这一处 commit `09df83f` 的根因分析在 `keyframe.py:211-226` 的多行注释里讲得很清楚：

- 旧实现把 PySceneDetect 的微秒精度浮点 `12.345678` 既塞进 `Keyframe.timestamp`，又用 `int(ts*1000)` 转成文件名 `00012345.jpg`。
- 二跑时 `_frame_from_path` 把文件名反推成 `12.345`，**和原始 `12.345678` 有 1ms 差距**。
- 这 1ms 让 `compute_prompt_hash` 里的 `round(ts, 3)` 桶分错，导致 `chapter_cache` 100% miss。
- 修复方法是 fresh 路径与 cached 路径都用 `ts_ms / 1000.0` 这个**毫秒量化值**作为唯一可信时间。

这是一个非常典型的「上游统一量化以让下游缓存命中」的工程修复，写进简历或面试都比「我做了视频抽帧」要扎实。

## CoverCache 与 Cookies

### CoverCache

`@d:\Diet_Agent_NEW\app\ingest\cover.py` 把封面下载到 `data/covers/<BV>.jpg`，作为渲染时的 hero 图。已存在则跳过。HTML 渲染时这张图会被 base64 内联进报告 HTML（详见 `09-Renderer` 文档）。

### Cookies

`@d:\Diet_Agent_NEW\app\ingest\cookies.py` 提供两个函数：

- `cookie_file_path(path)` — 给 yt-dlp 传 `--cookies <file>` 用，返回 `Path | None`。
- `load_cookie_jar(path)` — 把 Netscape 格式 cookie.txt 解析为 `{name: value}` dict，供 `httpx` 用。

这一层是 B 站抗 403 的关键。`README.md` 明确说明 `BILIBILI_COOKIE_FILE` 是处理「公开抓取不稳定 / 需要登录态」场景的开关。

## 并行编排：Pipeline 是怎么用这一层的

`@d:\Diet_Agent_NEW\app\pipeline.py:107-141`：

```python
async def _timed_subtitle() -> tuple[Any, float]:
    t = time.perf_counter()
    res = await self.subtitles.extract(meta.bv_id, meta.aid, meta.primary_cid)
    return res, time.perf_counter() - t

# 三个任务用 gather 并行
(sub_result, subtitle_sec), (frames, keyframes_sec), (cover_path, cover_sec) = (
    await asyncio.gather(_timed_subtitle(), _timed_keyframes(), _timed_cover())
)
```

注意三点：

- 每个子任务都被包了一层 `_timed_*` 协程，分别计时后塞到 `timing` dict。
- `keyframes` 已经包含了「最贵的 yt-dlp + ffmpeg」，所以它常常是 ingest 阶段的瓶颈。
- 三个任务只依赖 `meta`（已经在前面同步获取了），所以并行是天然安全的。

### 持久化的边界

并行返回后立即 sequential 落库（`@d:\Diet_Agent_NEW\app\pipeline.py:135-142`）：

```python
await self.db.add_asset(meta.bv_id, "subtitle", str(sub_result.raw_path), ...)
for f in frames:
    await self.db.add_asset(meta.bv_id, "keyframe", str(f.path), meta={"ts": f.timestamp})
```

`assets` 表用 `(bv_id, kind, path)` 三元组做主键。同一个 BV 的同一帧再写一次会被 `INSERT OR REPLACE` 覆盖，不重复污染。

## Telemetry 出口

Pipeline 把 ingest 阶段的所有时间和元数据落进 `last_run_timing`，验证脚本能直接读：

- `timing.metadata_sec` — `BilibiliIngest.fetch_meta` 耗时。
- `timing.subtitle_sec` / `timing.keyframes_sec` / `timing.cover_sec` — 三个并行子任务各自耗时。
- `timing.ingest_parallel_sec` — gather 整体 wall-clock 时间。
- `timing.subtitle_source` — `"cc"` 或 `"whisper"`，**这是排查质量问题的关键信号**。
- `timing.keyframes_count` — 抽出多少帧，可以用来反推 `keyframe_window` 是否按预期放大了上界。

## 写简历可以怎么提炼

- 搭建了 Bilibili 视频采集与预处理链路，覆盖元数据抓取、CC 字幕与 Whisper 兜底、PySceneDetect 关键帧抽取与封面缓存，并通过本地 SQLite + 文件系统持久化原始资产，支持失败重试和增量调试。
- 设计了三层降级路径（CC → Whisper 缓存 → Whisper 实跑），并通过 `tenacity` + `httpx → curl` 双客户端策略提升公开抓取的鲁棒性。
- 通过对关键帧时间戳进行毫秒量化，统一 fresh 抽取与 cached 回放路径的时间值，把章节级 prompt 缓存命中率从 0% 提升到 100%。
- 用 `asyncio.gather` 把字幕、关键帧、封面三个 I/O 任务并行化，减少 ingest 阶段 wall-clock 时间。

## 面试可展开点

### 1. 为什么把字幕、关键帧、封面三个任务并行

它们都只依赖 `VideoMeta`，互相之间没有依赖；并且都是 I/O 密集型（网络 + ffmpeg 子进程），事件循环切换收益明显。串行做的话，总时长 ≈ Σ 单任务；并行做的话，总时长 ≈ max(单任务)。

### 2. 为什么需要 Whisper 兜底而不是「没字幕就报错」

技术类长视频大概率没有 CC（UP 主没空打），而这些视频又恰好是讲义化最有价值的输入。直接报错等于把项目实际可用范围缩到很小。

### 3. 为什么关键帧时间戳要量化

因为下游章节级缓存用 `round(ts, 3)` 做 bucket。fresh 路径和 cached 路径如果数值不一致，缓存就全 miss。这是典型的「上游小数值不一致 → 下游缓存层失效」问题。修在上游一行代码，比在下游每个 cache 调用都加 normalize 函数要干净得多。

### 4. 为什么不是把整个 Whisper 模型搬上来用

`faster-whisper` 是懒导入的（`@d:\Diet_Agent_NEW\app\ingest\subtitle.py:189-196`），用户没装也能用 CC 路径。这种「关键依赖按需启用」的设计是工程意识的体现。

## 当前文档中的事实与推断

### 事实

- 入口文件有 5 个：`bilibili / subtitle / keyframe / cover / cookies`。
- `Pipeline._run_inner` 用 `asyncio.gather` 并行执行三个 I/O 任务。
- 字幕降级顺序：CC → cached Whisper → 实跑 Whisper。
- 关键帧时间戳已做毫秒量化，与 `chapter_cache.compute_prompt_hash` 配合。
- 所有原始资产都落到 `data/{subtitles,keyframes,audio,covers}/` 和 `assets` 表。

### 推断

- Ingest 层的设计目标是「让下游永远拿到稳定、可复用的原料」，而不仅仅是「拉数据」。

这个判断与 `pipeline.py` 的缓存命中和 `assets` 表的存在一致。
