# Config：配置体系与降级矩阵

## 文档定位

`Internship/04` 把 config 描述为「集中、类型化的配置层」，但没有展开 738 行的 `app/config.py` 长什么样、有哪些类别、降级开关是怎么工作的。本篇梳理整套配置体系，特别是「降级矩阵」：什么开关在什么情况下让什么模块退到什么行为。

## 一句话定位

`Settings` 是一个基于 `pydantic-settings.BaseSettings` 的中央配置容器（738 行），把模型 API Key、并发限额、长度路由阈值、缓存开关、降级标志、Whisper 参数等等全部用强类型字段集中管理。**几乎每个跨模块的运行行为都能通过环境变量调整**。

## 配置项分组（738 行字段速查）

`@d:\Diet_Agent_NEW\app\config.py:12-611` 按注释分组，大致 12 类：

### 1. Models（`@d:\Diet_Agent_NEW\app\config.py:24-83`）

模型栈分配，已在 `Detail/00-总览` 介绍：

- DeepSeek（文本 + Copilot）：`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_REQUEST_TIMEOUT` / `DEEPSEEK_TRUST_ENV`。
- DashScope（VLM + Embedding）：`DASHSCOPE_API_KEY` / `QWEN_VL_MODEL` / `QWEN_TEXT_MODEL`（alias 复用，实际跑 DeepSeek）/ `DASHSCOPE_BASE_URL` / `DASHSCOPE_REQUEST_TIMEOUT` / `DASHSCOPE_TRUST_ENV`。
- thinking 总开关：`QWEN_TEXT_ENABLE_THINKING`（默认 false）— 同一个字段路由两种 API：DashScope 用 `enable_thinking: bool`，DeepSeek 用 `thinking: {"type": "enabled"|"disabled"}`。

### 2. VLM tiering + cache（`@d:\Diet_Agent_NEW\app\config.py:84-131`）

- `VLM_TIERING_ENABLED=true` — 总开关。
- `VLM_TIERING_HIGH_FLOOR_RATIO=0.5` — HIGH 楼层比例。
- `VLM_JUNK_SIM_THRESHOLD=0.93` — dHash 相似度阈值。
- `VLM_JUNK_ENTROPY_THRESHOLD=2.0` — 熵阈值。
- `VLM_CACHE_ENABLED=true` — 缓存总开关。
- `VLM_CACHE_PATH=""` — 默认 `<DATA_DIR>/vlm_cache.sqlite`。

### 3. Whisper fallback（`@d:\Diet_Agent_NEW\app\config.py:133-136`）

- `WHISPER_MODEL=base` — 模型名（base / small / medium / large 或 HuggingFace snapshot 路径）。
- `WHISPER_DEVICE=cpu`。
- `WHISPER_COMPUTE_TYPE=int8`。

### 4. Auth + Service（`@d:\Diet_Agent_NEW\app\config.py:138-159`）

- `BASIC_AUTH_USER=admin` / `BASIC_AUTH_PASSWORD=change-me`。
- `APP_HOST=0.0.0.0` / `APP_PORT=8000`。
- `DATA_DIR=./data`。
- `MAX_CONCURRENT_JOBS=2` — Pipeline 全局并发上限。
- `KEYFRAME_MIN=8` / `KEYFRAME_MAX=20` / `KEYFRAME_THRESHOLD=27.0` / `KEYFRAME_LENGTH_ADAPT=true`。
- `BILIBILI_COOKIE_FILE=./data/cookies/bilibili.txt`。

### 5. Lecture v2 多 Agent（`@d:\Diet_Agent_NEW\app\config.py:161-212`）

- `LECTURE_QUESTION_DRIVEN=true` — Question-Driven 抽取。
- `LECTURE_STUDY_QUESTION_MULTI_WINDOW=true` — 多窗口采样。
- `LECTURE_CRITIC_ENABLED=true` — Critic 总开关。
- `LECTURE_REVISER_ENABLED=false` — Legacy 开关（默认关）。
- `LECTURE_REVISER_MODE=off` — `off / patch / full`，新的派遣模式。

### 6. M2 长度路由（`@d:\Diet_Agent_NEW\app\config.py:214-244`）

- `LECTURE_PROFILE_THRESHOLDS_SEC=180,900,3600` — tiny / standard / long / epic 切分阈值（CSV 字符串）。

### 7. M2 ChapterPlanner（`@d:\Diet_Agent_NEW\app\config.py:246-337`）

- `LECTURE_CHAPTER_PLANNER_ENABLED=true`。
- `LECTURE_CHAPTER_PLANNER_MIN_SEC=60` — 短视频不跑。
- `LECTURE_PLANNER_SILENCE_SEC=6.0` — 静音判定阈值。
- `LECTURE_PLANNER_TRANSITION_PHRASE_CONFIDENCE=0.85` / `_SILENCE_CONFIDENCE=0.6` / `_VISUAL_SHIFT_CONFIDENCE=0.55`。
- `LECTURE_CHAPTER_MAX_DURATION_SEC=720` — M2.2 章节宽度兜底（12 分钟）。

### 8. ChapterCache（`@d:\Diet_Agent_NEW\app\config.py:339-363`）

- `LECTURE_CHAPTER_CACHE_ENABLED=true`。
- `LECTURE_CHAPTER_CACHE_PATH=""` — 默认 `<DATA_DIR>/chapter_cache.sqlite`。

### 9. Map-Reduce timing / retry（`@d:\Diet_Agent_NEW\app\config.py:364-493`）

- `LECTURE_MAP_CHAPTER_MAX_RETRIES=1`。
- `LECTURE_MAP_CHAPTER_TIMEOUT=180.0`。
- `LECTURE_MAP_CHAPTER_MAX_TOKENS=8192`。
- `LECTURE_REDUCE_GLOBAL_TIMEOUT=120.0`。
- `LECTURE_REDUCE_GLOBAL_MAX_RETRIES=1`。
- `LECTURE_REDUCE_GLOBAL_MAX_TOKENS=4000`。
- `LECTURE_IR_MAX_TOKENS=8192`。
- `LECTURE_REVISER_MAX_TOKENS=8192`。
- `LECTURE_IR_AUTO_FALLBACK_TO_MAP_REDUCE=true` — 单次 IR 截断时自动 fallback。
- `LECTURE_MAP_PROMPT_VERSION=m2-map-v1` — bump 时全表 miss。

### 10. Critic timing（`@d:\Diet_Agent_NEW\app\config.py:495-554`）

- `LECTURE_CRITIC_MAX_ROUNDS=1`。
- `LECTURE_CRITIC_TIMEOUT=300.0` — Legacy。
- `LECTURE_CRITIC_FULL_TIMEOUT=180.0` — P5 full mode。
- `LECTURE_CRITIC_PROJECTED_TIMEOUT=120.0` — P5 projected mode。
- `LECTURE_CRITIC_MAX_PROMPT_CHARS=24000` — 字幕摘录字符上限。

### 11. Reviser + 其他（`@d:\Diet_Agent_NEW\app\config.py:555-589`）

- `LECTURE_REVISER_TIMEOUT=600.0` — 全 IR 重写时延（最大）。
- `LECTURE_REVISER_PROMPT_VERSION=m2-patch-v1`。
- `LECTURE_STRICT_AGENTS=false` — Agent 失败时是否抛异常。
- `LECTURE_CODE_HIGHLIGHTER=highlight.js`（或 `prism` 或空）。
- `LECTURE_MAP_REDUCE_THRESHOLD_SEC=1200` — 旧字段，被 profile 替代。
- `PIPELINE_WAIT_RAG=true` — 是否阻塞等 RAG 索引。

### 12. Copilot + MCP（`@d:\Diet_Agent_NEW\app\config.py:593-610`）

- `QWEN_COPILOT_MODEL=deepseek-v4-flash`。
- `QWEN_EMBEDDING_MODEL=tongyi-embedding-vision-flash-2026-03-06`。
- `COPILOT_FRAME_VISION_EMBED=false` — 帧视觉 embedding（默认关）。
- `COPILOT_MAX_TOOL_CALLS=6` — ReAct 步数上限。
- `COPILOT_MAX_CONCURRENT=4`。
- `BOCHA_API_KEY` / `BOCHA_BASE_URL` — 可选 Web 搜索后端。
- `MCP_SERVER_TOKEN` / `MCP_EXPOSE_SUMMARIZE=false`。

## 关键 validator

### CSV 字符串 → 三元组

`@d:\Diet_Agent_NEW\app\config.py:613-650` 的 `_parse_profile_thresholds`：

```python
@field_validator("lecture_profile_thresholds_sec", mode="before")
def _parse_profile_thresholds(cls, v):
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(...)
        v = tuple(int(p) for p in parts)
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError(...)
        ints = tuple(int(x) for x in v)
        if not (ints[0] < ints[1] < ints[2]):
            raise ValueError(...)  # 严格升序
        return ints
```

`NoDecode` 标记禁止 pydantic-settings 默认的 JSON 解码，让 CSV 字符串能原样到 validator。

### Reviser mode 向后兼容

`@d:\Diet_Agent_NEW\app\config.py:652-669` 的 `_apply_reviser_mode_compat`：

```python
@model_validator(mode="after")
def _apply_reviser_mode_compat(self) -> "Settings":
    if "lecture_reviser_mode" not in self.model_fields_set:
        # 用户没显式设新字段 → 旧字段映射
        new_mode = "full" if self.lecture_reviser_enabled else "off"
        object.__setattr__(self, "lecture_reviser_mode", new_mode)
    return self
```

`LECTURE_REVISER_ENABLED=true` 但没设 `LECTURE_REVISER_MODE` 时，自动映射成 `mode="full"`。如果用户同时设了两者，新字段优先。这是「老用户配置不破坏」的典型迁移路径。

## text_extra_body：跨 API 路由

`@d:\Diet_Agent_NEW\app\config.py:675-694`：

```python
def text_extra_body(self) -> dict[str, Any]:
    base = self.deepseek_base_url or ""
    if "dashscope" in base.lower():
        return {"enable_thinking": self.qwen_text_enable_thinking}
    if "deepseek.com" in base.lower():
        return {"thinking": {"type": "enabled" if self.qwen_text_enable_thinking else "disabled"}}
    return {}
```

**同一个开关**根据 base_url 选两种 API shape。DashScope 用 `enable_thinking: bool`，DeepSeek 用 `thinking: {"type": ...}`。任何 LLM 调用都通过 `settings.text_extra_body()` 拿这个 dict，避免每处都判断 backend。

## 派生路径

`@d:\Diet_Agent_NEW\app\config.py:697-720`：

```python
@property
def db_path(self) -> Path: return self.data_dir / "app.db"
@property
def reports_dir(self) -> Path: return self.data_dir / "reports"
@property
def keyframes_dir(self) -> Path: return self.data_dir / "keyframes"
@property
def subtitles_dir(self) -> Path: return self.data_dir / "subtitles"
@property
def audio_dir(self) -> Path: return self.data_dir / "audio"
@property
def cookies_dir(self) -> Path: return self.data_dir / "cookies"
```

所有路径派生于 `DATA_DIR`，所以一次 `DATA_DIR=/mnt/external` 切换让所有数据搬家。

## get_settings：单例

`@d:\Diet_Agent_NEW\app\config.py:733-737`：

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
```

`@lru_cache` 保证整个进程只构造一个 `Settings`，且在第一次拿配置时自动建好所有数据目录。

## 完整降级矩阵

下面是所有「失败 / 配置缺失 / 资源不足」时模块降级到什么行为的矩阵：

| 场景 | 触发条件 | 降级目标 | 控制开关 |
|------|---------|---------|---------|
| **B 站 API 失败** | `httpx` 抛错 | curl.exe subprocess 重试 | 内置 |
| **B 站 403** | API code != 0 | 用 cookie 文件重试 | `BILIBILI_COOKIE_FILE` |
| **CC 字幕缺失** | 没找到 subtitle URL | Whisper 兜底 | `WHISPER_*` 系列 |
| **Whisper 未安装** | `import faster_whisper` 失败 | 抛 RuntimeError | — |
| **关键帧抽取失败** | yt-dlp / ffmpeg 异常 | 空 frames 列表，主链路继续 | — |
| **VLM 单帧调用失败** | tenacity 3 次重试后 | `_empty_parse()` 返回空描述 | — |
| **VLM 缓存初始化失败** | OperationalError | 整库变 no-op | `VLM_CACHE_ENABLED=false` |
| **VLM tiering 误判** | 所有帧被标 junk | 楼层保护强制部分进 HIGH | `VLM_TIERING_HIGH_FLOOR_RATIO` |
| **ChapterPlanner 失败** | 任意异常 | `chapter_plan=[]`，由 LLM 自己切 | `LECTURE_CHAPTER_PLANNER_ENABLED=false` |
| **ChapterCache 失败** | 初始化或 get/put 异常 | 静默回退到无缓存 | `LECTURE_CHAPTER_CACHE_ENABLED=false` |
| **单次 IR 截断** | `finish_reason=length` | 自动 fallback 到 Map-Reduce | `LECTURE_IR_AUTO_FALLBACK_TO_MAP_REDUCE=true` |
| **Map 单章失败** | 重试耗尽 | 占位符（不入缓存） | `LECTURE_MAP_CHAPTER_MAX_RETRIES` |
| **Reduce-Global 失败** | 重试耗尽 | 抛 `ReduceGlobalError`，整次 build 失败 | `LECTURE_REDUCE_GLOBAL_MAX_RETRIES` |
| **StudyQuestion 失败** | strict=false | 空列表 | `LECTURE_STRICT_AGENTS` |
| **Critic 失败** | strict=false | `verdict="fail"` 但不抛 | 同上 |
| **Reviser 失败** | strict=false | IR 不动 | 同上 |
| **LectureIR v2 全失败** | 任意 LectureIR 验证错误 | v1 fallback 路径（`Lecturizer`） | `LECTURE_STRICT_AGENTS=false` |
| **patch Reviser schema 违规** | `apply_patches` 后 LectureIR 校验失败 | 整批补丁回滚到原 IR | 内置 |
| **sqlite-vec 加载失败** | C 扩展无法 load | RAG 退到 FTS-only | 内置 |
| **embedding 维度变化** | 已存 dim ≠ 新 dim | 抛 `EmbeddingError`，要求 reset_vectors | — |
| **embed 单条失败** | DashScope 异常 | 整批 raise，不写表 | — |
| **RAG 索引失败** | upsert_chunks 异常 | logger.warning，主链路返回 | `index_lecture` 内 catch |
| **RAG 检索向量失败** | vec_search 异常 | 退到 FTS-only 检索 | 内置 |
| **MCP RAG 初始化失败** | RAG init 异常 | 启动继续，search 工具返回空 | 内置 |
| **Copilot 配额耗尽** | error 含 quota / 额度 | SSE error event with `quota_exhausted` | 内置 |
| **Copilot 上游 403** | error code 403 | SSE error event with `upstream_forbidden` | 内置 |
| **Bocha web_search 失败** | API 异常 | output 含 error，但不抛 | `BOCHA_API_KEY` |

## 「严格模式」LECTURE_STRICT_AGENTS

这个开关值得单独说。`@d:\Diet_Agent_NEW\app\config.py:569`：

```python
lecture_strict_agents: bool = Field(default=False, alias="LECTURE_STRICT_AGENTS")
```

- `false`（默认）— Agent 失败降级，主链路继续，最差走 v1 fallback。适合**生产环境**优先保证用户拿到结果。
- `true` — Agent 失败直接抛异常。适合**开发 / 调试**揪出 prompt 问题。

`@d:\Diet_Agent_NEW\app\pipeline.py:250-258` 的核心控制：

```python
except Exception as exc:
    if getattr(self.settings, "lecture_strict_agents", False):
        logger.exception("LectureIR v2 failed for %s in strict mode", meta.bv_id)
        raise
    logger.warning("LectureIR v2 failed for %s; falling back to v1: %s", meta.bv_id, exc)
    lecture, stats = await self.lecturizer.lecturize(ctx)
    lecture.generation_mode = "v1_fallback"
```

## .env.example 12914 字节

`.env.example` 几乎为每个 Settings 字段都写了注释（约 700 行），是项目最重要的运维文档之一。它的作用：

- 新部署者直接 `cp .env.example .env` 就能拿到完整配置模板。
- 注释里写了每个字段的默认值、典型生产值、调试用值。
- 不需要去翻 `config.py` 才能知道有什么可调。

## 写简历可以怎么提炼

- 设计了一个 12 类、~50 个字段的中央配置体系（pydantic-settings + 多字段 validator + 派生属性 + .env.example），用一份 `Settings` 实例统一管理 4 个 LLM 模型、3 个 SQLite 缓存、6 个数据目录、20+ 个并发 / 超时 / 阈值开关。
- 通过 `text_extra_body()` 把同一个 thinking 总开关路由到 DashScope (`enable_thinking`) 和 DeepSeek (`thinking.type`) 两种 API shape，让上层 Agent 代码完全感知不到 backend 差异。
- 在 24+ 处实现了运行时降级（CC → Whisper / 单次 IR → Map-Reduce / 向量 → FTS-only / Agent 失败 → v1 fallback / sqlite-vec 缺失 → FTS-only），并通过 `LECTURE_STRICT_AGENTS` 开关让开发态和生产态有不同的失败语义。
- 用 `_apply_reviser_mode_compat` model_validator 实现 `LECTURE_REVISER_ENABLED` → `LECTURE_REVISER_MODE` 的向后兼容映射，让老用户 .env 无需修改也能跑新版本。

## 面试可展开点

### 1. 为什么 thinking 总开关不直接在 Agent 代码里判断 backend

「同一个语义」放在配置层一次性处理。Critic / Reviser / IR builder / Copilot 全都调 `settings.text_extra_body()` 拿这个 dict，未来加第三个 backend（比如 Anthropic）只需要改一个函数。把 backend-specific 拼接散落到每个调用点会让维护成本爆炸。

### 2. 为什么把 reviser 旧 / 新两个开关都保留

为了「向后兼容」。`LECTURE_REVISER_ENABLED` 是 M1 时代的开关，已经在大量部署里有值。直接重命名会让所有老 .env 失效。用 model_validator 自动迁移让两个开关并存，新部署用新字段，老部署不用动。

### 3. LECTURE_STRICT_AGENTS 为什么默认 false

生产环境用户的核心需求是「拿到结果」，不是「完美结果」。即便 v2 失败了，v1 fallback 也能给一份可用 HTML。开发态打开 strict 是为了立刻看到错误堆栈，反而是少数场景。**默认值应该匹配多数场景**。

### 4. 为什么 LECTURE_MAP_PROMPT_VERSION 这种字符串字段也要做配置

prompt 模板修改后想立刻让所有 chapter cache 失效，最干净的做法是 bump 版本号。版本号通过 `compute_prompt_hash` 进 SHA-256，自动让旧 row 全 miss。把它做成 env 字段后，运维不用动代码也能强制一次缓存清理。

### 5. 为什么 24000 字符的 LECTURE_CRITIC_MAX_PROMPT_CHARS 是「字幕摘录」上限而不是「整 prompt」上限

`config.py:540-554` 注释直白：「IR JSON 是审计对象，无法压缩」。所以 Critic 上限只能裁字幕，不能裁 IR。配置项语义精确到「这是裁哪部分」让用户不会误以为整 prompt 被截断了。

## 当前文档中的事实与推断

### 事实

- `app/config.py` 共 738 行，12 类配置。
- `.env.example` 12914 字节。
- `_apply_reviser_mode_compat` 实现 enabled → mode 自动映射。
- `text_extra_body()` 同时支持 DashScope 和 DeepSeek 两种 API。
- 至少 24 处运行时降级路径覆盖整条流水线。

### 推断

- 这一层是项目「工程化交付」最不显眼但最有价值的部分：它让一个 50+ 字段的 LLM 系统能被一个 .env 文件、几个 strict 开关、几个 mode 字段精确控制，是从「能跑」到「可部署」的关键桥梁。
