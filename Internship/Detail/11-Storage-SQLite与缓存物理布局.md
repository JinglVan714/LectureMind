# Storage：SQLite 与缓存物理布局

## 文档定位

`Internship/04` 简单提了「SQLite schema + 集中配置」，本篇展开整套存储系统的物理布局：

- 主库 `data/app.db` — `app/storage/db.py`。
- VLM 缓存 `data/vlm_cache.sqlite` — `app/understand/vlm_cache.py`。
- 章节缓存 `data/chapter_cache.sqlite` — `app/understand/chapter_cache.py`。
- 文件资产 `data/{reports,keyframes,subtitles,audio,cookies,debug}/`。

## 一句话定位

LectureMind 的存储完全本地化：一份主库 + 两份缓存库 + 几个文件目录，全部位于 `DATA_DIR`（默认 `./data`）下。**一个 tar 备份就能搬走所有数据**，没有外部数据库依赖。

## 物理布局总览

```
data/
├── app.db                       # 主库（summaries / assets / jobs / kv / lecture_chunks / *_fts / *_vecs）
├── vlm_cache.sqlite             # 内容寻址：(sha256, model, tier) → VLM 描述
├── chapter_cache.sqlite         # prompt_hash → 章节 LLM 输出
├── reports/                     # 生成的 HTML 讲义
│   ├── BV1xxx.html
│   └── assets/katex/...         # 离线 KaTeX 资源
├── keyframes/<BV>/*.jpg         # 每 BV 一个目录
├── subtitles/                   # CC 原始 JSON + Whisper 缓存
│   ├── BV1xxx.cc.json
│   └── BV1xxx.whisper.json
├── audio/                       # Whisper 转录的中间音频
│   └── BV1xxx.m4a
├── cookies/                     # Bilibili Netscape 格式 cookie
│   └── bilibili.txt
└── debug/                       # 验证 / 矩阵脚本输出
    ├── BV1xxx.lecture_ir.json   # IR 原始 dump
    ├── BV1xxx.timing.json       # 单次 run timing
    └── lecture_matrix_*.{json,md}
```

`Settings.ensure_dirs()`（`@d:\Diet_Agent_NEW\app\config.py:721-730`）在启动时自动创建这些目录。

## 主库 app.db：6 张表 + 2 个虚拟表

### summaries

`@d:\Diet_Agent_NEW\app\storage\db.py:19-38`：

```sql
CREATE TABLE summaries (
  bv_id        TEXT PRIMARY KEY,
  url          TEXT NOT NULL,
  title        TEXT,
  author       TEXT,
  duration     INTEGER,
  cover_url    TEXT,
  category     TEXT DEFAULT 'lecture',
  domain_tags  TEXT DEFAULT '[]',
  domain       TEXT,         -- 后续追加
  direction    TEXT,         -- 后续追加
  summary_json TEXT NOT NULL,
  report_path  TEXT NOT NULL,
  model_used   TEXT,
  token_cost   INTEGER DEFAULT 0,
  status       TEXT DEFAULT 'done',
  error_msg    TEXT,
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

`summary_json` 是**完整的 `LectureJSON.model_dump(mode="json")`**——含双投影、composition、taxonomy 全部字段。这意味着一行 summary 行就能完整还原一份讲义，不需要联合其他表。

`status` 取值：`done / running / error`。Pipeline 缓存命中检查的就是 `status == "done"`。

### assets

`@d:\Diet_Agent_NEW\app\storage\db.py:40-46`：

```sql
CREATE TABLE assets (
  bv_id TEXT NOT NULL,
  kind  TEXT NOT NULL,        -- 'subtitle' / 'keyframe' / ...
  path  TEXT NOT NULL,
  meta  TEXT DEFAULT '{}',    -- JSON: {ts, source, language, ...}
  PRIMARY KEY (bv_id, kind, path)
);
```

三元组主键 `(bv_id, kind, path)` 让 `INSERT OR REPLACE` 自动幂等。每个关键帧文件、每份字幕都会在这里有一行。

### jobs

`@d:\Diet_Agent_NEW\app\storage\db.py:48-56`：

```sql
CREATE TABLE jobs (
  job_id     TEXT PRIMARY KEY,
  bv_id      TEXT,
  status     TEXT NOT NULL,     -- 'queued' / 'running' / 'done' / 'error'
  progress   INTEGER DEFAULT 0,
  error_msg  TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Web `POST /api/summarize` 创建 job 后立即返回 `job_id`，让前端走 `GET /api/jobs/{job_id}` 轮询进度。

### lecture_chunks + lecture_chunk_fts (FTS5) + lecture_chunk_vecs (sqlite-vec)

`@d:\Diet_Agent_NEW\app\storage\db.py:58-87`：

```sql
CREATE TABLE lecture_chunks (
  chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  bv_id        TEXT NOT NULL,
  kind         TEXT NOT NULL,       -- 7 种 ChunkKind
  chapter_idx  INTEGER,
  t_start      INTEGER,
  t_end        INTEGER,
  text         TEXT NOT NULL,
  meta_json    TEXT DEFAULT '{}',
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE lecture_chunk_fts USING fts5(
  text,
  content='lecture_chunks', content_rowid='chunk_id',
  tokenize='unicode61'
);

-- lecture_chunk_vecs 由 RAGStore 在第一次 upsert 时按 embedding dim 动态建：
-- CREATE VIRTUAL TABLE lecture_chunk_vecs USING vec0(
--     chunk_id INTEGER PRIMARY KEY,
--     embedding FLOAT[<dim>]
-- );
```

详见 `Detail/08-Copilot-RAG.md`。

### FTS5 三个 trigger

`@d:\Diet_Agent_NEW\app\storage\db.py:102-116`：

```sql
CREATE TRIGGER lecture_chunks_ai AFTER INSERT ON lecture_chunks BEGIN
  INSERT INTO lecture_chunk_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
CREATE TRIGGER lecture_chunks_ad AFTER DELETE ON lecture_chunks BEGIN
  INSERT INTO lecture_chunk_fts(lecture_chunk_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
END;
CREATE TRIGGER lecture_chunks_au AFTER UPDATE ON lecture_chunks BEGIN
  INSERT INTO lecture_chunk_fts(lecture_chunk_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
  INSERT INTO lecture_chunk_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
```

三个 trigger 让 FTS5 索引自动同步——`lecture_chunks` 上的 INSERT / DELETE / UPDATE 都会镜像到 `lecture_chunk_fts`。Python 代码完全不需要手工维护 FTS。

### kv

`@d:\Diet_Agent_NEW\app\storage\db.py:74-78`：

```sql
CREATE TABLE kv (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

通用 key-value 元数据，目前已用到的 key：

- `embedding_dim` — RAG 向量维度（让进程重启后能从这里还原）。
- `embedding_model` — 用了哪个 embedding 模型。
- `last_reindex_at` — 最近一次 reindex 时间。

设计动机：Stage 2 自由演进 schema，不用每次都改 DDL。

## 数据库初始化：两阶段 + 迁移

### Database.init

`@d:\Diet_Agent_NEW\app\storage\db.py:207-213`：

```python
async def init(self) -> None:
    async with aiosqlite.connect(self.db_path) as db:
        await db.executescript(SCHEMA_TABLES)
        await self._migrate_summaries(db)
        await db.executescript(SCHEMA_AUX)
        await db.commit()
```

两阶段切分原因（`@d:\Diet_Agent_NEW\app\storage\db.py:16-91` 注释明确）：

- `SCHEMA_TABLES` 只建基础表，不引用任何会被 migrate 进来的新列。
- `_migrate_summaries` 检查 `PRAGMA table_info(summaries)`，缺什么 column 就 `ALTER TABLE ADD COLUMN`。
- `SCHEMA_AUX` 建索引和 trigger（其中索引 / trigger 会引用新加的列）。

这样老数据库（没有 `domain` / `direction` 列的版本）也能平滑升级——加列、再建对应索引。

### Migration 表

`@d:\Diet_Agent_NEW\app\storage\db.py:125-128`：

```python
_SUMMARIES_MIGRATIONS = (
    ("domain", "ALTER TABLE summaries ADD COLUMN domain TEXT"),
    ("direction", "ALTER TABLE summaries ADD COLUMN direction TEXT"),
)
```

未来加列继续往这里加 tuple 就行。`_migrate_summaries` 只跑缺的列，已存在的列跳过。

## SummaryRow dataclass

`@d:\Diet_Agent_NEW\app\storage\db.py:131-174`：

```python
@dataclass
class SummaryRow:
    bv_id: str
    url: str
    title: str | None
    author: str | None
    duration: int | None
    cover_url: str | None
    category: str
    domain_tags: list[str]
    summary_json: dict[str, Any]
    report_path: str
    model_used: str | None
    token_cost: int
    status: str
    error_msg: str | None
    created_at: str
    updated_at: str
    domain: str | None = None
    direction: str | None = None
```

`from_row(row)` 把 aiosqlite Row 转 dataclass。注意 `domain_tags` 用 `json.loads(row["domain_tags"] or "[]")` 解码，`summary_json` 同理。**老 schema 没有 `domain` / `direction` 列时安全返回 None**（`if "domain" in keys`）。

## DAO 接口

`@d:\Diet_Agent_NEW\app\storage\db.py:235-412` 的 `Database` 类暴露异步接口：

- **summaries**:
  - `get_summary(bv_id) -> SummaryRow | None`
  - `list_summaries(limit, offset) -> list[SummaryRow]`
  - `upsert_summary(...)` — `INSERT ... ON CONFLICT(bv_id) DO UPDATE`
  - `update_taxonomy(bv_id, domain, direction, domain_tags)` — 只 patch taxonomy 三列，不动其他
- **kv**: `kv_get(key)` / `kv_set(key, value)`
- **assets**: `add_asset(bv_id, kind, path, meta)` — INSERT OR REPLACE 三元组
- **jobs**: `create_job` / `update_job` / `get_job`

### 单 writer 协议

每个方法都用 `async with self._conn() as db: ... await db.commit()`。SQLite 在 WAL 模式下允许并发 reader + 单 writer。Pipeline 的 `MAX_CONCURRENT_JOBS` semaphore + 这里的 single-writer 模式共同保证不会出 BUSY error。

## VLM 缓存物理表

`@d:\Diet_Agent_NEW\app\understand\vlm_cache.py:64-77`（详见 `Detail/02-VLM`）：

```sql
CREATE TABLE vlm_cache (
    image_sha256 TEXT NOT NULL,
    model        TEXT NOT NULL,
    tier         TEXT NOT NULL CHECK (tier IN ('high','low')),
    payload      TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (image_sha256, model, tier)
);
CREATE INDEX idx_vlm_cache_created ON vlm_cache(created_at);
```

WAL + `synchronous=NORMAL`（`@d:\Diet_Agent_NEW\app\understand\vlm_cache.py:304-320`）。

## ChapterCache 物理表

`@d:\Diet_Agent_NEW\app\understand\chapter_cache.py:142-164`（详见 `Detail/03-长度路由`）：

```sql
CREATE TABLE chapter_cache (
    prompt_hash    TEXT PRIMARY KEY,
    chapter_json   TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    bv_id          TEXT NOT NULL,
    chapter_index  INTEGER NOT NULL,
    chapter_start  REAL NOT NULL,
    chapter_end    REAL NOT NULL,
    created_at     REAL NOT NULL,
    accessed_at    REAL NOT NULL,
    hit_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_chapter_cache_bv ON chapter_cache(bv_id);
CREATE INDEX idx_chapter_cache_model_version ON chapter_cache(model_id, prompt_version);
```

## 为什么分三个 SQLite 文件而不是一个

三个数据库的**语义和生命周期完全不同**：

| 文件 | 寻址方式 | 生命周期 | 清理时机 |
|------|---------|---------|---------|
| `app.db` | 按 BV / job 实体 | 永久 | 用户主动删 |
| `vlm_cache.sqlite` | 按内容 hash 寻址 | 长期（跨 BV 复用） | 模型升级时 |
| `chapter_cache.sqlite` | 按 prompt_hash 寻址 | 中期（章节窗口稳定时复用） | prompt_version bump 时 |

把它们分文件让：

- 备份 / 删除 / 迁移可以独立做（`rm vlm_cache.sqlite` 让所有视觉描述失效但不动主数据）。
- 写入并发不互相影响（VLM 缓存的高并发写不会阻塞 summaries 主表读）。
- 文件大小可观察（一份 `du -sh data/*.sqlite` 就能看每种缓存占多少盘）。

## ensure_dirs：启动时建目录

`@d:\Diet_Agent_NEW\app\config.py:696-730` 的 `Settings.ensure_dirs()`：

```python
def ensure_dirs(self) -> None:
    for p in (self.data_dir, self.reports_dir, self.keyframes_dir,
              self.subtitles_dir, self.audio_dir, self.cookies_dir):
        p.mkdir(parents=True, exist_ok=True)
```

`get_settings()` 用 `@lru_cache(maxsize=1)` 包了，首次调用建一次目录就完事。

## 数据可移植性

LectureMind 的核心承诺是「自托管 + 本地可控」。三层证据：

- **零外部数据库依赖**：SQLite 文件存所有结构化数据。
- **HTML 报告自包含**：`data/reports/*.html` 内联 CSS + base64 封面 + 离线 KaTeX，可以直接发到任何邮箱 / 即时通讯里。
- **rerender_reports**：模板改动后 `python -m scripts.rerender_reports` 直接从 `summary_json` 重渲染所有历史报告——不需要重调 LLM。

## 写简历可以怎么提炼

- 设计了「主库 `app.db` + 内容寻址 VLM 缓存 + prompt_hash 章节缓存」三库分离的本地化存储架构，让每种缓存生命周期独立可控、写入互不影响，并通过 SQLite WAL 模式支持单 writer + 并发 reader。
- 把 `LectureJSON` 完整序列化到 `summaries.summary_json` 单列，让一行 summary 就能完整还原讲义、HTML 和 RAG chunk 全部内容，支持「不调 LLM 的纯重渲染」能力。
- 通过 SQLite FTS5 `lecture_chunk_fts` 虚拟表 + 三个自动同步 trigger（INSERT / DELETE / UPDATE），让 Python 代码不必维护全文索引；并通过 `kv` 元数据表持久化 embedding dim / model / last_reindex_at，让进程重启后状态可恢复。
- 用两阶段 schema 初始化（`SCHEMA_TABLES` → `_migrate_summaries` → `SCHEMA_AUX`）实现老库平滑升级，避免新增列引发的级联 schema 重建。

## 面试可展开点

### 1. 为什么不用 PostgreSQL + pgvector

LectureMind 的部署目标是本地 / 自托管。Postgres + pgvector 需要额外的服务、配置、备份策略。SQLite + sqlite-vec + FTS5 单文件搞定了等价能力，对单用户 / 小团队场景足够；如果未来要规模化，可以把 RAG 层抽出来换 backend，主库结构不用改。

### 2. 为什么 summary_json 一列存这么多东西

JSON 列 + 单行 = 「一次写入，多种消费」。HTML 渲染、Copilot 检索、MCP 工具调用、`reindex` 重建索引，都基于这一列。如果拆成多张关联表，每次新增字段都要改 schema + 数据迁移，工程成本巨大。LectureJSON 通过 Pydantic schema 演进，未来加字段对 DB 透明。

### 3. 为什么 FTS5 用 trigger 而不是手写同步

trigger 是 SQLite 内部事务的一部分——主表 INSERT 成功，FTS 一定同步成功。手工同步要在每个 DAO 方法里加代码、考虑事务边界、处理失败回滚。trigger 是「数据库本身保证一致」，可靠性更高。

### 4. 为什么三个 SQLite 文件而不是三套表

物理分离让一份缓存挂掉（磁盘满 / 错误 schema）不会拖死整个系统。`vlm_cache.sqlite` 如果写入失败，主库的 summary 写入完全不受影响。分文件也让用户能精确控制缓存策略——「我要清 VLM 缓存重跑视觉理解」直接 `rm vlm_cache.sqlite`。

### 5. WAL 模式 + 单 writer 的取舍

WAL 让多个 reader 并发不互相阻塞，写也只锁定单条记录的事务。代价是数据库文件之外多一个 `-wal` 文件，需要 checkpoint。对单机自托管场景这是个非常合适的折中——比起强一致性，吞吐和响应性更重要。

## 当前文档中的事实与推断

### 事实

- 主库表：summaries / assets / jobs / lecture_chunks / kv + FTS5 虚拟表 + vec0 虚拟表（懒建）。
- 三个 SQLite 文件物理分离：`app.db` / `vlm_cache.sqlite` / `chapter_cache.sqlite`。
- FTS5 通过 3 个 trigger 自动同步。
- 两阶段 schema 初始化：基础表 → migration → 索引/trigger。
- `summary_json` 单列存完整 LectureJSON（含双投影）。

### 推断

- 这套存储设计是「单机自托管 + 工程化最少依赖」的样板：没有 Redis / Postgres / Milvus，但功能完备。
