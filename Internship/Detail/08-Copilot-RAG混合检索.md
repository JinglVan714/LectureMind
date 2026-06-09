# Copilot RAG：混合检索

## 文档定位

`Internship/03` 提到了「混合检索 + chunk 类型 + 锚点校验」，本篇展开 `app/copilot/rag.py`（1023 行）的完整设计，包括：

- 7 种 chunk 类型怎么定的？
- DashScope MultiModalEmbedding 怎么集成？
- sqlite-vec + FTS5 + RRF 怎么联动？
- Embedding 维度怎么验证、不一致怎么自愈？
- 失败时怎么自动降级到 FTS-only？

## 一句话定位

`RAGStore` 是一个**单 SQLite 文件、单连接、async 包裹、向量 + 全文混合检索**的轻量级检索引擎。它和主库 `data/app.db` 共用一个文件，通过 `sqlite-vec` 扩展加载向量虚拟表，加上 FTS5 全文索引，最后用 RRF 融合两路结果。

## 模块构成

`app/copilot/rag.py` 一个文件 1023 行，三个核心 class：

- **`chunk_lecture(lecture, lecture_ir)`** — 把 `LectureJSON` 拆成 chunk 列表。
- **`EmbeddingClient`** — 异步包装 `dashscope.MultiModalEmbedding`。
- **`RAGStore`** — 持久化 + 检索。

## Chunking：7 种 chunk 类型

`@d:\Diet_Agent_NEW\app\copilot\rag.py:50-72`：

```python
ChunkKind = Literal[
    "teaching_note",   # 章节教学笔记段落
    "quote",           # 章节论点 + 原字幕引用
    "pitfall",         # 章节陷阱
    "knowledge_unit",  # 知识单元
    "frame_ocr",       # 关键帧 caption / OCR
    "code_block",      # v2 一等公民
    "formula_block",   # v2 一等公民
    "study_question",  # Question-Driven 阶段产物
]
```

每种 chunk 都对应**讲义的某种语义结构**，而不是任意文本切片。这一点是项目区别于通用 RAG 的关键。

### `chunk_lecture` 的两条路径

`@d:\Diet_Agent_NEW\app\copilot\rag.py:97-337` 的 `chunk_lecture(lecture, lecture_ir)`：

- **路径 A**（推荐）：当 `lecture.evidence_index` 存在时，调 `_evidence_index_chunks(lecture)`，直接读 `EvidenceObject.rag_chunks`。这意味着**双投影编译已经做完 chunking 决策**，索引器只是把它落库。
- **路径 B**（fallback）：没有 evidence_index 时，手工拼章节 / point / pitfall / code / formula / frame / KU / visual_evidence / study_question 9 类来源。

路径 A 是 v2 IR 的常规路径，路径 B 是 v1 fallback 的兼容路径。

### v2 路径里的 code_block chunk

`@d:\Diet_Agent_NEW\app\copilot\rag.py:188-216`：

```python
text = f"[code:{language}]\n{code}"
if explanation:
    text = f"{text}\n说明：{explanation}"
out.append(Chunk(
    bv_id, kind="code_block", text=text, chapter_idx, t_start=ts, t_end=ts,
    meta={"language": ..., "code": ..., "source": ..., "explanation": ...},
))
```

注意 `text` 字段把 **code + explanation 拼在一起**，目的是让向量检索能命中自然语言查询（「show me the loss code」），同时 BM25 能命中标识符（`def loss_fn`）。`meta.code` 保留原始代码，渲染时可以恢复缩进与高亮。

### frame_ocr 的合并文本

`@d:\Diet_Agent_NEW\app\copilot\rag.py:384-396` 的 `_frame_text(caption, insight, ocr_text)`：

```python
parts = []
seen = set()
for piece in (caption, insight, ocr_text):
    text = piece.strip()
    if not text or text.lower() in seen:
        continue
    seen.add(text.lower())
    parts.append(text)
return " | ".join(parts)
```

去重 + 大小写无关 + 用 `|` 拼。一张帧只生成一个 chunk，文本是 caption / insight / ocr_text 的并集（去掉重复）。

## Embedding：DashScope MultiModalEmbedding

`@d:\Diet_Agent_NEW\app\copilot\rag.py:415-523` 的 `EmbeddingClient`：

```python
class EmbeddingClient:
    def __init__(self, *, model=None, api_key=None, max_concurrent=None):
        s = get_settings()
        self._model = model or s.qwen_embedding_model
        self._api_key = api_key or s.dashscope_api_key
        self._sem = asyncio.Semaphore(max_concurrent or s.copilot_max_concurrent or 4)
        self._dim: int | None = None
```

### 同步 SDK + asyncio.to_thread

DashScope `MultiModalEmbedding` 是同步 SDK，所以每次调用都通过 `asyncio.to_thread(self._call_text, text)` 派发到 worker 线程，外层 `asyncio.Semaphore(4)` 限流。

为什么**逐条调用**而不是批量？docstring 说：「多条 batch 的语义在 DashScope 上不稳定（可能返回 N 个向量，也可能返回一个加权平均）」，逐条最确定。

### Dim 探测 + 一致性校验

`_extract_vec(resp)`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:503-523`）：

```python
if self._dim is None:
    self._dim = len(vec)
elif len(vec) != self._dim:
    raise EmbeddingError(f"embedding dim drift: expected {self._dim}, got {len(vec)}")
```

第一次调用就锁定维度，后续每个返回都验证。这是「模型偶尔被服务端切换」的防御层。

## RAGStore：单文件 + 单连接

### 为什么用同步 `sqlite3.Connection`

`@d:\Diet_Agent_NEW\app\copilot\rag.py:559-567`：

```python
def __init__(self, db_path, embedder=None):
    self._db_path = db_path
    self._embedder = embedder or EmbeddingClient()
    self._conn: sqlite3.Connection | None = None
    self._lock = asyncio.Lock()
    self._vec_available = False
    self._vec_table_ready = False
    self._dim: int | None = None
```

原因写在 docstring：`aiosqlite` 不暴露 `enable_load_extension`。`sqlite-vec` 是 C 扩展，必须 `enable_load_extension(True)` 才能 `sqlite_vec.load(conn)`。所以这层用同步 connection + `asyncio.Lock` + `asyncio.to_thread` 包装。

### 加载 sqlite-vec 的容错

`_sync_init`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:596-634`）：

```python
try:
    conn.enable_load_extension(True)
except sqlite3.NotSupportedError:
    # 编译版 sqlite3 没开 enable_load_extension → 整库降级到 FTS-only
    self._vec_available = False
    ...
    return

try:
    import sqlite_vec
    sqlite_vec.load(conn)
    self._vec_available = True
except Exception:
    self._vec_available = False  # → FTS-only
```

两层 try-except：

- 编译版 SQLite 不支持加载扩展 → FTS-only 模式。
- 加载扩展本身报错 → FTS-only 模式。

主链路启动时（`@d:\Diet_Agent_NEW\app\main.py:42-47`）若 RAGStore 初始化失败，也只是 logger.warning，**不阻断 FastAPI 启动**。

### vec0 虚拟表

`@d:\Diet_Agent_NEW\app\copilot\rag.py:644-654`：

```python
def _ensure_vec_table(self, conn, dim):
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS lecture_chunk_vecs USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        )
    """)
```

注意 `FLOAT[<dim>]` 不能 parameter binding，只能字符串拼接。这是 sqlite-vec 的硬约束。

### upsert_chunks：先 embed 后 mutate

`@d:\Diet_Agent_NEW\app\copilot\rag.py:658-676`：

```python
async def upsert_chunks(self, bv_id, chunks):
    await self.init()
    if not chunks:
        await asyncio.to_thread(self._sync_purge_bv, bv_id)
        return 0

    # 先 embed，让 EmbeddingError 在 mutate 之前抛出
    vectors = None
    if self._vec_available:
        vectors = await self._embedder.embed_text([c.text for c in chunks])
        await self._record_dim(self._embedder.dim)

    return await asyncio.to_thread(self._sync_upsert, bv_id, list(chunks), vectors)
```

**先 embed 再写表**是关键。如果先 delete 老 chunks 再 embed 失败，BV 的索引就丢了。先 embed 拿到所有向量再开事务，是「事务边界 + 故障隔离」的标准做法。

### Embedding dim 漂移自愈

`_record_dim`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:764-787`）：

```python
async def _record_dim(self, dim):
    if dim == self._dim:
        return
    if self._dim is not None and dim != self._dim:
        raise EmbeddingError(
            f"embedding_dim mismatch: stored={self._dim}, new={dim}; "
            f"run scripts/reindex_copilot.py --reset-vectors to rebuild."
        )
    self._dim = dim
    # 写入 kv 表（embedding_dim / embedding_model）
```

`kv` 表把维度持久化，进程重启后能从 kv 还原 dim，避免「重启后第一次 embed 就建错维度的表」。维度切换时**主动报错**，要求用户走 `reindex_copilot.py --reset-vectors` 重建——比静默失败安全。

### 删除老 chunks 的事务

`_sync_upsert`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:699-762`）：

```python
conn.execute("BEGIN")
try:
    # 删老 chunks
    if old_ids:
        conn.execute(f"DELETE FROM lecture_chunks WHERE chunk_id IN ({...})", old_ids)
        if self._vec_available and self._vec_table_ready:
            conn.execute(f"DELETE FROM lecture_chunk_vecs WHERE chunk_id IN ({...})", old_ids)

    # 写新 chunks（INSERT 触发 FTS5 trigger 自动同步）
    new_ids = []
    for chunk in chunks:
        cur = conn.execute("INSERT INTO lecture_chunks (...) VALUES (...)", ...)
        new_ids.append(cur.lastrowid)

    # 写向量
    if vectors:
        for chunk_id, vec in zip(new_ids, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            conn.execute("INSERT INTO lecture_chunk_vecs (chunk_id, embedding) VALUES (?, ?)", (chunk_id, blob))

    conn.execute("COMMIT")
except Exception:
    conn.execute("ROLLBACK")
    raise
```

整个 upsert 是一个 SQL 事务。FTS5 表通过 `app/storage/db.py` 里的 `lecture_chunks_ai / ad / au` 三个 TRIGGER 自动同步，不用手工维护。

### `struct.pack(f"{len(vec)}f", *vec)` —— 32-bit float 二进制布局

sqlite-vec 期望 `FLOAT[<dim>]` 列存储为 little-endian 32-bit float 的 raw bytes。Python `struct.pack(f"{N}f", ...)` 直接出这个格式。

## search：三路融合

`@d:\Diet_Agent_NEW\app\copilot\rag.py:791-837`：

```python
async def search(self, query, *, bv_id=None, top_k=5):
    await self.init()
    if not query.strip():
        return []

    # 1. 向量检索 top-20（如有）
    vec_hits = []
    if self._vec_available and self._vec_table_ready:
        try:
            qvec = (await self._embedder.embed_text([query]))[0]
            await self._record_dim(self._embedder.dim)
            vec_hits = await asyncio.to_thread(self._sync_vec_search, qvec, bv_id, _VEC_TOPK)
        except Exception:
            vec_hits = []  # 向量失败 → 不阻断，让 FTS 兜底

    # 2. FTS5 检索 top-20
    fts_hits = await asyncio.to_thread(self._sync_fts_search, query, bv_id, _FTS_TOPK)

    # 3. RRF 融合
    merged = _rrf_merge(vec_hits, fts_hits, k=_RRF_K)
    chunk_ids = [cid for cid, _ in merged[:top_k]]

    # 4. Hydrate
    rows = await asyncio.to_thread(self._sync_hydrate, chunk_ids)
    return rows[:top_k]
```

### 向量检索

`_sync_vec_search`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:839-870`）：

- `WHERE embedding MATCH ? AND k = ?` 是 sqlite-vec 的 KNN 查询语法。
- `oversample = k * 5` 当指定 bv_id 时（因为要后过滤）。
- 后过滤 `bv_id` 用 `JOIN lecture_chunks ON chunk_id` 因为 `vec0` 表里没有 bv_id 字段。

### FTS5 检索

`_sync_fts_search`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:872-901`）：

```sql
SELECT lecture_chunk_fts.rowid AS chunk_id, bm25(lecture_chunk_fts) AS rank
FROM lecture_chunk_fts
JOIN lecture_chunks ON lecture_chunks.chunk_id = lecture_chunk_fts.rowid
WHERE lecture_chunk_fts MATCH ?  -- _escape_fts_query 转义后
[AND lecture_chunks.bv_id = ?]
ORDER BY rank LIMIT ?
```

`_escape_fts_query`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:985-1003`）把用户的中英混合 query 转成安全的 FTS5 MATCH 表达式：

- 按 whitespace split。
- 丢弃 `AND / OR / NOT / NEAR`（FTS5 保留词）。
- 每个 token 用双引号包起来（避免特殊字符炸 parser）。
- 用空格拼回（FTS5 默认 AND）。

### RRF 融合

`_rrf_merge`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:966-982`）：

```python
def _rrf_merge(vec, fts, *, k=60):
    score = {}
    for rank, (cid, _) in enumerate(vec):
        score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(fts):
        score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)
```

经典 Reciprocal Rank Fusion 公式 `1 / (k + rank + 1)`，`k=60` 是论文默认值。

注意：**两路输入的数值分数（distance / BM25 rank）完全不用**，只用排序。这避免了「向量 distance 和 BM25 rank 数值不可直接比较」的麻烦。

### Hydrate

`_sync_hydrate`（`@d:\Diet_Agent_NEW\app\copilot\rag.py:903-931`）按 chunk_id 反查完整 row，解出 `meta_json`，把 `quote` 字段从 meta 里 hoist 到顶层（让 quote / frame_ocr 类型的 chunk 不需要二次解 JSON）。

## 索引入口：`index_lecture`

`@d:\Diet_Agent_NEW\app\copilot\indexer.py` 是一段非常薄的胶水：

```python
async def index_lecture(rag, bv_id, lecture, lecture_ir=None) -> int:
    try:
        chunks = chunk_lecture(lecture, lecture_ir)
        if not chunks:
            return 0
        n = await rag.upsert_chunks(bv_id, chunks)
        return n
    except Exception as exc:
        logger.warning("Copilot indexing failed for %s: %s", bv_id, exc, exc_info=True)
        return 0
```

「best-effort」——任何异常都吞掉只 warn，**讲义渲染不会因 RAG 失败而失败**。这是 Pipeline 集成的关键解耦。

主链路在 `@d:\Diet_Agent_NEW\app\pipeline.py:347-378` 调它：

```python
wait_rag = settings.pipeline_wait_rag  # 默认 True
async def _do_rag_index() -> int | None:
    try:
        return await index_lecture(self.rag, meta.bv_id, lecture, lecture_ir)
    except Exception:
        return None

if wait_rag:
    await _do_rag_index()
else:
    self.last_run_rag_task = asyncio.create_task(_bg_rag())
```

`PIPELINE_WAIT_RAG=false` 时索引在后台跑，HTML 渲染完就返回，让首页响应更快。

## reset_vectors：维度切换的兜底

`@d:\Diet_Agent_NEW\app\copilot\rag.py:935-958`：

```python
async def reset_vectors(self):
    await self.init()
    async with self._lock:
        await asyncio.to_thread(self._sync_reset_vectors)
        self._vec_table_ready = False
        self._dim = None

def _sync_reset_vectors(self):
    conn.execute("DROP TABLE IF EXISTS lecture_chunk_vecs")
    conn.execute("DELETE FROM kv WHERE key IN ('embedding_dim', 'embedding_model')")
```

切换 embedding 模型（不同维度）时调用：删向量表 + 清 kv，下次 `upsert_chunks` 会用新维度重建表。这套路径由 `scripts/reindex_copilot.py --reset-vectors` 触发。

## 写简历可以怎么提炼

- 实现了基于 `sqlite-vec` + FTS5 + RRF 的混合检索，单 SQLite 文件管理向量与全文索引，并把 `enable_load_extension` 失败 / `sqlite-vec` 加载失败 / EmbeddingClient 失败三类异常分别降级到 FTS-only 模式，让 RAG 不可用时主链路依然能跑。
- 设计了 7 种与讲义结构对齐的 chunk kind（teaching_note / quote / pitfall / knowledge_unit / frame_ocr / code_block / formula_block / study_question），让向量检索和 BM25 都能命中各自最擅长的内容形态。
- 在 `EmbeddingClient` 里实现了首调用探测维度 + 后续校验，并在 `RAGStore.upsert_chunks` 里坚持「先 embed 后 mutate」，让网络抖动和模型切换都不会让索引数据进入不一致状态；维度切换通过 `reset_vectors` + kv 清理走显式重建路径。
- 用 `struct.pack(f"{N}f", *vec)` 把 Python `list[float]` 编码为 sqlite-vec 期望的 raw bytes，避免依赖 NumPy。

## 面试可展开点

### 1. 为什么把向量和全文都塞进一个 SQLite

LectureMind 的部署目标是本地 / 自托管，引入 Postgres + pgvector / Milvus / Qdrant 都会增加运维负担。SQLite 在单机场景下完全够用，且 `sqlite-vec` + FTS5 已经能做到向量 + 全文混合检索。一份 `data/app.db` 备份就能搬走所有数据。

### 2. 为什么要做混合检索

视频讲义的查询语义很多样：「损失函数怎么算」是纯自然语言（向量擅长），「QKV 的 softmax」是术语 + 公式片段（BM25 擅长），「[t=05:46] 的关键帧讲了什么」是结构化锚点。任何单一检索方式都会漏一类查询。混合 + RRF 让两路结果互补。

### 3. 为什么 RRF 不用 distance / BM25 分数本身做权重

向量 distance 和 BM25 rank 单位不可比，强行加权会让两路里数值范围更小的那一路总是赢。RRF 只用「排名」让两路平等竞争，是「无脑融合」里效果最稳定的一种。

### 4. 为什么 chunk 是「与讲义结构对齐」而不是定长切片

定长切片会把一句完整的引用切到两个 chunk，导致检索时只命中一半信息。按讲义结构切让每个 chunk 都是一个**语义完整单元**，命中后的 hydrate 也能直接拿到锚点 / meta，不需要拼回上下文。

### 5. 为什么 `upsert_chunks` 要先 embed 再 mutate

如果先 delete 老 chunks，embed 中途失败，索引就丢了。先 embed 拿到全部向量再开事务，要么整批新数据进库要么完全不变。这是「事务边界 + 故障隔离」的标准模式，类比 git 的 staged commit。

## 当前文档中的事实与推断

### 事实

- 7 种 chunk kind 定义在 `@d:\Diet_Agent_NEW\app\copilot\rag.py:50-72`。
- `chunk_lecture` 优先用 `evidence_index.rag_chunks`，否则手工拼。
- `EmbeddingClient` 每条 text 单独调一次 DashScope，限流 4 路。
- `RAGStore` 用 `sqlite-vec` + FTS5 + RRF 混合检索。
- 三层降级：扩展加载失败 / sqlite-vec 加载失败 / 向量调用失败都退回 FTS-only。
- `PIPELINE_WAIT_RAG` 控制索引是否阻塞主链路。

### 推断

- 把整套 RAG 压在一个 SQLite 文件里、且每一层都能优雅降级，是项目「本地自托管 + 单机交付」目标下最务实的设计。
