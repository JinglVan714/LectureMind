"""Stage 1 tests — Copilot taxonomy normalisation.

These tests deliberately import nothing from the pipeline / FastAPI layer
so they can run alongside ``test_smoke.py`` without spinning up a DB or
the LLM client.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Match the fixture isolation pattern in test_smoke.py: any code that
# touches Settings during import will pick up a sandboxed DATA_DIR.
_tmp = tempfile.mkdtemp(prefix="lecturemind-copilot-test-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("BASIC_AUTH_PASSWORD", "test-pwd")

import sqlite3  # noqa: E402

import pytest  # noqa: E402

from app.copilot.rag import (  # noqa: E402
    Chunk,
    EmbeddingClient,
    RAGStore,
    chunk_lecture,
)
from app.copilot.taxonomy import (  # noqa: E402
    DOMAIN_WHITELIST,
    needs_review,
    normalize,
)
from app.storage.db import Database  # noqa: E402
from app.understand.schema import (  # noqa: E402
    Chapter,
    EvidenceObject,
    EvidenceIndex,
    Frame,
    KnowledgeUnitView,
    LectureJSON,
    LectureNoteIR,
    Point,
    Taxonomy,
)


class TestTaxonomyNormalize:
    def test_none_in_none_out(self):
        assert normalize(None) is None

    def test_unknown_domain_falls_back_to_other_with_low_confidence(self):
        t = Taxonomy(domain="量子计算", direction="量子纠错", tags=["QEC"], confidence=0.9)
        out = normalize(t)
        assert out is not None
        assert out.domain == "其他"
        # Confidence ceiling is 0.3 when the LLM drifted off the white-list.
        assert out.confidence == 0.3

    def test_known_domain_kept_with_confidence(self):
        t = Taxonomy(
            domain="AI 技术",
            direction="注意力机制",
            tags=["Transformer", "Attention"],
            confidence=0.85,
        )
        out = normalize(t)
        assert out is not None
        assert out.domain == "AI 技术"
        assert out.direction == "注意力机制"
        assert out.confidence == 0.85
        assert out.tags == ["Transformer", "Attention"]

    def test_direction_strips_stage_suffixes(self):
        t = Taxonomy(domain="编程开发", direction="Rust 并发模型 教程详解", confidence=0.7)
        out = normalize(t)
        assert out is not None
        assert out.direction == "Rust 并发模型"

    def test_tags_dedup_case_insensitive_and_truncate_to_8(self):
        raw = [
            "Transformer",
            "transformer",  # dup of #1, should drop
            "KV-cache",
            "kv-cache",  # dup of #3, should drop
            "Attention",
            "Position",
            "MASK",
            "Multi-head",
            "Norm",
            "RoPE",
            "ALiBi",
            "FlashAttn",
        ]
        t = Taxonomy(domain="AI 技术", direction="注意力机制", tags=raw, confidence=0.8)
        out = normalize(t)
        assert out is not None
        assert len(out.tags) == 5
        # Original casing of the *first* occurrence is preserved.
        assert out.tags[0] == "Transformer"
        assert out.tags[1] == "KV-cache"
        # No duplicates, case-insensitive.
        lowered = [t.lower() for t in out.tags]
        assert len(set(lowered)) == 5

    def test_tags_drop_empty_and_non_string(self):
        t = Taxonomy(
            domain="AI 技术",
            direction="x",
            tags=["", "  ", "Real"],
            confidence=0.6,
        )
        out = normalize(t)
        assert out is not None
        assert out.tags == ["Real"]

    def test_confidence_clamped_to_unit_interval(self):
        # Pydantic's own ge/le on the model already raises for out-of-range
        # inputs, so feed in an in-range but very small float to confirm
        # normalize() does not zero it.
        t = Taxonomy(domain="AI 技术", direction="x", confidence=0.05)
        out = normalize(t)
        assert out is not None
        assert out.confidence == 0.05


class TestTaxonomyNeedsReview:
    def test_none_needs_review(self):
        assert needs_review(None) is True

    def test_low_confidence_needs_review(self):
        t = Taxonomy(domain="AI 技术", direction="x", confidence=0.3)
        assert needs_review(t) is True

    def test_other_domain_needs_review(self):
        t = Taxonomy(domain="其他", direction="x", confidence=0.9)
        assert needs_review(t) is True

    def test_empty_direction_needs_review(self):
        t = Taxonomy(domain="AI 技术", direction="", confidence=0.9)
        assert needs_review(t) is True

    def test_high_confidence_white_listed_passes(self):
        t = Taxonomy(domain="AI 技术", direction="注意力机制", confidence=0.85)
        assert needs_review(t) is False


class TestTaxonomyWhitelist:
    def test_other_is_in_whitelist(self):
        # The "其他" sentinel must be reachable so unknown domains have
        # somewhere to land without violating the white-list contract.
        assert "其他" in DOMAIN_WHITELIST

    def test_whitelist_is_unique(self):
        assert len(DOMAIN_WHITELIST) == len(set(DOMAIN_WHITELIST))


# ---------------------------------------------------------------------------
# Stage 2 — RAG layer
# ---------------------------------------------------------------------------


def _make_lecture(bv: str = "BV_TEST") -> LectureJSON:
    """Synthetic LectureJSON exercising every chunk kind."""
    return LectureJSON(
        bv_id=bv,
        title="Test Lecture",
        author="UP",
        duration=600,
        cover_url="",
        one_liner="一句话",
        category="lecture",
        domain_tags=["Test"],
        chapters=[
            Chapter(
                index=1,
                title="第一章",
                start=0,
                end=200,
                summary="第一章概览段落",
                learning_goal="理解基础概念",
                teaching_notes=[
                    "第一段教学讲解，介绍概念 A 的来龙去脉。",
                    "第二段教学讲解，扩展到与 B 的关系。",
                ],
                points=[
                    Point(
                        text="A 的核心定义是 X",
                        ts=30,
                        quote="原字幕：A 就是 X，因为 ...",
                    ),
                ],
                pitfalls=["容易把 A 误解为 B 的子集"],
                key_takeaways=["要点 1"],
                frames=[
                    Frame(
                        ts=45,
                        path="/img/ch1_diagram.jpg",
                        caption="架构图",
                        ocr_text="模块 A → 模块 B",
                        insight="说明 A 与 B 的依赖关系",
                        visual_type="diagram",
                    ),
                ],
            ),
            Chapter(
                index=2,
                title="第二章",
                start=200,
                end=400,
                summary="第二章概览",
                learning_goal="掌握进阶用法",
                teaching_notes=["第三段教学讲解，开始进入进阶。"],
                points=[],
                pitfalls=[],
                frames=[],
            ),
        ],
        knowledge_units=[
            KnowledgeUnitView(
                id="ku1",
                type="concept",
                title="概念 A",
                explanation="A 的完整解释，覆盖定义、用法和边界。",
                ts=40,
                quote="字幕证据",
                chapter_index=1,
            ),
        ],
        visual_evidence=[],
    )


class TestChunkLecture:
    def test_emits_all_five_kinds(self):
        chunks = chunk_lecture(_make_lecture())
        kinds = {c.kind for c in chunks}
        assert kinds == {
            "teaching_note",
            "quote",
            "pitfall",
            "knowledge_unit",
            "frame_ocr",
        }

    def test_chunk_counts(self):
        chunks = chunk_lecture(_make_lecture())
        kinds = [c.kind for c in chunks]
        # 2 teaching_notes in ch1 + 1 in ch2 = 3
        assert kinds.count("teaching_note") == 3
        assert kinds.count("quote") == 1
        assert kinds.count("pitfall") == 1
        assert kinds.count("knowledge_unit") == 1
        assert kinds.count("frame_ocr") == 1

    def test_chunk_carries_bv_and_anchors(self):
        chunks = chunk_lecture(_make_lecture("BV_X"))
        assert all(c.bv_id == "BV_X" for c in chunks)
        # quote chunk anchors must point at the exact subtitle ts
        quote = next(c for c in chunks if c.kind == "quote")
        assert quote.t_start == quote.t_end == 30
        assert quote.chapter_idx == 1
        # teaching_note spans the chapter range
        tn = next(c for c in chunks if c.kind == "teaching_note" and c.chapter_idx == 1)
        assert tn.t_start == 0 and tn.t_end == 200

    def test_skips_empty_text(self):
        lecture = _make_lecture()
        # Inject an empty teaching_note + an empty pitfall; expect both dropped.
        lecture.chapters[0].teaching_notes.append("")
        lecture.chapters[0].teaching_notes.append("   ")
        lecture.chapters[0].pitfalls.append("")
        chunks = chunk_lecture(lecture)
        # still 3 teaching_notes (no extras), still 1 pitfall.
        assert sum(1 for c in chunks if c.kind == "teaching_note") == 3
        assert sum(1 for c in chunks if c.kind == "pitfall") == 1

    def test_frame_text_dedupes_caption_insight_ocr(self):
        chunks = chunk_lecture(_make_lecture())
        frame_chunk = next(c for c in chunks if c.kind == "frame_ocr")
        # caption / insight / ocr are all distinct → all three present once
        text = frame_chunk.text
        assert "架构图" in text
        assert "说明 A 与 B" in text
        assert "模块 A" in text
        # path preserved in meta for downstream tools
        assert frame_chunk.meta["path"] == "/img/ch1_diagram.jpg"
    def test_prefers_evidence_index_rag_chunks_when_present(self):
        lecture = _attach_note_and_evidence(_make_lecture("BV_EVIDENCE"))
        chunks = chunk_lecture(lecture)

        assert [chunk.kind for chunk in chunks] == ["quote", "frame_ocr"]
        quote_chunk = chunks[0]
        assert quote_chunk.text == "证据块：先比较再归一化"
        assert quote_chunk.meta["evidence_id"] == "ev-quote-1"
        assert quote_chunk.meta["note_node_id"] == "block-1"
        assert quote_chunk.meta["evidence_kind"] == "quote_evidence"
        assert quote_chunk.meta["quote"] == "原字幕：A 就是 X，因为先比较再归一化。"

    def test_falls_back_to_legacy_chunks_when_evidence_index_has_no_rag_chunks(self):
        lecture = _make_lecture("BV_EVIDENCE_EMPTY")
        lecture.evidence_index = EvidenceIndex.model_validate(
            {
                "evidence_objects": [
                    {
                        "evidence_id": "ev-empty",
                        "kind": "compound_evidence",
                    }
                ]
            }
        )

        chunks = chunk_lecture(lecture)

        assert any(chunk.kind == "teaching_note" for chunk in chunks)
        assert any(chunk.kind == "quote" for chunk in chunks)


class _StubEmbedder:
    """Deterministic embedder for RAG tests; no DashScope traffic."""

    model = "stub-embed"

    def __init__(self, mapping: dict[str, list[float]], dim: int = 4) -> None:
        self._mapping = mapping
        self.dim = dim

    async def embed_text(self, texts):  # type: ignore[no-untyped-def]
        return [list(self._mapping.get(t, [0.0] * self.dim)) for t in texts]

    async def embed_image(self, paths):  # type: ignore[no-untyped-def]
        raise NotImplementedError("image embedding not used in tests")


class TestRAGSearch:
    async def test_hybrid_search_returns_top_k(self, tmp_path):
        db_path = tmp_path / "app.db"
        db = Database(db_path)
        await db.init()

        mapping = {
            "Apple is a fruit": [1.0, 0.0, 0.0, 0.0],
            "Banana is also a fruit": [0.95, 0.05, 0.0, 0.0],
            "Car is a vehicle": [0.0, 1.0, 0.0, 0.0],
            "Bicycle is also a vehicle": [0.0, 0.95, 0.05, 0.0],
            "fruit": [1.0, 0.0, 0.0, 0.0],
        }
        embedder = _StubEmbedder(mapping, dim=4)
        rag = RAGStore(db_path, embedder=embedder)
        await rag.init()

        if not rag.vec_available:
            pytest.skip("sqlite-vec extension unavailable in this environment")

        chunks = [
            Chunk(bv_id="BV_T", kind="teaching_note", text=t, chapter_idx=1)
            for t in mapping
            if t != "fruit"
        ]
        n = await rag.upsert_chunks("BV_T", chunks)
        assert n == 4
        assert rag.dim == 4

        results = await rag.search("fruit", bv_id="BV_T", top_k=2)
        assert 1 <= len(results) <= 2
        # Top hits must be the fruit-related chunks, not the vehicle ones.
        top_texts = " | ".join(r["text"] for r in results)
        assert "Apple" in top_texts or "Banana" in top_texts
        assert "Car" not in top_texts
        assert "Bicycle" not in top_texts

        # bv filter actually filters
        cross = await rag.search("fruit", bv_id="BV_DOES_NOT_EXIST", top_k=2)
        assert cross == []

        await rag.close()

    async def test_upsert_replaces_existing_bv(self, tmp_path):
        db_path = tmp_path / "app.db"
        db = Database(db_path)
        await db.init()

        embedder = _StubEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.0, 1.0, 0.0, 0.0],
                "gamma": [0.0, 0.0, 1.0, 0.0],
            },
            dim=4,
        )
        rag = RAGStore(db_path, embedder=embedder)
        await rag.init()
        if not rag.vec_available:
            pytest.skip("sqlite-vec extension unavailable in this environment")

        await rag.upsert_chunks("BV_R", [
            Chunk(bv_id="BV_R", kind="teaching_note", text="alpha"),
            Chunk(bv_id="BV_R", kind="teaching_note", text="beta"),
        ])
        # Re-upsert with a single different chunk; old chunks must be gone.
        await rag.upsert_chunks("BV_R", [
            Chunk(bv_id="BV_R", kind="teaching_note", text="gamma"),
        ])

        with sqlite3.connect(db_path) as raw:
            count = raw.execute(
                "SELECT count(*) FROM lecture_chunks WHERE bv_id=?", ("BV_R",)
            ).fetchone()[0]
        assert count == 1
        await rag.close()


class TestRAGFTSFallback:
    async def test_fts_only_when_vec_extension_unavailable(self, tmp_path, monkeypatch):
        # Simulate the realistic failure mode: the wheel exists but the
        # native ``vec0`` shared lib refuses to load on this host.  This
        # exercises the same fallback branch as a missing wheel.  Patching
        # sqlite3.Connection.enable_load_extension itself is not possible
        # because it is an immutable C type.
        import sqlite_vec

        def _fake_load(conn):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated: sqlite-vec native lib not loadable")

        monkeypatch.setattr(sqlite_vec, "load", _fake_load)

        db_path = tmp_path / "app.db"
        db = Database(db_path)
        await db.init()

        embedder = _StubEmbedder({}, dim=4)
        rag = RAGStore(db_path, embedder=embedder)
        await rag.init()
        assert rag.vec_available is False

        chunks = [
            Chunk(bv_id="BV_T", kind="teaching_note", text="The quick brown fox jumps over the lazy dog"),
            Chunk(bv_id="BV_T", kind="teaching_note", text="A second teaching paragraph about cats"),
        ]
        n = await rag.upsert_chunks("BV_T", chunks)
        assert n == 2

        # FTS5-only retrieval still works
        results = await rag.search("fox", bv_id="BV_T", top_k=5)
        assert len(results) >= 1
        assert "fox" in results[0]["text"]

        # Vector branch was skipped: kv.embedding_dim should not be set.
        with sqlite3.connect(db_path) as raw:
            row = raw.execute(
                "SELECT value FROM kv WHERE key='embedding_dim'"
            ).fetchone()
        assert row is None

        await rag.close()


# ---------------------------------------------------------------------------
# Stage 3 — Tool Registry
# ---------------------------------------------------------------------------


from app.copilot.tools import (  # noqa: E402
    COPILOT_TOOLS,
    MCP_TOOLS,
    ToolContext,
    ToolError,
    explain_frame,
    get_evidence_object,
    get_chapter,
    get_frame,
    get_frame_description,
    get_knowledge_units,
    get_note_unit,
    get_quote_context,
    list_lectures,
    search_evidence,
    search_lecture,
    search_lectures,
    summarize_video,
    web_search,
    _extract_web_results,
)
from app.copilot.sections import (  # noqa: E402
    SECTION_TYPES,
    extract_section_types,
    normalise_section_type,
    transform,
)


class TestSections:
    def test_transform_markdown(self):
        raw = "[[evidence]]\n讲义说 A [Ch1]\n\n[[background]]\n补充 B"
        out = transform(raw, "markdown")
        assert "### 讲义证据" in out
        assert "### 背景补全" in out
        assert "[[evidence]]" not in out

    def test_transform_html(self):
        raw = "[[deep_dive]]\nA < B"
        out = transform(raw, "html")
        assert 'class="cp-section cp-section-deep_dive"' in out
        assert "原理深挖" in out
        assert "A &lt; B" in out

    def test_section_types_include_v2_contract(self):
        assert set(SECTION_TYPES) == {
            "evidence",
            "extension",
            "background",
            "deep_dive",
            "application",
            "boundary",
            "offtopic",
        }

    def test_extract_section_types_normalises(self):
        raw = "[[ Evidence ]]\nA\n[[DEEP_DIVE]]\nB"
        assert extract_section_types(raw) == ["evidence", "deep_dive"]

    def test_normalise_section_type(self):
        assert normalise_section_type("  BACKGROUND ") == "background"

    def test_transform_text(self):
        raw = "[[application]]\n用于排查工具调用预算。"
        assert transform(raw, "text") == "【应用举例】\n用于排查工具调用预算。"

    def test_transform_no_known_section_returns_raw(self):
        raw = "普通答案，没有段头。"
        assert transform(raw, "markdown") == raw

    def test_transform_ignores_unknown_section_type(self):
        raw = "[[unknown]]\nX\n\n[[boundary]]\nY"
        out = transform(raw, "text")
        assert "【边界说明】" in out
        assert "unknown" not in out

    def test_transform_rejects_unknown_target(self):
        with pytest.raises(ValueError):
            transform("[[evidence]]\nA", "xml")

    def test_transform_html_preserves_line_breaks(self):
        out = transform("[[background]]\nA\nB", "html")
        assert "A<br>B" in out

    def test_transform_markdown_strips_empty_body(self):
        assert transform("[[offtopic]]\n", "markdown") == "### 问题引导"


async def _build_ctx(
    tmp_path,
    lectures,
    *,
    embedder_mapping=None,
    rows_meta=None,
):
    """Spin up real Database + RAGStore + insert summaries, return a ToolContext.

    ``lectures`` is a list of (LectureJSON, summary_overrides) where
    ``summary_overrides`` is a dict of extra columns (``domain``,
    ``direction``, …) for ``upsert_summary``.  ``rows_meta`` mirrors
    that for explicit per-row taxonomy.  ``embedder_mapping`` seeds the
    StubEmbedder so that ``search_lecture`` can be exercised
    deterministically.
    """
    from app.storage.db import Database as _Database  # noqa: WPS433

    db_path = tmp_path / "app.db"
    db = _Database(db_path)
    await db.init()

    embedder = _StubEmbedder(embedder_mapping or {}, dim=4)
    rag = RAGStore(db_path, embedder=embedder)
    await rag.init()

    rows_meta = rows_meta or {}
    for lec in lectures:
        meta = rows_meta.get(lec.bv_id, {})
        await db.upsert_summary(
            bv_id=lec.bv_id,
            url=f"https://www.bilibili.com/video/{lec.bv_id}",
            title=lec.title,
            author=lec.author,
            duration=int(lec.duration),
            cover_url=lec.cover_url,
            category=lec.category,
            domain_tags=lec.domain_tags,
            summary_json=lec.model_dump(mode="json"),
            report_path=f"reports/{lec.bv_id}.html",
            model_used="test",
            domain=meta.get("domain"),
            direction=meta.get("direction"),
        )

    return ToolContext(db=db, rag=rag, pipeline=None), rag


def _lecture_with_taxonomy(bv: str, *, domain="AI 技术", direction="注意力机制") -> LectureJSON:
    lec = _make_lecture(bv)
    lec.taxonomy = Taxonomy(
        domain=domain,
        direction=direction,
        tags=["Transformer", "Attention"],
        confidence=0.85,
    )
    # Also propagate frame metadata so explain_frame has something to chew on.
    lec.visual_evidence = list(lec.chapters[0].frames)
    return lec


def _attach_note_and_evidence(lecture: LectureJSON) -> LectureJSON:
    lecture.lecture_note_ir = LectureNoteIR.model_validate(
        {
            "front_matter": {
                "one_sentence_claim": "这份讲义先交代主问题，再解释证据如何逐步支撑结论。",
                "takeaways_top": [
                    "先看主线，再定位证据",
                    "point A 是第一段解释的核心",
                ],
            },
            "body": {
                "teaching_units": [
                    {
                        "unit_id": "unit-1",
                        "title": "主线单元一",
                        "core_message": "先理解概念 A 的作用，再连接到概念 B。",
                        "content_blocks": [
                            {
                                "block_id": "block-1",
                                "title": "概念 A",
                                "paragraphs": ["概念 A 决定了后续比较方式。"],
                            }
                        ],
                    },
                    {
                        "unit_id": "unit-2",
                        "title": "主线单元二",
                        "core_message": "再用关键证据说明概念 B 如何收束前面的论证。",
                        "content_blocks": [
                            {
                                "block_id": "block-2",
                                "title": "概念 B",
                                "paragraphs": ["概念 B 负责把前面的局部判断组合起来。"],
                            }
                        ],
                    },
                ]
            },
        }
    )
    lecture.evidence_index = EvidenceIndex.model_validate(
        {
            "evidence_objects": [
                {
                    "evidence_id": "ev-quote-1",
                    "kind": "quote_evidence",
                    "summary": "原话先比较再归一化。",
                    "quote": "原字幕：A 就是 X，因为先比较再归一化。",
                    "chapter_index": 1,
                    "note_node_ids": ["block-1"],
                    "rag_chunks": [
                        {
                            "kind": "quote",
                            "chapter_idx": 1,
                            "t_start": 30,
                            "t_end": 30,
                            "text": "证据块：先比较再归一化",
                            "note_node_id": "block-1",
                            "meta": {"source": "subtitle"},
                        }
                    ],
                },
                {
                    "evidence_id": "ev-frame-1",
                    "kind": "frame_evidence",
                    "summary": "关键图示把 A 和 B 的关系放在一张图里。",
                    "chapter_index": 1,
                    "path": "/img/ch1_diagram.jpg",
                    "note_node_ids": ["unit-1"],
                    "rag_chunks": [
                        {
                            "kind": "frame_ocr",
                            "chapter_idx": 1,
                            "t_start": 45,
                            "t_end": 45,
                            "text": "证据图：A -> B",
                            "note_node_id": "unit-1",
                        }
                    ],
                },
            ],
            "evidence_relations": [
                {"relation": "supports", "from_id": "ev-quote-1", "to_id": "block-1"},
                {"relation": "supports", "from_id": "ev-frame-1", "to_id": "unit-1"},
            ],
        }
    )
    return lecture


class TestToolGetChapter:
    async def test_valid(self, tmp_path):
        lec = _make_lecture("BV_GC")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_chapter(ctx, bv="BV_GC", chapter_idx=1)
            assert out.chapter_idx == 1
            assert out.title == "第一章"
            assert len(out.teaching_notes) == 2
            assert len(out.points) == 1
            assert out.points[0].quote.startswith("原字幕")
            assert len(out.frames) == 1
            assert out.frames[0].path == "/img/ch1_diagram.jpg"
            assert out.frames[0].frame_id >= 1
        finally:
            await rag.close()

    async def test_out_of_range(self, tmp_path):
        lec = _make_lecture("BV_GC2")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            with pytest.raises(ToolError) as ei:
                await get_chapter(ctx, bv="BV_GC2", chapter_idx=99)
            assert ei.value.code == "not_found"
        finally:
            await rag.close()

    async def test_unknown_bv_returns_not_found(self, tmp_path):
        lec = _make_lecture("BV_KNOWN")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            with pytest.raises(ToolError) as ei:
                await get_chapter(ctx, bv="BV_NOPE", chapter_idx=1)
            assert ei.value.code == "not_found"
        finally:
            await rag.close()

    async def test_exposes_note_and_evidence_entrypoints_when_present(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_GC_NOTE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_chapter(ctx, bv="BV_GC_NOTE", chapter_idx=1)
            assert "ev-quote-1" in out.related_evidence_ids
            assert out.related_note_node_ids == []
            assert out.compatibility_anchor_kind == "chapter"
            assert out.compatibility_anchor_id == "Ch1"
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()

    async def test_chapter_primary_requires_explicit_chapter_bridge(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_GC_CHAPTER"))
        lec.evidence_index.evidence_objects.append(
            EvidenceObject.model_validate(
                {
                    "evidence_id": "chapter-1",
                    "kind": "chapter_evidence",
                    "title": "第一章主桥",
                    "chapter_index": 1,
                    "note_node_ids": ["unit-1"],
                }
            )
        )
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_chapter(ctx, bv="BV_GC_CHAPTER", chapter_idx=1)
            assert "ev-quote-1" in out.related_evidence_ids
            assert "chapter-1" in out.related_evidence_ids
            assert out.compatibility_anchor_kind == "chapter"
            assert out.compatibility_anchor_id == "Ch1"
            assert out.primary_ref_kind == "evidence"
            assert out.primary_ref_id == "chapter-1"
            assert out.primary_evidence_id == "chapter-1"
            assert out.primary_note_node_id == "unit-1"
            assert out.primary_note_unit_id == "unit-1"
            assert out.primary_note_node_type == "teaching_unit"
            assert out.primary_note_title == "主线单元一"
            assert out.primary_evidence_kind == "chapter_evidence"
        finally:
            await rag.close()


class TestToolGetFrame:
    async def test_valid(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_GF")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_frame(ctx, bv="BV_GF", frame_id=1)
            assert out.path == "/img/ch1_diagram.jpg"
            assert out.ts == 45
            assert out.chapter_idx == 1
            # alias must agree
            alias = await get_frame_description(ctx, bv="BV_GF", frame_id=1)
            assert alias.path == out.path
        finally:
            await rag.close()

    async def test_missing(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_GF2")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            with pytest.raises(ToolError) as ei:
                await get_frame(ctx, bv="BV_GF2", frame_id=999)
            assert ei.value.code == "not_found"
        finally:
            await rag.close()

    async def test_returns_related_evidence_backlinks_when_present(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_GF_NOTE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_frame(ctx, bv="BV_GF_NOTE", frame_id=1)
            assert "ev-frame-1" in out.related_evidence_ids
            assert "unit-1" in out.related_note_node_ids
            assert out.compatibility_anchor_kind == "frame"
            assert out.compatibility_anchor_id == "F1"
            assert out.primary_ref_kind == "evidence"
            assert out.primary_ref_id == "ev-frame-1"
            assert out.primary_evidence_id == "ev-frame-1"
            assert out.primary_note_node_id == "unit-1"
            assert out.primary_note_unit_id == "unit-1"
            assert out.primary_note_node_type == "teaching_unit"
            assert out.primary_note_title == "主线单元一"
            assert out.primary_evidence_kind == "frame_evidence"
        finally:
            await rag.close()

    async def test_frame_time_only_match_keeps_canonical_fields_empty(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_GF_TIMEONLY"))
        lec.evidence_index.evidence_objects[1].path = ""
        lec.evidence_index.evidence_objects[1].ts = 45
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_frame(ctx, bv="BV_GF_TIMEONLY", frame_id=1)
            assert "ev-frame-1" in out.related_evidence_ids
            assert "unit-1" in out.related_note_node_ids
            assert out.compatibility_anchor_kind == "frame"
            assert out.compatibility_anchor_id == "F1"
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()


class TestToolExplainFrame:
    async def test_returns_context_quotes(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_EF")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await explain_frame(ctx, bv="BV_EF", frame_id=1)
            # Frame ts=45, chapter has point at ts=30 → within ±60s.
            assert out.why_useful, "why_useful must be non-empty when frame.insight is set"
            assert any(q.ts == 30 for q in out.context_quotes)
        finally:
            await rag.close()

    async def test_radius_zero_filters_neighbours(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_EF2")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await explain_frame(ctx, bv="BV_EF2", frame_id=1, context_radius_seconds=0)
            assert out.context_quotes == []
        finally:
            await rag.close()

    async def test_returns_related_backlinks_when_present(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_EF_NOTE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await explain_frame(ctx, bv="BV_EF_NOTE", frame_id=1)
            assert "ev-frame-1" in out.related_evidence_ids
            assert "unit-1" in out.related_note_node_ids
            assert out.compatibility_anchor_kind == "frame"
            assert out.compatibility_anchor_id == "F1"
            assert out.primary_ref_kind == "evidence"
            assert out.primary_ref_id == "ev-frame-1"
            assert out.primary_evidence_id == "ev-frame-1"
            assert out.primary_note_node_id == "unit-1"
            assert out.primary_note_unit_id == "unit-1"
            assert out.primary_note_node_type == "teaching_unit"
            assert out.primary_note_title == "主线单元一"
            assert out.primary_evidence_kind == "frame_evidence"
        finally:
            await rag.close()

    async def test_explain_frame_time_only_match_keeps_canonical_fields_empty(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_EF_TIMEONLY"))
        lec.evidence_index.evidence_objects[1].path = ""
        lec.evidence_index.evidence_objects[1].ts = 45
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await explain_frame(ctx, bv="BV_EF_TIMEONLY", frame_id=1)
            assert "ev-frame-1" in out.related_evidence_ids
            assert "unit-1" in out.related_note_node_ids
            assert out.compatibility_anchor_kind == "frame"
            assert out.compatibility_anchor_id == "F1"
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()


class TestToolGetQuoteContext:
    async def test_substring_match_returns_anchor(self, tmp_path):
        lec = _make_lecture("BV_QC")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_quote_context(ctx, bv="BV_QC", quote="A 就是 X")
            assert out.chapter_idx == 1
            assert out.t_start == 30
            assert "A 就是 X" in out.matched_quote
        finally:
            await rag.close()

    async def test_no_match_raises(self, tmp_path):
        lec = _make_lecture("BV_QC2")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            with pytest.raises(ToolError) as ei:
                await get_quote_context(ctx, bv="BV_QC2", quote="不存在的句子")
            assert ei.value.code == "not_found"
        finally:
            await rag.close()


    async def test_returns_related_backlinks_when_present(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_QC_NOTE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_quote_context(ctx, bv="BV_QC_NOTE", quote="A 就是 X")
            assert "ev-quote-1" in out.related_evidence_ids
            assert "block-1" in out.related_note_node_ids
            assert out.compatibility_anchor_kind == "quote"
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()

    async def test_quote_context_does_not_backfill_chapter_wide_quote_evidence(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_QC_STRICT"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_quote_context(ctx, bv="BV_QC_STRICT", quote="因为 ...")
            assert out.related_evidence_ids == []
            assert out.related_note_node_ids == []
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()

    async def test_quote_context_primary_requires_explicit_quote_bridge(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_QC_EXPLICIT"))
        unit = lec.lecture_note_ir.body.teaching_units[0]
        block = unit.content_blocks[0]
        unit.source_chapter_refs = [1]
        block.source_chapter_refs = [1]
        block.evidence_refs = ["ev-quote-1"]
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_quote_context(ctx, bv="BV_QC_EXPLICIT", quote="A 就是 X")
            assert "ev-quote-1" in out.related_evidence_ids
            assert "block-1" in out.related_note_node_ids
            assert out.compatibility_anchor_kind == "quote"
            assert out.compatibility_anchor_id == "A 就是 X"
            assert out.primary_ref_kind == "evidence"
            assert out.primary_ref_id == "ev-quote-1"
            assert out.primary_evidence_id == "ev-quote-1"
            assert out.primary_note_node_id == "block-1"
            assert out.primary_note_unit_id == "unit-1"
            assert out.primary_note_node_type == "content_block"
            assert out.primary_note_title == "概念 A"
            assert out.primary_evidence_kind == "quote_evidence"
        finally:
            await rag.close()

    async def test_get_chapter_without_explicit_bridge_keeps_canonical_fields_empty(self, tmp_path):
        lec = _make_lecture("BV_GC_STRICT")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_chapter(ctx, bv="BV_GC_STRICT", chapter_idx=1)
            assert out.compatibility_anchor_kind == "chapter"
            assert out.compatibility_anchor_id == "Ch1"
            assert out.related_evidence_ids == []
            assert out.related_note_node_ids == []
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()

    async def test_get_frame_without_explicit_bridge_keeps_canonical_fields_empty(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_GF_STRICT")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_frame(ctx, bv="BV_GF_STRICT", frame_id=1)
            assert out.compatibility_anchor_kind == "frame"
            assert out.compatibility_anchor_id == "F1"
            assert out.related_evidence_ids == []
            assert out.related_note_node_ids == []
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()

    async def test_explain_frame_without_explicit_bridge_keeps_canonical_fields_empty(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_EF_STRICT")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await explain_frame(ctx, bv="BV_EF_STRICT", frame_id=1)
            assert out.compatibility_anchor_kind == "frame"
            assert out.compatibility_anchor_id == "F1"
            assert out.related_evidence_ids == []
            assert out.related_note_node_ids == []
            assert out.primary_ref_kind == ""
            assert out.primary_ref_id == ""
            assert out.primary_evidence_id == ""
            assert out.primary_note_node_id == ""
            assert out.primary_note_unit_id == ""
            assert out.primary_note_node_type == ""
            assert out.primary_note_title == ""
            assert out.primary_evidence_kind == ""
        finally:
            await rag.close()


class TestToolGetNoteUnit:
    async def test_returns_teaching_unit(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_GNU"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_note_unit(ctx, bv="BV_GNU", unit_id="unit-1")
            assert out.unit.unit_id == "unit-1"
            assert out.focused_block is None
            assert "ev-frame-1" in out.related_evidence_ids
            assert out.primary_ref_kind == "note"
            assert out.primary_ref_id == "unit-1"
            assert out.primary_evidence_id == "ev-frame-1"
        finally:
            await rag.close()

    async def test_accepts_block_id_and_returns_focused_block(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_GNU_BLOCK"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_note_unit(ctx, bv="BV_GNU_BLOCK", unit_id="block-1")
            assert out.unit.unit_id == "unit-1"
            assert out.focused_block is not None
            assert out.focused_block.block_id == "block-1"
            assert out.resolved_node_type == "content_block"
            assert "ev-quote-1" in out.related_evidence_ids
        finally:
            await rag.close()


class TestToolGetEvidenceObject:
    async def test_returns_evidence_and_relations(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_GEO"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_evidence_object(ctx, bv="BV_GEO", evidence_id="ev-quote-1")
            assert out.evidence.evidence_id == "ev-quote-1"
            assert any(rel.relation == "supports" for rel in out.relations)
            assert "unit-1" in out.related_note_unit_ids
            assert out.primary_ref_kind == "evidence"
            assert out.primary_ref_id == "ev-quote-1"
            assert out.primary_note_node_id == "unit-1"
        finally:
            await rag.close()


class TestToolSearchEvidence:
    async def test_searches_evidence_index(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_SE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await search_evidence(ctx, bv="BV_SE", query="归一化", top_k=3)
            assert out.hits
            assert out.hits[0].evidence_id == "ev-quote-1"
            assert "block-1" in out.hits[0].note_node_ids
        finally:
            await rag.close()


class TestToolSearchLecture:
    async def test_fts_only_falls_back_to_evidence_backed_chunks(self, tmp_path, monkeypatch):
        try:
            import sqlite_vec
        except ModuleNotFoundError:
            sqlite_vec = None

        if sqlite_vec is not None:
            def _fake_load(conn):  # type: ignore[no-untyped-def]
                raise RuntimeError("simulated: sqlite-vec native lib not loadable")

            monkeypatch.setattr(sqlite_vec, "load", _fake_load)

        lec = _attach_note_and_evidence(_make_lecture("BV_S1_EVIDENCE_FTS"))
        lec.evidence_index.evidence_objects[0].title = "Harness 可控性"
        lec.evidence_index.evidence_objects[0].summary = "隔离样本强调 Harness 可控性与可复现验证。"
        lec.evidence_index.evidence_objects[0].rag_chunks[0].text = "deterministic fixture setup"
        ctx, rag = await _build_ctx(tmp_path, [lec])
        assert rag.vec_available is False
        try:
            from app.copilot.indexer import index_lecture as _index_lecture

            n = await _index_lecture(rag, "BV_S1_EVIDENCE_FTS", lec)
            assert n == 2
            assert await rag.search("Harness 可控性", bv_id="BV_S1_EVIDENCE_FTS", top_k=3) == []

            out = await search_lecture(
                ctx,
                bv="BV_S1_EVIDENCE_FTS",
                query="Harness 可控性",
                top_k=3,
            )

            assert out.chunks
            hit = out.chunks[0]
            assert hit.text == "deterministic fixture setup"
            assert hit.evidence_id == "ev-quote-1"
            assert hit.note_node_id == "block-1"
            assert hit.evidence_kind == "quote_evidence"
            assert hit.meta["evidence_id"] == "ev-quote-1"
        finally:
            await rag.close()

    async def test_top_k_returns_chunks(self, tmp_path):
        lec = _make_lecture("BV_S1")
        mapping = {
            "concept query": [1.0, 0.0, 0.0, 0.0],
            "第一段教学讲解，介绍概念 A 的来龙去脉。": [0.95, 0.05, 0.0, 0.0],
            "第二段教学讲解，扩展到与 B 的关系。": [0.9, 0.1, 0.0, 0.0],
        }
        ctx, rag = await _build_ctx(tmp_path, [lec], embedder_mapping=mapping)
        if not rag.vec_available:
            await rag.close()
            pytest.skip("sqlite-vec extension unavailable")
        try:
            # First populate the index for this BV.
            from app.copilot.indexer import index_lecture as _index_lecture

            n = await _index_lecture(rag, "BV_S1", lec)
            assert n > 0
            out = await search_lecture(ctx, bv="BV_S1", query="concept query", top_k=3)
            assert 1 <= len(out.chunks) <= 3
            assert all(c.bv_id == "BV_S1" for c in out.chunks)
            # top_k cap is honoured.
            out_more = await search_lecture(ctx, bv="BV_S1", query="concept query", top_k=10)
            assert len(out_more.chunks) <= 10
        finally:
            await rag.close()

    async def test_evidence_index_hits_preserve_backlinks(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_S1_EVIDENCE"))
        mapping = {
            "主线证据 query": [1.0, 0.0, 0.0, 0.0],
            "证据块：先比较再归一化": [0.95, 0.05, 0.0, 0.0],
            "证据图：A -> B": [0.9, 0.1, 0.0, 0.0],
        }
        ctx, rag = await _build_ctx(tmp_path, [lec], embedder_mapping=mapping)
        if not rag.vec_available:
            await rag.close()
            pytest.skip("sqlite-vec extension unavailable")
        try:
            from app.copilot.indexer import index_lecture as _index_lecture

            n = await _index_lecture(rag, "BV_S1_EVIDENCE", lec)
            assert n == 2
            out = await search_lecture(ctx, bv="BV_S1_EVIDENCE", query="主线证据 query", top_k=2)
            assert len(out.chunks) == 2
            hit = out.chunks[0]
            assert hit.meta["evidence_id"] == "ev-quote-1"
            assert hit.meta["note_node_id"] == "block-1"
            assert hit.meta["evidence_kind"] == "quote_evidence"
            assert hit.evidence_id == "ev-quote-1"
            assert hit.note_node_id == "block-1"
            assert hit.evidence_kind == "quote_evidence"
            assert hit.quote == "原字幕：A 就是 X，因为先比较再归一化。"
        finally:
            await rag.close()

    async def test_search_lecture_backlink_matches_evidence_object_rag_chunk_note_node(self, tmp_path):
        lec = _attach_note_and_evidence(_make_lecture("BV_S1_CHAIN"))
        mapping = {
            "主线证据 chain query": [1.0, 0.0, 0.0, 0.0],
            "证据块：先比较再归一化": [0.95, 0.05, 0.0, 0.0],
            "证据图：A -> B": [0.9, 0.1, 0.0, 0.0],
        }
        ctx, rag = await _build_ctx(tmp_path, [lec], embedder_mapping=mapping)
        if not rag.vec_available:
            await rag.close()
            pytest.skip("sqlite-vec extension unavailable")
        try:
            from app.copilot.indexer import index_lecture as _index_lecture

            await _index_lecture(rag, "BV_S1_CHAIN", lec)
            out = await search_lecture(ctx, bv="BV_S1_CHAIN", query="主线证据 chain query", top_k=2)
            hit = next(chunk for chunk in out.chunks if chunk.evidence_id == "ev-quote-1")
            evidence_out = await get_evidence_object(ctx, bv="BV_S1_CHAIN", evidence_id=hit.evidence_id)
            expected_note_node_id = evidence_out.evidence.rag_chunks[0].note_node_id

            assert expected_note_node_id == "block-1"
            assert hit.note_node_id == expected_note_node_id
        finally:
            await rag.close()


class TestToolSearchLectures:
    async def test_domain_filter(self, tmp_path):
        from app.copilot.indexer import index_lecture as _index_lecture

        lec_a = _lecture_with_taxonomy("BV_A", domain="AI 技术", direction="注意力机制")
        lec_b = _lecture_with_taxonomy("BV_B", domain="健身运动", direction="脂肪燃烧")
        mapping = {
            "concept query": [1.0, 0.0, 0.0, 0.0],
            "第一段教学讲解，介绍概念 A 的来龙去脉。": [0.95, 0.05, 0.0, 0.0],
            "第二段教学讲解，扩展到与 B 的关系。": [0.9, 0.1, 0.0, 0.0],
            "第三段教学讲解，开始进入进阶。": [0.7, 0.3, 0.0, 0.0],
        }
        ctx, rag = await _build_ctx(
            tmp_path,
            [lec_a, lec_b],
            embedder_mapping=mapping,
            rows_meta={
                "BV_A": {"domain": "AI 技术", "direction": "注意力机制"},
                "BV_B": {"domain": "健身运动", "direction": "脂肪燃烧"},
            },
        )
        if not rag.vec_available:
            await rag.close()
            pytest.skip("sqlite-vec extension unavailable")
        try:
            await _index_lecture(rag, "BV_A", lec_a)
            await _index_lecture(rag, "BV_B", lec_b)
            # Without filter we may see both BVs.
            mixed = await search_lectures(ctx, query="concept query", top_k=10)
            bvs_mixed = {c.bv_id for c in mixed.chunks}
            assert "BV_A" in bvs_mixed or "BV_B" in bvs_mixed

            # With domain filter only AI 技术 BVs survive.
            ai_only = await search_lectures(
                ctx, query="concept query", top_k=10, domain="AI 技术"
            )
            assert ai_only.chunks  # not empty
            assert all(c.bv_id == "BV_A" for c in ai_only.chunks)
        finally:
            await rag.close()


class TestToolWebSearch:
    async def test_missing_api_key_returns_error_dict(self, tmp_path):
        lec = _make_lecture("BV_WEB")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await web_search(ctx, query="DAC SAC entropy", count=2)
            assert out.query == "DAC SAC entropy"
            assert out.error is not None
            assert out.error["code"] in {"missing_api_key", "request_failed"}
        finally:
            await rag.close()


class TestToolGetKnowledgeUnits:
    async def test_returns_units_and_kind_filter(self, tmp_path):
        lec = _make_lecture("BV_KU")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            out = await get_knowledge_units(ctx, bv="BV_KU")
            assert len(out.units) == 1
            assert out.units[0].kind == "concept"

            # Filter by kind that doesn't exist.
            none = await get_knowledge_units(ctx, bv="BV_KU", kind="formula")
            assert none.units == []
        finally:
            await rag.close()


class TestToolListLectures:
    async def test_filter_by_domain(self, tmp_path):
        lec_a = _make_lecture("BV_LA")
        lec_b = _make_lecture("BV_LB")
        ctx, rag = await _build_ctx(
            tmp_path,
            [lec_a, lec_b],
            rows_meta={
                "BV_LA": {"domain": "AI 技术", "direction": "Agent 框架"},
                "BV_LB": {"domain": "烹饪", "direction": "中餐家常"},
            },
        )
        try:
            all_out = await list_lectures(ctx)
            assert {l.bv_id for l in all_out.lectures} == {"BV_LA", "BV_LB"}
            ai = await list_lectures(ctx, domain="AI 技术")
            assert {l.bv_id for l in ai.lectures} == {"BV_LA"}
            none = await list_lectures(ctx, domain="DoesNotExist")
            assert none.lectures == []
        finally:
            await rag.close()


class TestToolSummarizeVideo:
    async def test_unsupported_without_pipeline(self, tmp_path):
        lec = _make_lecture("BV_SV")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            with pytest.raises(ToolError) as ei:
                await summarize_video(ctx, url="https://www.bilibili.com/video/BV_SV")
            assert ei.value.code == "unsupported"
        finally:
            await rag.close()


class TestToolRegistryShape:
    def test_copilot_tools_count(self):
        assert set(COPILOT_TOOLS) == {
            "search_lecture",
            "search_evidence",
            "search_lectures",
            "get_note_unit",
            "get_evidence_object",
            "get_chapter",
            "get_frame",
            "get_quote_context",
            "explain_frame",
        }

    def test_mcp_tools_count(self):
        assert set(MCP_TOOLS) == {
            "summarize_video",
            "search_evidence",
            "search_lectures",
            "get_note_unit",
            "get_evidence_object",
            "get_chapter",
            "get_frame",
            "get_quote_context",
            "explain_frame",
            "get_knowledge_units",
            "list_lectures",
        }

    def test_tool_error_round_trip(self):
        e = ToolError("invalid_arg", "x must be > 0")
        assert e.code == "invalid_arg"
        assert "x must be > 0" in str(e)
        assert e.to_dict() == {"code": "invalid_arg", "message": "x must be > 0"}


# ---------------------------------------------------------------------------
# Stage 4 — Agent Runtime (LangGraph)
# ---------------------------------------------------------------------------


from langchain_core.messages import AIMessage  # noqa: E402

from app.copilot.agent import (  # noqa: E402
    FINAL_RESPONSE_PROMPT,
    build_graph,
    build_initial_state,
    make_copilot_tools,
    quick_ask,
    validate_anchors,
)
from app.copilot.prompts import (  # noqa: E402
    ANCHOR_PATTERNS,
    COPILOT_SYSTEM,
    format_references_block,
)


class TestAnchorPatterns:
    def test_t_pattern_matches(self):
        matches = ANCHOR_PATTERNS["t"].findall("preface [t=12:34] tail [t=1:05] [t=24]")
        assert matches == ["12:34", "1:05", "24"]

    def test_f_pattern_matches(self):
        assert ANCHOR_PATTERNS["F"].findall("引 [F12] [F03]") == ["12", "03"]

    def test_ch_pattern_matches(self):
        assert ANCHOR_PATTERNS["Ch"].findall("see [Ch3] [CH15]") == ["3", "15"]

    def test_system_prompt_has_all_placeholders(self):
        # Rendering with all keys must not KeyError.
        rendered = COPILOT_SYSTEM.format(
            bv="BV_X",
            title="示例",
            domain="AI 技术",
            direction="注意力机制",
            references_block="（无）",
            max_tool_calls=6,
        )
        assert "BV_X" in rendered
        assert "AI 技术" in rendered
        assert "[t=05:46]" in rendered
        assert "[[evidence]]" in rendered
        assert "[F<" not in rendered
        assert "[Ch<" not in rendered

    def test_system_prompt_prevents_template_four_sections(self):
        rendered = COPILOT_SYSTEM.format(
            bv="BV_X",
            title="示例",
            domain="AI 技术",
            direction="注意力机制",
            references_block="（无）",
            max_tool_calls=6,
        )
        assert "默认只输出 [[evidence]] 1 段" in rendered
        assert "第一行必须是 [[evidence]]" in rendered
        assert "严禁固定输出 4 段" in rendered
        assert "严禁固定输出 4 段" in FINAL_RESPONSE_PROMPT
        assert "即使只有一段也不能省略" in FINAL_RESPONSE_PROMPT
        assert "[[background]]、[[deep_dive]]、[[application]]" in FINAL_RESPONSE_PROMPT

    def test_system_prompt_declares_note_evidence_native_default_strategy(self):
        rendered = COPILOT_SYSTEM.format(
            bv="BV_X",
            title="example",
            domain="AI",
            direction="attention",
            references_block="none",
            max_tool_calls=6,
        )
        assert "Default tool path: search_lecture -> get_note_unit -> search_evidence -> get_evidence_object." in rendered
        assert "Copilot is note-first, evidence-second." in rendered
        assert "Compatibility tools (get_chapter, get_frame, get_quote_context, explain_frame)" in rendered
        assert "Only use compatibility tools when the user explicitly asks for chapter/frame/quote anchors" in rendered
        assert "Treat chapter/frame references as compatibility anchors" in rendered
        assert "they may be omitted entirely from the available tool list" in rendered

    def test_format_references_block_empty(self):
        assert format_references_block(None) == "（无）"
        assert format_references_block([]) == "（无）"

    def test_format_references_block_shapes(self):
        refs = [
            {"kind": "chapter", "id": 3},
            {"kind": "frame", "id": 12},
            {"kind": "selection", "text": "把注意力分数 softmax 过后"},
        ]
        block = format_references_block(refs)
        assert "[Ch3]" in block and "[F12]" in block and "softmax" in block
        assert "兼容章节锚点" in block
        assert "兼容关键帧锚点" in block

    def test_format_references_block_supports_note_and_evidence_kinds(self):
        refs = [
            {"kind": "note", "note_node_id": "block-1", "text": "mainline block"},
            {"kind": "evidence", "evidence_id": "ev-quote-1", "text": "key quote"},
        ]
        block = format_references_block(refs)
        assert "block-1" in block
        assert "ev-quote-1" in block
        assert "mainline block" in block


    def test_system_prompt_makes_note_evidence_path_explicit(self):
        rendered = COPILOT_SYSTEM.format(
            bv="BV_X",
            title="示例",
            domain="AI 技术",
            direction="注意力机制",
            references_block="（无）",
            max_tool_calls=6,
        )
        assert "get_note_unit" in rendered
        assert "search_evidence" in rendered
        assert "get_evidence_object" in rendered
        assert "chapter/frame/quote" in rendered

class TestValidateAnchors:
    async def test_ok_keeps_answer_unchanged(self, tmp_path):
        lec = _make_lecture("BV_VA_OK")  # duration = 120 per _make_lecture fixture
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\ncore point A [t=00:30] see [Ch1] and [F1]."
            patched, warnings = await validate_anchors(answer, "BV_VA_OK", ctx.db, ctx.rag)
            assert patched == answer
            assert warnings == []
        finally:

            await rag.close()

    async def test_bad_timestamp_wrapped(self, tmp_path):
        lec = _make_lecture("BV_VA_BAD")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\n跑偏的锚点 [t=99:59]."
            patched, warnings = await validate_anchors(answer, "BV_VA_BAD", ctx.db, ctx.rag)
            assert "⚠ t=99:59" in patched
            assert warnings and warnings[0]["kind"] == "t"
        finally:
            await rag.close()

    async def test_bad_frame_and_chapter_wrapped(self, tmp_path):
        lec = _make_lecture("BV_VA_CH")  # 1 chapter, 1 frame in visual_evidence
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\nvalid [Ch1]; invalid [Ch9] [F9]"
            patched, warnings = await validate_anchors(answer, "BV_VA_CH", ctx.db, ctx.rag)
            assert " Ch9" in patched
            assert " F9" in patched
            kinds = {w["kind"] for w in warnings}
            assert kinds == {"Ch", "F", "section"}
            assert "evidence_section_chapter_only_anchor" in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_chapter_only_evidence_anchor_warns(self, tmp_path):
        lec = _make_lecture("BV_VA_CH_ONLY")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\nonly chapter fallback [Ch1]"
            patched, warnings = await validate_anchors(answer, "BV_VA_CH_ONLY", ctx.db, ctx.rag)
            assert patched == answer
            assert "evidence_section_chapter_only_anchor" in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_malformed_anchors_are_normalized_and_warned(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_MAL", domain="AI ??", direction="?????")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\nproof at [t=065], [t=24], [CH<2>] and [F<1>]."
            patched, warnings = await validate_anchors(answer, "BV_VA_MAL", ctx.db, ctx.rag)
            assert "[t=01:05]" in patched
            assert "[t=00:24]" in patched
            assert "[Ch2]" in patched
            assert "[F1]" in patched
            assert "[CH<2>]" not in patched
            assert "[F<1>]" not in patched
            reasons = {w["reason"] for w in warnings}
            assert "malformed_seconds" in reasons
            assert "malformed_angle_brackets" in reasons
        finally:
            await rag.close()
    async def test_uppercase_chapter_anchor_is_normalized(self, tmp_path):
        lec = _make_lecture("BV_VA_CH_UPPER")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\ncompat chapter [CH1]."
            patched, warnings = await validate_anchors(answer, "BV_VA_CH_UPPER", ctx.db, ctx.rag)
            assert "[Ch1]" in patched
            assert "[CH1]" not in patched
            assert "evidence_section_chapter_only_anchor" in {w["reason"] for w in warnings}
        finally:
            await rag.close()


    async def test_v2_section_warnings(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_SEC", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            dot = chr(183)
            answer = (
                f"[[evidence]]\nweb in evidence [web {dot} example.com]\n\n"
                "[[unknown]]\nunknown section\n\n"
                f"[[background]]\ncross bv outside extension [BV1abc123XYZ {dot} Ch2]"
            )
            _, warnings = await validate_anchors(answer, "BV_VA_SEC", ctx.db, ctx.rag)
            reasons = {w["reason"] for w in warnings}
            assert "unknown_section_type" in reasons
            assert "evidence_section_no_anchor" in reasons
            assert "web_anchor_in_evidence" in reasons
            assert "cross_bv_anchor_outside_extension" in reasons
        finally:
            await rag.close()


    async def test_missing_section_label_warns(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_NOSEC", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            _, warnings = await validate_anchors("plain answer [Ch1]", "BV_VA_NOSEC", ctx.db, ctx.rag)
            assert {w["reason"] for w in warnings} == {"missing_section_label"}
        finally:
            await rag.close()

    async def test_first_section_must_be_evidence_unless_pure_offtopic(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_FIRST", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[background]]\n鍏堣鑳屾櫙 [t=00:30]\n\n[[evidence]]\n鍚庨潰鍐嶇粰璇佹嵁 [t=00:45]"
            _, warnings = await validate_anchors(answer, "BV_VA_FIRST", ctx.db, ctx.rag)
            assert "first_section_not_evidence" in {w["reason"] for w in warnings}

            _, warnings = await validate_anchors(
                "[[offtopic]]\nThis is off-topic.",
                "BV_VA_FIRST",
                ctx.db,
                ctx.rag,
            )
            assert "first_section_not_evidence" not in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_evidence_section_cannot_start_with_compatibility_framing(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_FRAME", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[evidence]]\n[Ch1] 鏍稿績鍒ゆ柇鍦ㄥ悗闈?[t=00:30]"
            _, warnings = await validate_anchors(answer, "BV_VA_FRAME", ctx.db, ctx.rag)
            assert "evidence_section_starts_with_compatibility_framing" in {
                w["reason"] for w in warnings
            }
            assert "evidence_section_chapter_only_anchor" not in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_web_anchor_allowed_outside_evidence(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_WEB_OK", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            dot = chr(183)
            answer = f"[[background]]\n补充资料 [web {dot} example.com]"
            _, warnings = await validate_anchors(answer, "BV_VA_WEB_OK", ctx.db, ctx.rag)
            assert "web_anchor_in_evidence" not in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_cross_bv_anchor_allowed_in_extension(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_XBV_OK", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            dot = chr(183)
            answer = f"[[extension]]\n可参考 [BV1abc123XYZ {dot} Ch2]"
            _, warnings = await validate_anchors(answer, "BV_VA_XBV_OK", ctx.db, ctx.rag)
            assert "cross_bv_anchor_outside_extension" not in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_malformed_web_anchor_warns(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_WEB_BAD", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = "[[background]]\n补充资料 [web: example.com]"
            _, warnings = await validate_anchors(answer, "BV_VA_WEB_BAD", ctx.db, ctx.rag)
            assert "malformed_web_anchor" in {w["reason"] for w in warnings}
        finally:
            await rag.close()

    async def test_section_count_duplicate_and_offtopic_rules_warn(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_VA_SECTION_RULES", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            answer = (
                "[[evidence]]\n证据 [Ch1]\n\n"
                "[[background]]\n背景\n\n"
                "[[deep_dive]]\n原理\n\n"
                "[[application]]\n应用\n\n"
                "[[background]]\n重复背景\n\n"
                "[[offtopic]]\n跑题引导"
            )
            _, warnings = await validate_anchors(answer, "BV_VA_SECTION_RULES", ctx.db, ctx.rag)
            reasons = {w["reason"] for w in warnings}
            assert "evidence_section_chapter_only_anchor" in reasons
            assert "section_count_exceeded" in reasons
            assert "support_section_triple_combo" in reasons
            assert "duplicated_section_type" in reasons
            assert "offtopic_mixed_with_answer" in reasons
        finally:
            await rag.close()



class TestBuildInitialState:
    async def test_injects_taxonomy_and_question(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_BIS", domain="AI 技术", direction="注意力机制")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            state = await build_initial_state(
                ctx,
                "BV_BIS",
                "KV cache 做了什么？",
                references=[{"kind": "chapter", "id": 1}],
            )
            assert state["bv"] == "BV_BIS"
            assert state["step_count"] == 0
            assert len(state["messages"]) == 2
            sys_msg = state["messages"][0]
            assert "BV_BIS" in sys_msg.content
            assert "AI 技术" in sys_msg.content
            assert "注意力机制" in sys_msg.content
            assert "[Ch1]" in sys_msg.content  # reference rendered
            human = state["messages"][1]
            assert human.content == "KV cache 做了什么？"
        finally:
            await rag.close()


    async def test_injects_note_outline_and_evidence_context_when_present(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_BIS_NOTE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            state = await build_initial_state(ctx, "BV_BIS_NOTE", "这节课主线是什么？")
            assert len(state["messages"]) == 3
            note_context = state["messages"][1]
            assert "Lecture Note IR" in note_context.content
            assert "主线单元一" in note_context.content
            assert "Evidence Index" in note_context.content
            assert "ev-quote-1" in note_context.content
            assert state["messages"][2].content == "这节课主线是什么？"
        finally:
            await rag.close()


    async def test_renders_note_and_evidence_references_into_system_prompt(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_BIS_REF"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            refs = [
                {"kind": "note", "note_node_id": "block-1", "text": "mainline block"},
                {"kind": "evidence", "evidence_id": "ev-quote-1", "text": "key quote"},
            ]
            state = await build_initial_state(
                ctx,
                "BV_BIS_REF",
                "use references",
                references=refs,
            )
            sys_msg = state["messages"][0]
            rendered_block = format_references_block(refs)
            assert "block-1" in sys_msg.content
            assert "ev-quote-1" in sys_msg.content
            assert rendered_block in sys_msg.content
        finally:
            await rag.close()


class _StubChatModel:
    """Minimal ChatOpenAI-compatible stub: tool call on turn 1, final answer on turn 2."""

    def __init__(self, *, tool_name: str, tool_args: dict, final: str):
        self._tool_name = tool_name
        self._tool_args = tool_args
        self._final = final
        self.calls = 0
        self.bound_tool_names: list[str] | None = None

    def bind_tools(self, tools):
        self.bound_tool_names = [getattr(t, "name", None) for t in tools]
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{self.calls}",
                        "name": self._tool_name,
                        "args": self._tool_args,
                    }
                ],
            )
        return AIMessage(content=self._final)


class _BoundToolLoopModel:
    def __init__(self, parent):
        self._parent = parent

    async def ainvoke(self, messages, **_kwargs):  # noqa: ARG002
        self._parent.bound_calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"loop_{self._parent.bound_calls}",
                    "name": "get_chapter",
                    "args": {"bv": self._parent.bv, "chapter_idx": 1},
                }
            ],
        )


class _BudgetLoopModel:
    def __init__(self, bv: str, final: str):
        self.bv = bv
        self.final = final
        self.bound_calls = 0
        self.final_calls = 0

    def bind_tools(self, tools):  # noqa: ARG002
        return _BoundToolLoopModel(self)

    async def ainvoke(self, messages, **_kwargs):  # noqa: ARG002
        self.final_calls += 1
        return AIMessage(content=self.final)


class _BoundEmptyAfterToolModel:
    def __init__(self, parent):
        self._parent = parent

    async def ainvoke(self, messages, **_kwargs):  # noqa: ARG002
        self._parent.bound_calls += 1
        if self._parent.bound_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "empty_1",
                        "name": "get_chapter",
                        "args": {"bv": self._parent.bv, "chapter_idx": 1},
                    }
                ],
            )
        return AIMessage(content="")


class _EmptyAfterToolModel:
    def __init__(self, bv: str, final: str):
        self.bv = bv
        self.final = final
        self.bound_calls = 0
        self.final_calls = 0

    def bind_tools(self, tools):  # noqa: ARG002
        return _BoundEmptyAfterToolModel(self)

    async def ainvoke(self, messages, **_kwargs):  # noqa: ARG002
        self.final_calls += 1
        return AIMessage(content=self.final)


class TestAgentInvoke:
    async def test_invoke_mock_llm_runs_tool_and_returns_anchored_answer(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_AG", domain="AI 技术", direction="注意力机制")
        ctx, rag = await _build_ctx(tmp_path, [_attach_note_and_evidence(lec)])
        try:
            stub = _StubChatModel(
                tool_name="get_note_unit",
                tool_args={"bv": "BV_AG", "unit_id": "unit-1"},
                final="第一章讲了核心概念 [t=00:30] 详见 [Ch1] 和 [F1]。",
            )
            graph = build_graph(ctx, model=stub, question="核心问题是什么？")
            init = await build_initial_state(ctx, "BV_AG", "这一章的核心问题？")
            final = await graph.ainvoke(init)

            # Two LLM turns (tool-call, final answer).
            assert stub.calls == 2
            assert stub.bound_tool_names == [
                "search_lecture",
                "get_note_unit",
                "search_evidence",
                "get_evidence_object",
                "search_lectures",
            ]
            # step_count increments once per call_model, and tools ran once.
            assert final["step_count"] == 2
            msg_types = [type(m).__name__ for m in final["messages"]]
            assert "SystemMessage" in msg_types
            assert "HumanMessage" in msg_types
            assert msg_types.count("AIMessage") == 2
            assert "ToolMessage" in msg_types
            # Final assistant content has the anchors the stub emitted.
            last = final["messages"][-1]
            assert "[Ch1]" in last.content
        finally:
            await rag.close()

    async def test_budget_exhaustion_runs_plain_final_model(self, tmp_path):
        from app.config import get_settings

        lec = _lecture_with_taxonomy("BV_BUDGET", domain="AI 技术", direction="注意力机制")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            settings = get_settings()
            stub = _BudgetLoopModel(
                bv="BV_BUDGET",
                final="已经基于现有工具结果汇总答案 [Ch1]。",
            )
            graph = build_graph(ctx, model=stub, question="第1章怎么讲？")
            init = await build_initial_state(ctx, "BV_BUDGET", "一直检索会怎样？")
            final = await graph.ainvoke(init)

            assert stub.bound_calls == settings.copilot_max_tool_calls
            assert stub.final_calls == 1
            assert final["step_count"] == settings.copilot_max_tool_calls + 1
            assert final["messages"][-1].content == "已经基于现有工具结果汇总答案 [Ch1]。"
        finally:
            await rag.close()

    async def test_empty_ai_after_tool_runs_plain_final_model(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_EMPTY_FINAL", domain="AI 技术", direction="注意力机制")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            stub = _EmptyAfterToolModel(
                bv="BV_EMPTY_FINAL",
                final="工具后空响应已被兜底为最终答案 [Ch1]。",
            )
            graph = build_graph(ctx, model=stub, question="第1章怎么看？")
            init = await build_initial_state(ctx, "BV_EMPTY_FINAL", "工具后空响应怎么办？")
            final = await graph.ainvoke(init)

            assert stub.bound_calls == 2
            assert stub.final_calls == 1
            assert final["messages"][-1].content == "工具后空响应已被兜底为最终答案 [Ch1]。"
        finally:
            await rag.close()

    async def test_quick_ask_validates_anchors(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_QA", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [_attach_note_and_evidence(lec)])
        try:
            stub = _StubChatModel(
                tool_name="search_evidence",
                tool_args={"bv": "BV_QA", "query": "归一化", "top_k": 3},
                final="[[evidence]]\nclaim [t=00:30]; invalid [Ch9] [F9].",
            )
            answer, warnings = await quick_ask(
                ctx, "BV_QA", "core question?", model=stub
            )
            assert " Ch9" in answer
            assert " F9" in answer
            assert "[t=00:30]" in answer  # valid anchor stays
            assert {w["kind"] for w in warnings} == {"Ch", "F"}
        finally:
            await rag.close()



# ---------------------------------------------------------------------------
# Stage 5 — Copilot API (SSE / lectures / taxonomy)
# ---------------------------------------------------------------------------


import base64  # noqa: E402
import json as _json  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402

from app.copilot.api import (  # noqa: E402
    Reference,
    _extract_chunk_text,
    _sse_frame,
    _strip_thinking_tags,
    _summarise_tool_output,
    router as copilot_router,
    run_agent_sse,
)


class _FakeGraph:
    """Pre-recorded ``astream_events`` for SSE-layer tests.

    Mimics just enough of LangGraph's event shape that ``run_agent_sse``
    can drive it without pulling in a real model.
    """

    def __init__(self, events):
        self._events = list(events)

    async def astream_events(self, _state, *, version="v2"):  # noqa: ARG002
        for ev in self._events:
            yield ev


def _ai_chunk(text):
    class _Chunk:
        content = text

    return _Chunk()


class TestDeepSeekThinkingDisable:
    """Regression for the multi-turn 400 we hit on 2026-05-11:

    DeepSeek V4 (``api.deepseek.com``) defaults to thinking mode for any
    tool-call response. Its raw ``reasoning_content`` is *not* surfaced
    by langchain-openai into ``additional_kwargs``, so the Copilot
    ReAct loop cannot echo it back, and the second turn 400s with
    ``"reasoning_content in the thinking mode must be passed back"``.

    The fix lives in :meth:`Settings.text_extra_body` — it must emit the
    DeepSeek-shaped ``{"thinking": {"type": "disabled"}}`` whenever the
    text base URL points at deepseek.com **and** the user has not opted
    back into thinking via ``QWEN_TEXT_ENABLE_THINKING=true``.
    ``_build_model`` in the Copilot reuses this helper, keeping the
    lecturize / critic / reviser path and the Copilot path in lock-step.
    """

    def _settings(self, **overrides):
        from app.config import Settings

        defaults = dict(
            dashscope_api_key="placeholder",
            deepseek_api_key="placeholder",
            deepseek_base_url="https://api.deepseek.com",
            qwen_text_model="deepseek-v4-flash",
            qwen_copilot_model="deepseek-v4-flash",
            basic_auth_password="placeholder",
        )
        defaults.update(overrides)
        return Settings(**defaults)

    def test_deepseek_base_url_disables_thinking_by_default(self):
        body = self._settings().text_extra_body()
        assert body == {"thinking": {"type": "disabled"}}

    def test_dashscope_keeps_legacy_enable_thinking_key(self):
        body = self._settings(
            deepseek_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).text_extra_body()
        assert body == {"enable_thinking": False}

    def test_user_opt_in_enables_thinking_on_deepseek(self):
        body = self._settings(qwen_text_enable_thinking=True).text_extra_body()
        assert body == {"thinking": {"type": "enabled"}}

    def test_unknown_base_url_returns_empty_extra_body(self):
        body = self._settings(deepseek_base_url="https://example.com/v1").text_extra_body()
        assert body == {}

    def test_copilot_build_model_forwards_text_extra_body(self, monkeypatch):
        """``_build_model`` must thread ``text_extra_body()`` into ChatOpenAI.

        We replace ``ChatOpenAI`` with a recorder so we can inspect the
        kwargs without needing the real langchain-openai package, then
        assert the disable struct showed up exactly once.
        """
        from app.config import get_settings
        from app.copilot import agent as agent_mod

        captured: dict[str, object] = {}

        class _Recorder:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        # ``_build_model`` does ``from langchain_openai import ChatOpenAI``
        # inside the function, so monkeypatch the module attribute it will
        # resolve.
        import langchain_openai

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)

        # Force a fresh Settings object pointing at deepseek.com so the
        # process-wide cached settings (which may use any URL) cannot leak.
        settings = get_settings()
        monkeypatch.setattr(settings, "deepseek_base_url", "https://api.deepseek.com")
        monkeypatch.setattr(settings, "qwen_text_enable_thinking", False)

        agent_mod._build_model()
        assert captured.get("extra_body") == {"thinking": {"type": "disabled"}}


class TestSSEHelpers:
    def test_sse_frame_encodes_cjk(self):
        raw = _sse_frame("token", {"delta": "中文 delta"})
        assert raw.startswith(b"event: token\n")
        assert b"\n\n" in raw
        assert "中文 delta" in raw.decode("utf-8")

    def test_extract_chunk_text_string(self):
        assert _extract_chunk_text(_ai_chunk("hello")) == "hello"

    def test_extract_chunk_text_parts(self):
        class _C:
            content = [
                {"type": "text", "text": "A "},
                {"type": "tool_use", "name": "x"},
                {"type": "text", "text": "B"},
            ]

        assert _extract_chunk_text(_C()) == "A B"

    def test_extract_chunk_text_nested_generations(self):
        class _Generation:
            message = AIMessage(content="最终答案 [Ch1]")

        class _Result:
            generations = [[_Generation()]]

        assert _extract_chunk_text(_Result()) == "最终答案 [Ch1]"

    def test_strip_thinking_tags_drops_bare_tags(self):
        assert _strip_thinking_tags("</think>\n") == ""
        assert _strip_thinking_tags("<think>答案</think>") == "答案"

    def test_summarise_tool_output_chunks(self):
        out = _summarise_tool_output(
            "search_lecture", {"chunks": [{}, {}, {}]}
        )
        assert out == {"name": "search_lecture", "chunks_count": 3, "ok": True}

    def test_summarise_tool_output_error(self):
        out = _summarise_tool_output(
            "get_chapter", {"error": {"code": "not_found", "message": "x"}}
        )
        assert out["ok"] is False
        assert out["error"]["code"] == "not_found"

    def test_summarise_tool_output_note_and_evidence_counts(self):
        out = _summarise_tool_output(
            "get_frame",
            {
                "frame_id": 1,
                "caption": "架构图",
                "path": "/img/ch1_diagram.jpg",
                "compatibility_anchor_kind": "frame",
                "compatibility_anchor_id": "F1",
                "primary_ref_kind": "evidence",
                "primary_ref_id": "ev-frame-1",
                "primary_evidence_id": "ev-frame-1",
                "primary_note_node_id": "unit-1",
                "primary_note_unit_id": "unit-1",
                "primary_note_node_type": "teaching_unit",
                "primary_note_title": "主线单元一",
                "primary_evidence_kind": "frame_evidence",
                "related_evidence_ids": ["ev-frame-1"],
                "related_note_node_ids": ["unit-1", "block-1"],
            },
        )
        assert out["evidence_count"] == 1
        assert out["note_nodes_count"] == 2
        assert out["primary_ref_kind"] == "evidence"
        assert out["primary_ref_id"] == "ev-frame-1"
        assert out["primary_evidence_id"] == "ev-frame-1"
        assert out["primary_note_node_id"] == "unit-1"
        assert out["primary_note_unit_id"] == "unit-1"
        assert out["primary_note_node_type"] == "teaching_unit"
        assert out["primary_note_title"] == "主线单元一"
        assert out["primary_evidence_kind"] == "frame_evidence"
        assert out["compatibility_anchor_kind"] == "frame"
        assert out["compatibility_anchor_id"] == "F1"
        assert out["caption"] == "架构图"
        assert out["path"] == "/img/ch1_diagram.jpg"
        assert out["evidence_id"] == "ev-frame-1"
        assert out["note_node_id"] == "unit-1"

    def test_summarise_tool_output_keeps_display_fields_for_core_four(self):
        chapter = _summarise_tool_output(
            "get_chapter",
            {"title": "第一章", "compatibility_anchor_kind": "chapter", "compatibility_anchor_id": "Ch1"},
        )
        frame = _summarise_tool_output(
            "get_frame",
            {"frame_id": 1, "caption": "架构图", "path": "/img/ch1_diagram.jpg"},
        )
        quote = _summarise_tool_output(
            "get_quote_context",
            {"matched_quote": "原字幕：A 就是 X，因为先比较再归一化。"},
        )
        explain = _summarise_tool_output(
            "explain_frame",
            {"frame_id": 1, "why_useful": "说明 A 与 B 的依赖关系"},
        )
        assert chapter["title"] == "第一章"
        assert frame["caption"] == "架构图"
        assert frame["path"] == "/img/ch1_diagram.jpg"
        assert quote["matched_quote"].startswith("原字幕")
        assert explain["why_useful"] == "说明 A 与 B 的依赖关系"

    def test_summarise_tool_output_does_not_invent_primary_or_compatibility_shape(self):
        out = _summarise_tool_output(
            "get_frame",
            {
                "frame_id": 1,
                "chapter_idx": 2,
                "related_evidence_ids": ["ev-frame-1"],
                "related_note_node_ids": ["unit-1"],
            },
        )
        assert out["frame_id"] == 1
        assert out["chapter_idx"] == 2
        assert out["evidence_count"] == 1
        assert out["note_nodes_count"] == 1
        assert "primary_ref_kind" not in out
        assert "primary_ref_id" not in out
        assert "primary_evidence_id" not in out
        assert "primary_note_node_id" not in out
        assert "evidence_id" not in out
        assert "note_node_id" not in out
        assert "compatibility_anchor_kind" not in out
        assert "compatibility_anchor_id" not in out

    def test_summarise_tool_output_keeps_dedicated_explicit_aliases_without_primary_backfill(self):
        out = _summarise_tool_output(
            "get_evidence_object",
            {
                "evidence": {"evidence_id": "ev-quote-1"},
                "requested_node_id": "block-1",
                "related_evidence_ids": ["ev-quote-1"],
                "related_note_node_ids": ["block-1"],
            },
        )
        assert out["evidence_count"] == 1
        assert out["note_nodes_count"] == 1
        assert out["evidence_id"] == "ev-quote-1"
        assert out["note_node_id"] == "block-1"
        assert "primary_evidence_id" not in out
        assert "primary_note_node_id" not in out

    def test_reference_accepts_string_native_ids(self):
        ref = Reference.model_validate(
            {"kind": "evidence", "id": "ev-quote-1", "evidence_id": "ev-quote-1"}
        )
        assert ref.id == "ev-quote-1"
        assert ref.evidence_id == "ev-quote-1"


class TestCopilotRealE2EReport:
    def test_report_records_section_and_web_diagnostics(self, tmp_path):
        from scripts._verify_copilot_real_e2e import _write_report

        out = tmp_path / "report.md"
        events = [
            ("tool_call", {"name": "web_search", "args": {"query": "x"}}),
            ("tool_result", {"name": "web_search", "ok": False}),
            ("token", {"delta": "[[evidence]]\nA [Ch1]\n\n"}),
            ("token", {"delta": "[[background]]\nB\n\n"}),
            ("token", {"delta": "[[deep_dive]]\nC\n\n"}),
            ("token", {"delta": "[[application]]\nD\n\n"}),
            ("token", {"delta": "[[boundary]]\nE"}),
            (
                "done",
                {
                    "warnings": [
                        {"kind": "section", "value": "x", "reason": "section_count_exceeded"}
                    ],
                    "patched_answer": None,
                },
            ),
        ]
        _write_report(
            out,
            url="http://test",
            bv="BV_TEST",
            question="问题",
            status_code=200,
            elapsed=1.0,
            use_web=True,
            events=events,
        )
        text = out.read_text(encoding="utf-8")
        assert "section_count: 5" in text
        assert "section_count_exceeded: True" in text
        assert "web_search_called: True" in text
        assert "warning_codes: section_count_exceeded" in text


class TestRunAgentSse:
    async def test_emits_token_tool_and_done(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_SSE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            events = [
                {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": _ai_chunk("")},
                },
                {
                    "event": "on_tool_start",
                    "name": "get_chapter",
                    "data": {"input": {"bv": "BV_SSE", "chapter_idx": 1}},
                },
                {
                    "event": "on_tool_end",
                    "name": "get_chapter",
                    "data": {"output": {"chapter_idx": 1, "title": "t"}},
                },
                {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": _ai_chunk("[[evidence]]\nchapter [t=00:05] ")},
                },
                {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": _ai_chunk("[Ch1] [Ch9] [F9]")},
                },
            ]
            frames: list[bytes] = []
            async for b in run_agent_sse(_FakeGraph(events), {}, ctx, "BV_SSE"):
                frames.append(b)

            text = b"\n".join(frames).decode("utf-8")
            assert "event: token" in text
            assert "event: tool_call" in text
            assert "event: tool_result" in text
            assert "event: done" in text
            # The final ``done`` payload must carry anchor-validation warnings
            # for [Ch9] and [F9] and a patched_answer wrapping them with ⚠.
            done_line = [f for f in frames if f.startswith(b"event: done")][0]
            payload = _json.loads(done_line.decode("utf-8").split("data: ", 1)[1])
            assert payload["anchors_validated"] is True
            assert payload["note_context"]["source"] == "lecture_note_ir"
            assert payload["evidence_context"]["source"] == "evidence_index"
            kinds = {w["kind"] for w in payload["warnings"]}
            assert kinds == {"Ch", "F"}
            assert payload["patched_answer"] is not None
            assert "⚠ Ch9" in payload["patched_answer"]
            assert "⚠ F9" in payload["patched_answer"]
        finally:
            await rag.close()

    async def test_done_payload_carries_framing_contract_warnings(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_SSE_FRAME"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            events = [
                {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": _ai_chunk("[[background]]\nContext first [t=00:05]\n\n")},
                },
                {
                    "event": "on_chat_model_stream",
                    "data": {
                        "chunk": _ai_chunk(
                            "[[evidence]]\n[Ch1] Core point comes after this [t=00:30]"
                        )
                    },
                },
            ]
            frames: list[bytes] = []
            async for b in run_agent_sse(_FakeGraph(events), {}, ctx, "BV_SSE_FRAME"):
                frames.append(b)

            done_line = [f for f in frames if f.startswith(b"event: done")][0]
            payload = _json.loads(done_line.decode("utf-8").split("data: ", 1)[1])
            reasons = {w["reason"] for w in payload["warnings"]}
            assert "first_section_not_evidence" in reasons
            assert "evidence_section_starts_with_compatibility_framing" in reasons
        finally:
            await rag.close()

    async def test_surfaces_internal_error(self, tmp_path):
        lec = _make_lecture("BV_ERR")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:

            class _BoomGraph:
                async def astream_events(self, _s, *, version="v2"):  # noqa: ARG002
                    yield {"event": "on_chat_model_stream", "data": {"chunk": _ai_chunk("hi")}}
                    raise RuntimeError("boom")

            frames: list[bytes] = []
            async for b in run_agent_sse(_BoomGraph(), {}, ctx, "BV_ERR"):
                frames.append(b)

            text = b"\n".join(frames).decode("utf-8")
            assert "event: error" in text
            assert "boom" in text
        finally:
            await rag.close()


# ---- FastAPI integration (TestClient) ----


def _basic_headers() -> dict[str, str]:
    from app.config import get_settings

    s = get_settings()
    token = base64.b64encode(
        f"{s.basic_auth_user}:{s.basic_auth_password}".encode("utf-8")
    ).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _build_app_with_state(ctx):
    """Mount a minimal FastAPI app that exposes ``copilot_router`` only."""
    app = FastAPI()
    app.state.db = ctx.db
    app.state.rag = ctx.rag
    app.state.pipeline = None
    app.include_router(copilot_router)
    return app


class TestCopilotLecturesEndpoint:
    async def test_returns_taxonomy_fields(self, tmp_path):
        lec_a = _make_lecture("BV_LL1")
        lec_b = _make_lecture("BV_LL2")
        ctx, rag = await _build_ctx(
            tmp_path,
            [lec_a, lec_b],
            rows_meta={
                "BV_LL1": {"domain": "AI 技术", "direction": "注意力机制"},
                "BV_LL2": {"domain": "健身运动", "direction": "脂肪燃烧"},
            },
        )
        try:
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.get("/api/copilot/lectures", headers=_basic_headers())
            assert resp.status_code == 200
            data = resp.json()
            bvs = {d["bv"]: d for d in data}
            assert bvs["BV_LL1"]["domain"] == "AI 技术"
            assert bvs["BV_LL2"]["direction"] == "脂肪燃烧"
        finally:
            await rag.close()


class TestCopilotTaxonomyEndpoint:
    async def test_patch_domain_and_direction(self, tmp_path):
        lec = _make_lecture("BV_TX")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/taxonomy",
                    headers=_basic_headers(),
                    json={
                        "bv": "BV_TX",
                        "domain": "AI 技术",
                        "direction": "注意力机制",
                        "tags": ["Attention"],
                    },
                )
            assert resp.status_code == 200
            assert resp.json()["ok"] is True
            row = await ctx.db.get_summary("BV_TX")
            assert row is not None
            assert row.domain == "AI 技术"
            assert row.direction == "注意力机制"
            assert "Attention" in row.domain_tags
        finally:
            await rag.close()

    async def test_404_on_unknown_bv(self, tmp_path):
        lec = _make_lecture("BV_TX2")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/taxonomy",
                    headers=_basic_headers(),
                    json={"bv": "BV_NOPE", "domain": "AI 技术"},
                )
            assert resp.status_code == 404
        finally:
            await rag.close()


class TestCopilotAskRoute:
    async def test_404_on_unknown_bv(self, tmp_path):
        lec = _make_lecture("BV_ASK1")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/ask",
                    headers=_basic_headers(),
                    json={"bv": "BV_NOPE", "question": "hi", "references": []},
                )
            assert resp.status_code == 404
        finally:
            await rag.close()

    async def test_preserves_note_and_evidence_reference_identity(self, tmp_path, monkeypatch):
        import app.copilot.api as api_mod
        from langchain_core.messages import HumanMessage, SystemMessage

        lec = _attach_note_and_evidence(_make_lecture("BV_ASK_NATIVE"))
        ctx, rag = await _build_ctx(tmp_path, [lec])
        captured: dict[str, object] = {}

        async def _fake_build_initial_state(
            _ctx,
            bv,
            question,
            references=None,
            history=None,
        ):
            captured["initial_state"] = {
                "bv": bv,
                "question": question,
                "references": references,
                "history": history,
            }
            return {
                "messages": [
                    SystemMessage(content="system"),
                    HumanMessage(content=question),
                ]
            }

        def _fake_build_graph(_ctx, **kwargs):
            captured["graph"] = kwargs
            return _FakeGraph([])

        monkeypatch.setattr(api_mod.agent_mod, "build_initial_state", _fake_build_initial_state)
        monkeypatch.setattr(api_mod.agent_mod, "build_graph", _fake_build_graph)

        try:
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/ask",
                    headers=_basic_headers(),
                    json={
                        "bv": "BV_ASK_NATIVE",
                        "question": "聚焦这两个原生引用",
                        "references": [
                            {
                                "kind": "note",
                                "id": "block-1",
                                "note_node_id": "block-1",
                                "text": "mainline block",
                            },
                            {
                                "kind": "evidence",
                                "id": "ev-quote-1",
                                "evidence_id": "ev-quote-1",
                                "text": "key quote",
                            },
                        ],
                    },
                )
            assert resp.status_code == 200
            assert "event: done" in resp.text

            initial = captured["initial_state"]
            graph = captured["graph"]
            assert initial["references"] == graph["references"]
            assert initial["references"][0]["kind"] == "note"
            assert initial["references"][0]["id"] == "block-1"
            assert initial["references"][0]["note_node_id"] == "block-1"
            assert initial["references"][1]["kind"] == "evidence"
            assert initial["references"][1]["id"] == "ev-quote-1"
            assert initial["references"][1]["evidence_id"] == "ev-quote-1"
        finally:
            await rag.close()


class TestCopilotReindexEndpoint:
    async def test_taxonomy_only_patches_rows(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_RX", domain="AI 技术", direction="注意力机制")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            # Corrupt the taxonomy column so reindex has to normalise.
            await ctx.db.update_taxonomy("BV_RX", domain=None, direction=None)
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/reindex",
                    headers=_basic_headers(),
                    json={"taxonomy_only": True},
                )
            assert resp.status_code == 200
            row = await ctx.db.get_summary("BV_RX")
            assert row is not None
            assert row.domain == "AI 技术"
        finally:
            await rag.close()

    async def test_missing_args_returns_400(self, tmp_path):
        lec = _make_lecture("BV_RX2")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/reindex",
                    headers=_basic_headers(),
                    json={},
                )
            assert resp.status_code == 400
        finally:
            await rag.close()


# ---------------------------------------------------------------------------
# Stage 6 — Frontend injection (renderer + static + template)
# ---------------------------------------------------------------------------


class TestFrontendRender:
    def test_render_injects_copilot_stub_and_js(self, tmp_path, monkeypatch):
        """Renderer must emit <aside id="cp-panel"> + copilot.js <script>."""
        from app.config import get_settings
        from app.render.renderer import Renderer

        # Force reports_dir to a throwaway location so the test does not
        # overwrite real data/reports output.
        settings = get_settings()
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

        lec = _lecture_with_taxonomy("BV_FE")
        renderer = Renderer()
        css = renderer.load_inline_css()
        out_path = renderer.render_lecture(lec, css)
        html = out_path.read_text(encoding="utf-8")

        assert '<aside id="cp-panel"' in html
        assert 'data-bv="BV_FE"' in html
        assert '/static/copilot.js' in html
        assert 'marked.min.js' in html  # marked CDN

        # Chapters are tagged with data-ref-kind=chapter + data-ch-start/end.
        assert 'data-ref-kind="chapter"' in html
        assert 'data-ch-start="0"' in html
        assert 'data-ch-end="200"' in html

        # Frame figures get the global flat id when present.
        assert 'data-ref-kind="frame"' in html
        assert 'data-ref-id="1"' in html

        # Inline Copilot CSS is injected (non-empty).
        assert '.cp-aside' in html
        assert '.cp-anchor' in html

    def test_frame_id_map_matches_flat_frames(self, tmp_path, monkeypatch):
        """The frame_id_map passed to Jinja must match Stage 3's _flat_frames."""
        from app.config import get_settings
        from app.copilot.tools import _flat_frames
        from app.render.renderer import Renderer

        settings = get_settings()
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

        lec = _lecture_with_taxonomy("BV_FE2")
        expected = {fr.path: idx for idx, (fr, _) in enumerate(_flat_frames(lec), start=1) if fr.path}
        assert expected
        renderer = Renderer()
        html = renderer.render_lecture(lec, renderer.load_inline_css()).read_text(encoding="utf-8")
        for path, fid in expected.items():
            # The template emits ``data-ref-id="<id>"`` on the figure.
            assert f'data-ref-id="{fid}"' in html, f"frame id {fid} missing for {path}"

    def test_note_and_evidence_native_dom_metadata_rendered(self, tmp_path, monkeypatch):
        from app.config import get_settings
        from app.render.renderer import Renderer

        settings = get_settings()
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_FE_NOTE"))
        renderer = Renderer()
        html = renderer.render_lecture(lec, renderer.load_inline_css()).read_text(encoding="utf-8")

        assert 'data-ref-kind="note"' in html
        assert 'data-note-node-id="unit-1"' in html
        assert 'data-note-node-id="block-1"' in html
        assert 'data-evidence-id="ev-quote-1"' in html


class TestStaticServing:
    def test_static_assets_reachable(self):
        from app.main import app

        with TestClient(app) as client:
            r = client.get("/static/copilot.css")
            assert r.status_code == 200
            assert "cp-aside" in r.text
            assert "cp-clear" in r.text
            assert "cp-header-actions" in r.text
            r = client.get("/static/copilot.js")
            assert r.status_code == 200
            assert "CopilotPanel" in r.text
            assert "STATE_KEY_PREFIX" in r.text
            assert "clearConversation" in r.text


class TestBuildInitialStateHistory:
    async def test_history_turns_spliced(self, tmp_path):
        from app.copilot.agent import build_initial_state
        from langchain_core.messages import AIMessage, HumanMessage

        lec = _lecture_with_taxonomy("BV_HIST")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            state = await build_initial_state(
                ctx,
                "BV_HIST",
                "下一个问题",
                history=[
                    {"role": "user", "content": "上一轮问题"},
                    {"role": "assistant", "content": "上一轮答案"},
                    {"role": "tool", "content": "should be dropped"},
                    {"role": "user", "content": ""},
                ],
            )
            # Should be [System, Human(prev), AI(prev), Human(new)]
            roles = [type(m).__name__ for m in state["messages"]]
            assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
            assert isinstance(state["messages"][1], HumanMessage)
            assert state["messages"][1].content == "上一轮问题"
            assert isinstance(state["messages"][2], AIMessage)
            assert state["messages"][2].content == "上一轮答案"
            assert state["messages"][-1].content == "下一个问题"
        finally:
            await rag.close()



class TestWebSearchParsing:
    def test_extract_web_results_from_nested_data(self):
        data = {"data": {"web_results": [{"title": "A", "url": "https://example.com"}]}}
        assert _extract_web_results(data)[0]["title"] == "A"

    def test_extract_web_results_from_top_level_list(self):
        assert _extract_web_results([{"title": "A"}, "bad"])[0]["title"] == "A"



class TestMakeCopilotToolsShape:
    async def test_returns_tools_with_note_and_evidence_names(self, tmp_path):
        lec = _make_lecture("BV_MT")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            tools = make_copilot_tools(ctx)
            names = [t.name for t in tools]
            assert names == [
                "search_lecture",
                "get_note_unit",
                "search_evidence",
                "get_evidence_object",
                "search_lectures",
                "get_chapter",
                "get_frame",
                "get_quote_context",
                "explain_frame",
            ]
            # Every tool must have a non-empty description (fastmcp/LangChain
            # both pull it from the docstring first line).
            assert all(t.description for t in tools)
        finally:
            await rag.close()

    async def test_can_hide_compatibility_tools_for_default_runtime(self, tmp_path):
        lec = _make_lecture("BV_MT_NOTE")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            tools = make_copilot_tools(ctx, include_compatibility_tools=False)
            names = [t.name for t in tools]
            assert names == [
                "search_lecture",
                "get_note_unit",
                "search_evidence",
                "get_evidence_object",
                "search_lectures",
            ]
        finally:
            await rag.close()


    async def test_enable_web_appends_web_search(self, tmp_path):
        lec = _make_lecture("BV_MT_WEB")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            tools = make_copilot_tools(ctx, enable_web=True)
            names = [t.name for t in tools]
            assert names[-1] == "web_search"
            assert "search_lectures" in names
        finally:
            await rag.close()

    async def test_explicit_anchor_question_keeps_compatibility_tools_available(self, tmp_path):
        lec = _lecture_with_taxonomy("BV_AG_COMPAT", domain="AI", direction="attention")
        ctx, rag = await _build_ctx(tmp_path, [_attach_note_and_evidence(lec)])
        try:
            stub = _StubChatModel(
                tool_name="get_chapter",
                tool_args={"bv": "BV_AG_COMPAT", "chapter_idx": 1},
                final="第一章补充定位 [Ch1]。",
            )
            graph = build_graph(ctx, model=stub, question="第1章具体讲了什么？")
            init = await build_initial_state(ctx, "BV_AG_COMPAT", "第1章具体讲了什么？")
            await graph.ainvoke(init)

            assert stub.bound_tool_names == [
                "search_lecture",
                "get_note_unit",
                "search_evidence",
                "get_evidence_object",
                "search_lectures",
                "get_chapter",
                "get_frame",
                "get_quote_context",
                "explain_frame",
            ]
        finally:
            await rag.close()


class TestFrontendReferencePayload:
    def test_visible_ref_labels_are_note_evidence_native_first(self):
        body = Path("app/static/copilot.js").read_text(encoding="utf-8")
        assert 'return "讲义节点";' in body
        assert 'return "证据对象";' in body
        assert 'return "兼容章节";' in body
        assert 'return "兼容关键帧";' in body
        assert 'return `兼容章节 Ch${ref.id}`;' in body
        assert 'return `兼容关键帧 F${ref.id}`;' in body

    def test_chip_prefers_native_dom_metadata_and_native_kind_promotion(self):
        body = Path("app/static/copilot.js").read_text(encoding="utf-8")
        assert 'const noteNodeId = host.dataset.noteNodeId || null' in body
        assert 'const evidenceId = host.dataset.evidenceId || null' in body
        assert 'if (evidenceId) return "evidence";' in body
        assert 'if (noteNodeId) return "note";' in body
        assert 'const nativeRef = refFromHost(host);' in body

    def test_chip_dedupe_key_prefers_native_ids_over_chapter_frame_ids(self):
        body = Path("app/static/copilot.js").read_text(encoding="utf-8")
        assert 'if (ref.evidence_id) return "evidence|" + ref.evidence_id;' in body
        assert 'if (ref.note_node_id) return "note|" + ref.note_node_id;' in body
        assert 'const nativeKey = chipKey(nativeRef);' in body

    def test_chip_payload_keeps_note_and_evidence_native_ids(self):
        body = Path("app/static/copilot.js").read_text(encoding="utf-8")
        assert "note_node_id: c.note_node_id || null" in body
        assert "evidence_id: c.evidence_id || null" in body
        assert "const noteNodeId = ref.note_node_id || null" in body
        assert "const evidenceId = ref.evidence_id || null" in body

    def test_tool_trace_prefers_note_and_evidence_signals_before_compat_ids(self):
        body = Path("app/static/copilot.js").read_text(encoding="utf-8")
        assert 'bits.push(`主引用 讲义节点 ${payload.primary_ref_id}`);' in body
        assert 'bits.push(`主引用 证据对象 ${payload.primary_ref_id}`);' in body
        assert 'bits.push(`讲义节点 ${payload.note_node_id}`);' in body
        assert 'bits.push(`证据 ${payload.evidence_id}`);' in body
        assert 'bits.push(`兼容章节 Ch${payload.chapter_idx}`);' in body
        assert 'bits.push(`兼容关键帧 F${payload.frame_id}`);' in body


# ---------------------------------------------------------------------------
# Stage 8 — Domain tree (index page grouping + inline taxonomy editor)
# ---------------------------------------------------------------------------


def _summary_row(
    *,
    bv: str,
    domain: str | None = None,
    direction: str | None = None,
    tags: list[str] | None = None,
    created_at: str = "2026-05-01T00:00:00Z",
    status: str = "done",
):
    """Build a minimal ``SummaryRow`` for grouping-logic tests.

    We go through the dataclass directly instead of round-tripping via
    the DB so these tests stay pure and fast (<1ms each).
    """
    from app.storage.db import SummaryRow

    return SummaryRow(
        bv_id=bv,
        url=f"https://www.bilibili.com/video/{bv}",
        title=f"Title {bv}",
        author="author",
        duration=100,
        cover_url="",
        category="lecture",
        domain_tags=list(tags or []),
        summary_json={},
        report_path=f"reports/{bv}.html",
        model_used="test",
        token_cost=0,
        status=status,
        error_msg=None,
        created_at=created_at,
        updated_at=created_at,
        domain=domain,
        direction=direction,
    )


class TestGroupSummariesByDomain:
    def test_empty_input_returns_empty_list(self):
        from app.render.renderer import group_summaries_by_domain

        assert group_summaries_by_domain([]) == []

    def test_single_fully_tagged_row_forms_one_domain_group(self):
        from app.render.renderer import group_summaries_by_domain

        row = _summary_row(bv="BV_A", domain="AI 技术", direction="注意力机制")
        groups = group_summaries_by_domain([row])
        assert len(groups) == 1
        g = groups[0]
        assert g["domain"] == "AI 技术"
        assert g["count"] == 1
        assert g["needs_review"] is False
        assert [d["direction"] for d in g["directions"]] == ["注意力机制"]
        assert g["directions"][0]["cards"][0].bv_id == "BV_A"

    def test_missing_domain_or_direction_lands_in_pending_bucket(self):
        from app.render.renderer import group_summaries_by_domain

        rows = [
            _summary_row(bv="BV_NO_DOMAIN", domain=None, direction="X"),
            _summary_row(bv="BV_OTHER", domain="其他", direction="X"),
            _summary_row(bv="BV_NO_DIR", domain="AI 技术", direction=""),
            _summary_row(bv="BV_OK", domain="AI 技术", direction="Y"),
        ]
        groups = group_summaries_by_domain(rows)
        assert groups[-1]["domain"] == "待归类"
        assert groups[-1]["needs_review"] is True
        bvs_in_pending = {
            item.bv_id for d in groups[-1]["directions"] for item in d["cards"]
        }
        assert bvs_in_pending == {"BV_NO_DOMAIN", "BV_OTHER", "BV_NO_DIR"}

        # The one well-tagged row stays in its proper domain bucket.
        ai_group = next(g for g in groups if g["domain"] == "AI 技术")
        assert ai_group["count"] == 1
        assert ai_group["needs_review"] is False

    def test_domains_sorted_by_count_desc_then_name(self):
        from app.render.renderer import group_summaries_by_domain

        rows = [
            _summary_row(bv="A1", domain="AI 技术", direction="X"),
            _summary_row(bv="A2", domain="AI 技术", direction="X"),
            _summary_row(bv="A3", domain="AI 技术", direction="Y"),
            _summary_row(bv="P1", domain="物理", direction="力学"),
            _summary_row(bv="M1", domain="数学", direction="代数"),
        ]
        groups = group_summaries_by_domain(rows)
        # No pending → no "待归类".
        assert all(g["domain"] != "待归类" for g in groups)
        # Count order: AI(3) > 数学(1) ~ 物理(1); tiebreak by domain ASC.
        assert [g["domain"] for g in groups] == ["AI 技术", "数学", "物理"]

    def test_directions_sorted_by_count_desc_and_items_by_created_at_desc(self):
        from app.render.renderer import group_summaries_by_domain

        rows = [
            _summary_row(bv="B1", domain="AI 技术", direction="注意力", created_at="2026-05-01"),
            _summary_row(bv="B2", domain="AI 技术", direction="注意力", created_at="2026-05-03"),
            _summary_row(bv="B3", domain="AI 技术", direction="注意力", created_at="2026-05-02"),
            _summary_row(bv="B4", domain="AI 技术", direction="RLHF",  created_at="2026-05-10"),
        ]
        g = group_summaries_by_domain(rows)[0]
        assert [d["direction"] for d in g["directions"]] == ["注意力", "RLHF"]
        first_dir_items = g["directions"][0]["cards"]
        assert [r.bv_id for r in first_dir_items] == ["B2", "B3", "B1"]

    def test_non_done_rows_are_skipped(self):
        from app.render.renderer import group_summaries_by_domain

        rows = [
            _summary_row(bv="OK", domain="数学", direction="代数", status="done"),
            _summary_row(bv="FAIL", domain="数学", direction="代数", status="failed"),
            _summary_row(bv="RUN", domain="数学", direction="代数", status="running"),
        ]
        groups = group_summaries_by_domain(rows)
        assert len(groups) == 1
        assert groups[0]["count"] == 1
        assert groups[0]["directions"][0]["cards"][0].bv_id == "OK"


class TestIndexPageRendersTree:
    def test_render_index_emits_domain_groups_and_editor(self, tmp_path, monkeypatch):
        """``Renderer.render_index`` must embed the tree + editor <dialog>."""
        from app.config import get_settings
        from app.render.renderer import Renderer

        # Isolate reports_dir in case render_index ever touches it (it
        # doesn't today, but Stage 9 might).
        settings = get_settings()
        monkeypatch.setattr(settings, "data_dir", tmp_path)

        rows = [
            _summary_row(bv="BV_OK", domain="AI 技术", direction="注意力机制", tags=["Transformer"]),
            _summary_row(bv="BV_D2", domain="AI 技术", direction="计算机视觉"),
            _summary_row(bv="BV_D3", domain="AI 技术", direction="多模态学习"),
            _summary_row(bv="BV_D4", domain="AI 技术", direction="智能体框架"),
            _summary_row(bv="BV_D5", domain="AI 技术", direction="向量检索"),
            _summary_row(bv="BV_PEND", domain=None, direction=None),
        ]
        html = Renderer().render_index(rows, css_inline="")

        # Tree shell + both buckets present.
        assert 'class="domain-tree"' in html
        assert 'data-testid="domain-tree"' in html
        assert 'id="library-controls"' in html
        assert 'id="library-search"' in html
        assert 'id="library-filter-domain"' in html
        assert 'id="library-filter-direction"' in html
        assert 'id="library-filter-status"' in html
        assert 'id="library-result-count"' in html
        assert "refreshDirectionOptions" in html
        assert "applyFilters" in html
        assert "待归类" in html
        assert "AI 技术" in html

        # "待归类" bucket flags needs_review visually.
        assert 'class="domain-group needs-review"' in html
        assert "待手动归类" in html  # badge label

        # Card carries the dataset attributes the editor reads.
        assert 'data-bv="BV_OK"' in html
        assert 'data-domain="AI 技术"' in html
        assert 'data-direction="注意力机制"' in html
        assert 'data-tags="Transformer"' in html

        # Per-card edit button + inline <dialog>.
        assert 'class="card-edit-btn"' in html
        assert 'id="tax-editor"' in html
        assert 'id="tax-domain"' in html
        assert 'id="tax-direction"' in html
        assert 'id="tax-direction-list"' in html
        assert 'id="tax-direction-governance"' in html
        assert "directionCountsByDomain" in html
        assert "updateDirectionSuggestions" in html
        assert 'class="direction-fragment-hint"' in html
        assert "当前领域已有 5 个方向" in html

        # Domain whitelist rendered as <option>s (spot-check a few).
        assert '<option value="AI 技术"' in html
        assert '<option value="其他"' in html

    def test_render_index_uses_local_frame_cover_fallback(self, tmp_path, monkeypatch):
        from app.config import get_settings
        from app.render.renderer import Renderer

        settings = get_settings()
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        row = _summary_row(bv="BV_COVER", domain="AI 技术", direction="视觉智能")
        row.summary_json = {
            "chapters": [
                {"frames": [{"path": str(tmp_path / "keyframes" / "BV_COVER" / "00001.jpg")}]}
            ]
        }

        html = Renderer().render_index([row], css_inline="")

        assert "/keyframes/BV_COVER/00001.jpg" in html


class TestIndexPageRoute:
    """End-to-end: GET / renders the tree with data from the DB."""

    async def test_root_route_groups_existing_summaries(self, tmp_path, monkeypatch):
        from app.config import get_settings
        from app.render.renderer import Renderer

        settings = get_settings()
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

        # Two rows: one well-tagged, one "needs_review".
        lec_ok = _make_lecture("BV_IDX1")
        lec_pending = _make_lecture("BV_IDX2")
        ctx, rag = await _build_ctx(
            tmp_path,
            [lec_ok, lec_pending],
            rows_meta={
                "BV_IDX1": {"domain": "AI 技术", "direction": "注意力机制"},
                # BV_IDX2 left unset → pending bucket.
            },
        )
        try:
            from app import api as app_api

            app = FastAPI()
            app.state.db = ctx.db
            app.state.rag = ctx.rag
            app.state.renderer = Renderer()
            app.include_router(app_api.router)
            with TestClient(app) as client:
                resp = client.get("/", headers=_basic_headers())
            assert resp.status_code == 200, resp.text
            html = resp.text
            assert "data-testid=\"domain-tree\"" in html
            assert "BV_IDX1" in html
            assert "BV_IDX2" in html
            # The well-tagged BV lands under "AI 技术"; the untagged one
            # under "待归类". Both substrings must coexist.
            assert "AI 技术" in html
            assert "待归类" in html
        finally:
            await rag.close()


class TestTaxonomyEditorPatchIntegration:
    """Simulate the dialog submitting to ``/api/copilot/taxonomy``."""

    async def test_editor_patch_moves_row_between_groups(self, tmp_path):
        from app.render.renderer import group_summaries_by_domain

        lec = _make_lecture("BV_EDIT")
        ctx, rag = await _build_ctx(tmp_path, [lec])
        try:
            # Before: domain/direction are NULL → lands in "待归类".
            rows_before = await ctx.db.list_summaries(limit=50, offset=0)
            groups_before = group_summaries_by_domain(rows_before)
            assert groups_before and groups_before[0]["domain"] == "待归类"

            app = _build_app_with_state(ctx)
            with TestClient(app) as client:
                resp = client.post(
                    "/api/copilot/taxonomy",
                    headers=_basic_headers(),
                    json={
                        "bv": "BV_EDIT",
                        "domain": "AI 技术",
                        "direction": "注意力机制",
                        "tags": ["Transformer", "Attention"],
                    },
                )
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

            # After: row must move to the "AI 技术" bucket and leave the
            # pending one empty (i.e. dropped from the groups list).
            rows_after = await ctx.db.list_summaries(limit=50, offset=0)
            groups_after = group_summaries_by_domain(rows_after)
            assert all(g["domain"] != "待归类" for g in groups_after)
            ai = next(g for g in groups_after if g["domain"] == "AI 技术")
            assert ai["count"] == 1
            assert ai["directions"][0]["cards"][0].bv_id == "BV_EDIT"
            # Persisted tags round-tripped through the PATCH.
            assert ai["directions"][0]["cards"][0].domain_tags == [
                "Transformer",
                "Attention",
            ]
        finally:
            await rag.close()
