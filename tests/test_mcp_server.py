"""MCP server smoke tests — in-process ``Client`` against ``build_mcp``.

We reuse the synthetic ``LectureJSON`` builder from ``test_copilot`` so
the MCP tests exercise real Pydantic models end-to-end without
depending on the on-disk DB.  Each test seeds a throwaway DB in
``tmp_path``, wraps it in a ``ToolContext``, and drives the fastmcp
client.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from app.copilot.mcp_server import build_mcp
from app.copilot.rag import RAGStore
from app.copilot.tools import ToolContext
from app.storage.db import Database
from tests.test_copilot import (  # noqa: E402  reuse helpers
    _attach_note_and_evidence,
    _lecture_with_taxonomy,
    _make_lecture,
)


async def _seed_ctx(tmp_path: Path, lectures, rows_meta=None) -> tuple[ToolContext, RAGStore]:
    db_path = tmp_path / "app.db"
    db = Database(db_path)
    await db.init()

    rag = RAGStore(db_path)
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


class TestMcpServerShape:
    async def test_default_tool_listing_has_note_and_evidence_tools(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_make_lecture("BV_MCP1")])
        try:
            mcp = build_mcp(ctx, expose_summarize=False)
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = sorted(t.name for t in tools)
            assert names == sorted(
                [
                    "search_evidence",
                    "search_lectures",
                    "get_note_unit",
                    "get_evidence_object",
                    "get_chapter",
                    "get_frame",
                    "get_quote_context",
                    "explain_frame",
                    "get_knowledge_units",
                    "get_frame_description",
                    "list_lectures",
                ]
            )
        finally:
            await rag.close()

    async def test_summarize_video_hidden_by_default(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_make_lecture("BV_MCP2")])
        try:
            mcp = build_mcp(ctx, expose_summarize=False)
            async with Client(mcp) as client:
                tools = await client.list_tools()
            assert "summarize_video" not in {t.name for t in tools}
        finally:
            await rag.close()

    async def test_summarize_video_registered_when_enabled(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_make_lecture("BV_MCP3")])
        try:
            mcp = build_mcp(ctx, expose_summarize=True)
            async with Client(mcp) as client:
                tools = await client.list_tools()
            assert "summarize_video" in {t.name for t in tools}
        finally:
            await rag.close()


class TestMcpToolCalls:
    async def test_list_lectures_returns_metadata(self, tmp_path):
        ctx, rag = await _seed_ctx(
            tmp_path,
            [_make_lecture("BV_L1"), _make_lecture("BV_L2")],
            rows_meta={
                "BV_L1": {"domain": "AI 技术", "direction": "注意力机制"},
                "BV_L2": {"domain": "健身运动", "direction": "脂肪燃烧"},
            },
        )
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool("list_lectures", {"limit": 50})
            assert result.is_error is False
            data = result.data
            lectures = data["lectures"] if isinstance(data, dict) else data
            bvs = {l["bv_id"]: l for l in lectures}
            assert {"BV_L1", "BV_L2"} <= set(bvs)
            assert bvs["BV_L1"]["domain"] == "AI 技术"
        finally:
            await rag.close()

    async def test_get_note_unit_returns_note_payload(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_attach_note_and_evidence(_make_lecture("BV_NU"))])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_note_unit", {"bv": "BV_NU", "unit_id": "unit-1"}
                )
            assert result.is_error is False
            assert result.data["unit"]["unit_id"] == "unit-1"
            assert "ev-frame-1" in result.data["related_evidence_ids"]
        finally:
            await rag.close()

    async def test_get_evidence_object_returns_evidence_payload(self, tmp_path):
        ctx, rag = await _seed_ctx(
            tmp_path, [_attach_note_and_evidence(_make_lecture("BV_EO"))]
        )
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_evidence_object", {"bv": "BV_EO", "evidence_id": "ev-quote-1"}
                )
            assert result.is_error is False
            assert result.data["evidence"]["evidence_id"] == "ev-quote-1"
            assert result.data["relations"]
        finally:
            await rag.close()

    async def test_search_evidence_returns_hits(self, tmp_path):
        ctx, rag = await _seed_ctx(
            tmp_path, [_attach_note_and_evidence(_make_lecture("BV_SE_MCP"))]
        )
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_evidence", {"bv": "BV_SE_MCP", "query": "归一化", "top_k": 3}
                )
            assert result.is_error is False
            assert result.data["hits"]
            assert result.data["hits"][0]["evidence_id"] == "ev-quote-1"
        finally:
            await rag.close()

    async def test_get_chapter_returns_full_payload(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_attach_note_and_evidence(_make_lecture("BV_GC"))])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_chapter", {"bv": "BV_GC", "chapter_idx": 1}
                )
            assert result.is_error is False
            data = result.data
            assert data["chapter_idx"] == 1
            assert data["title"] == "第一章"
            assert len(data["frames"]) >= 1
            assert data["compatibility_anchor_kind"] == "chapter"
            assert data["compatibility_anchor_id"] == "Ch1"
            assert data["primary_ref_kind"] == ""
            assert data["primary_ref_id"] == ""
            assert data["primary_evidence_id"] == ""
            assert data["primary_note_node_id"] == ""
            assert data["primary_note_unit_id"] == ""
            assert data["primary_note_node_type"] == ""
            assert data["primary_note_title"] == ""
            assert data["primary_evidence_kind"] == ""
        finally:
            await rag.close()

    async def test_get_chapter_unknown_bv_surfaces_error(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_make_lecture("BV_GC")])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                with pytest.raises(Exception) as ei:
                    await client.call_tool(
                        "get_chapter", {"bv": "BV_NOPE", "chapter_idx": 1}
                    )
            # ToolError("not_found", ...) → ValueError("[not_found] ...") on MCP side.
            assert "not_found" in str(ei.value).lower()
        finally:
            await rag.close()

    async def test_get_knowledge_units_kind_filter(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_make_lecture("BV_KU")])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_knowledge_units", {"bv": "BV_KU", "kind": "concept"}
                )
                assert result.is_error is False
                assert len(result.data["units"]) == 1

                result2 = await client.call_tool(
                    "get_knowledge_units", {"bv": "BV_KU", "kind": "formula"}
                )
                assert result2.is_error is False
                assert result2.data["units"] == []
        finally:
            await rag.close()

    async def test_get_frame_description_returns_frame(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_attach_note_and_evidence(_lecture_with_taxonomy("BV_FD"))])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_frame_description", {"bv": "BV_FD", "frame_id": 1}
                )
            assert result.is_error is False
            assert result.data["frame_id"] == 1
            assert result.data["path"] == "/img/ch1_diagram.jpg"
            assert result.data["compatibility_anchor_kind"] == "frame"
            assert result.data["compatibility_anchor_id"] == "F1"
            assert result.data["primary_ref_kind"] == "evidence"
            assert result.data["primary_ref_id"] == "ev-frame-1"
            assert result.data["primary_evidence_id"] == "ev-frame-1"
            assert result.data["primary_note_node_id"] == "unit-1"
            assert result.data["primary_note_unit_id"] == "unit-1"
            assert result.data["primary_note_node_type"] == "teaching_unit"
            assert result.data["primary_note_title"] == "主线单元一"
            assert result.data["primary_evidence_kind"] == "frame_evidence"
        finally:
            await rag.close()

    async def test_get_frame_returns_frame(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_attach_note_and_evidence(_lecture_with_taxonomy("BV_GF_MCP"))])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_frame", {"bv": "BV_GF_MCP", "frame_id": 1}
                )
            assert result.is_error is False
            data = result.data
            assert data["frame_id"] == 1
            assert data["compatibility_anchor_kind"] == "frame"
            assert data["compatibility_anchor_id"] == "F1"
            assert data["primary_ref_kind"] == "evidence"
            assert data["primary_ref_id"] == "ev-frame-1"
            assert data["caption"] == "架构图"
        finally:
            await rag.close()

    async def test_get_frame_description_time_only_match_keeps_canonical_empty(self, tmp_path):
        lec = _attach_note_and_evidence(_lecture_with_taxonomy("BV_FD_TIMEONLY"))
        lec.evidence_index.evidence_objects[1].path = ""
        lec.evidence_index.evidence_objects[1].ts = 45
        ctx, rag = await _seed_ctx(tmp_path, [lec])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_frame_description", {"bv": "BV_FD_TIMEONLY", "frame_id": 1}
                )
            assert result.is_error is False
            data = result.data
            assert data["compatibility_anchor_kind"] == "frame"
            assert data["compatibility_anchor_id"] == "F1"
            assert "ev-frame-1" in data["related_evidence_ids"]
            assert "unit-1" in data["related_note_node_ids"]
            assert data["primary_ref_kind"] == ""
            assert data["primary_ref_id"] == ""
            assert data["primary_evidence_id"] == ""
            assert data["primary_note_node_id"] == ""
            assert data["primary_note_unit_id"] == ""
            assert data["primary_note_node_type"] == ""
            assert data["primary_note_title"] == ""
            assert data["primary_evidence_kind"] == ""
        finally:
            await rag.close()

    async def test_get_quote_context_keeps_canonical_empty_without_explicit_quote_bridge(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_attach_note_and_evidence(_make_lecture("BV_QC_MCP"))])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_quote_context", {"bv": "BV_QC_MCP", "quote": "A 就是 X"}
                )
            assert result.is_error is False
            data = result.data
            assert data["compatibility_anchor_kind"] == "quote"
            assert data["compatibility_anchor_id"] == "A 就是 X"
            assert "ev-quote-1" in data["related_evidence_ids"]
            assert "block-1" in data["related_note_node_ids"]
            assert data["primary_ref_kind"] == ""
            assert data["primary_ref_id"] == ""
            assert data["primary_evidence_id"] == ""
            assert data["primary_note_node_id"] == ""
            assert data["primary_note_unit_id"] == ""
            assert data["primary_note_node_type"] == ""
            assert data["primary_note_title"] == ""
            assert data["primary_evidence_kind"] == ""
        finally:
            await rag.close()

    async def test_explain_frame_returns_payload(self, tmp_path):
        ctx, rag = await _seed_ctx(tmp_path, [_attach_note_and_evidence(_lecture_with_taxonomy("BV_EF_MCP"))])
        try:
            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "explain_frame", {"bv": "BV_EF_MCP", "frame_id": 1}
                )
            assert result.is_error is False
            data = result.data
            assert data["frame_id"] == 1
            assert data["compatibility_anchor_kind"] == "frame"
            assert data["compatibility_anchor_id"] == "F1"
            assert data["why_useful"]
        finally:
            await rag.close()

    async def test_search_lectures_domain_filter(self, tmp_path):
        from app.copilot.indexer import index_lecture as _index_lecture

        lec_a = _lecture_with_taxonomy("BV_SA", domain="AI 技术", direction="注意力机制")
        lec_b = _lecture_with_taxonomy("BV_SB", domain="健身运动", direction="脂肪燃烧")
        ctx, rag = await _seed_ctx(
            tmp_path,
            [lec_a, lec_b],
            rows_meta={
                "BV_SA": {"domain": "AI 技术", "direction": "注意力机制"},
                "BV_SB": {"domain": "健身运动", "direction": "脂肪燃烧"},
            },
        )
        if not rag.vec_available:
            await rag.close()
            pytest.skip("sqlite-vec unavailable")
        try:
            # Use real embedder here — with the real DashScope SDK unavailable
            # in the CI path the call will raise; guard with a stub.
            from tests.test_copilot import _StubEmbedder

            mapping = {
                "concept": [1.0, 0.0, 0.0, 0.0],
                "第一段教学讲解，介绍概念 A 的来龙去脉。": [0.95, 0.05, 0.0, 0.0],
                "第二段教学讲解，扩展到与 B 的关系。": [0.9, 0.1, 0.0, 0.0],
            }
            # Swap RAG embedder to stub so search is deterministic.
            rag._embedder = _StubEmbedder(mapping, dim=4)  # noqa: SLF001
            await _index_lecture(rag, "BV_SA", lec_a)
            await _index_lecture(rag, "BV_SB", lec_b)

            mcp = build_mcp(ctx)
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_lectures",
                    {"query": "concept", "top_k": 10, "domain": "AI 技术"},
                )
            assert result.is_error is False
            assert result.data["chunks"], "search returned no chunks"
            assert all(c["bv_id"] == "BV_SA" for c in result.data["chunks"])
        finally:
            await rag.close()


class TestMcpServerLifespan:
    async def test_owned_context_created_on_connect(self, tmp_path, monkeypatch):
        """When build_mcp is called without ctx it must open its own DB+RAG."""
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

        mcp = build_mcp(ctx=None, expose_summarize=False)
        async with Client(mcp) as client:
            tools = await client.list_tools()
            # list_lectures returns zero rows from the fresh DB but must not error.
            result = await client.call_tool("list_lectures", {"limit": 5})
            assert result.is_error is False
            assert result.data["lectures"] == []
        assert {t.name for t in tools} == {
            "search_evidence",
            "search_lectures",
            "get_note_unit",
            "get_evidence_object",
            "get_chapter",
            "get_frame",
            "get_quote_context",
            "explain_frame",
            "get_knowledge_units",
            "get_frame_description",
            "list_lectures",
        }
