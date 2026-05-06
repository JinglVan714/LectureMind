from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from .ir import LectureIR, hydrate_lecture_ir_data
from .lecturize import LecturizeContext, _format_frames, _format_segments
from .prompts import LECTURE_IR_SYSTEM, LECTURE_IR_USER_TEMPLATE
from .schema import LectureJSON

logger = logging.getLogger(__name__)
_ONE_LINER_MAX_CHARS = 96
_ONE_LINER_MIN_CUT_CHARS = 40
_TERMINAL_PUNCTUATION = "。！？!?"
_MAINLINE_MAX_ITEMS = 6
_MAINLINE_LONG_VIDEO_MAX_ITEMS = 8
_MAINLINE_LONG_VIDEO_SECONDS = 1800
_MIN_TEACHING_NOTE_CHARS = 24
_GENERIC_MAINLINE_KEYS = {
    "介绍相关背景",
    "介绍背景",
    "讲解核心知识",
    "讲解核心内容",
    "介绍核心内容",
    "总结全文内容",
    "总结全文",
    "总结内容",
}


class LectureIRBuilder:
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            api_key=s.dashscope_api_key,
            base_url=s.dashscope_base_url,
            timeout=s.dashscope_request_timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=s.dashscope_request_timeout,
                trust_env=s.dashscope_trust_env,
            ),
        )
        self._model = s.qwen_text_model
        self._timeout = s.dashscope_request_timeout
        self._enable_thinking = s.qwen_text_enable_thinking

    async def build(self, ctx: LecturizeContext) -> tuple[LectureIR, dict[str, Any]]:
        subtitle_block = _format_segments(ctx.segments)
        frames_block = _format_frames(ctx.frame_descs)
        user = LECTURE_IR_USER_TEMPLATE.format(
            bv_id=ctx.bv_id,
            title=ctx.title,
            author=ctx.author,
            duration=ctx.duration,
            cover_url=ctx.cover_url,
            subtitle_block=subtitle_block or "(无)",
            frames_block=frames_block or "(无)",
        )
        raw, usage = await self._call(user)
        ir = self._parse_and_validate(raw, ctx)
        return ir, {"tokens": usage, "model": self._model, "stage": "lecture_ir"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=15), reraise=True)
    async def _call(self, user_msg: str) -> tuple[str, dict[str, Any]]:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": LECTURE_IR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.25,
            response_format={"type": "json_object"},
            timeout=self._timeout,
            extra_body={"enable_thinking": self._enable_thinking},
        )
        content = resp.choices[0].message.content or ""
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return content, usage

    def _parse_and_validate(self, raw: str, ctx: LecturizeContext) -> LectureIR:
        try:
            data = _extract_json(raw)
        except ValueError as e:
            self._dump_raw(ctx.bv_id, raw, reason="ir-non-json")
            raise RuntimeError(f"LLM returned non-JSON LectureIR: {e}") from e
        hydrate_lecture_ir_data(data, ctx)
        try:
            return LectureIR.model_validate(data)
        except ValidationError as e:
            debug_path = self._dump_raw(ctx.bv_id, raw, reason="ir-schema-invalid")
            logger.error("LectureIR validation failed: %s\nraw saved to %s", e, debug_path)
            raise RuntimeError(f"LectureIR schema invalid: {e}") from e

    def _dump_raw(self, bv_id: str, raw: str, *, reason: str) -> Path:
        s = get_settings()
        debug_dir = s.data_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        p = debug_dir / f"{bv_id}.lecture_ir.{reason}.txt"
        p.write_text(raw, encoding="utf-8")
        return p


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty response")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(raw)
        if not m:
            raise ValueError("no JSON object found in response")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def lecture_ir_to_lecture_json(ir: LectureIR) -> LectureJSON:
    mainline = _clean_mainline(ir.mainline, ir.duration)
    # Prefer LLM-supplied taxonomy.tags as domain_tags; fall back to the
    # legacy profile.domain_tags so v1 lectures keep working.  Pipeline
    # later normalises the taxonomy via app.copilot.taxonomy.normalize.
    domain_tags = list(ir.taxonomy.tags) if (ir.taxonomy and ir.taxonomy.tags) else ir.profile.domain_tags
    data: dict[str, Any] = {
        "bv_id": ir.bv_id,
        "url": ir.url,
        "title": ir.title,
        "author": ir.author,
        "duration": ir.duration,
        "cover_url": ir.cover,
        "one_liner": _one_liner(ir),
        "category": ir.profile.primary_type,
        "domain_tags": domain_tags,
        "learning_path": mainline,
        "chapters": [_chapter_to_json(ch) for ch in ir.chapters],
        "final_synthesis": ir.final_synthesis,
        "highlights": [],
        "glossary": _units_to_glossary(ir),
        "review_questions": ir.review_questions,
        "profile": {
            "primary_type": ir.profile.primary_type,
            "density": ir.profile.density,
            "required_sections": ir.profile.required_sections,
            "optional_sections": ir.profile.optional_sections,
        },
        "core_question": ir.core_question,
        "mainline": mainline,
        "timeline": ir.timeline.model_dump(mode="json"),
        "completeness": ir.completeness.model_dump(mode="json"),
        "render_plan": ir.render_plan.model_dump(mode="json"),
        "knowledge_units": [u.model_dump(mode="json") for u in ir.knowledge_units],
        "visual_evidence": [_visual_to_frame(v) for v in ir.visual_evidence if v.selected and v.path],
        "taxonomy": ir.taxonomy.model_dump(mode="json") if ir.taxonomy else None,
        "generation_mode": "lecture_ir_v2",
    }
    lecture = LectureJSON.model_validate(data)
    if not lecture.review_questions:
        lecture.review_questions = [f"如何用自己的话解释：{item}" for item in lecture.learning_path[:5]]
    return lecture


def _chapter_to_json(ch: Any) -> dict[str, Any]:
    return {
        "index": ch.index,
        "title": ch.title,
        "start": ch.start,
        "end": ch.end,
        "summary": ch.summary,
        "learning_goal": ch.learning_goal,
        "teaching_notes": _teaching_notes(ch),
        "process_steps": ch.process_steps,
        "points": [p.model_dump(mode="json") for p in ch.points],
        "frames": [f.model_dump(mode="json") for f in ch.frames if f.path],
        "pitfalls": ch.pitfalls,
        "key_takeaways": ch.key_takeaways,
    }


def _clean_mainline(items: list[str], duration: float) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    max_items = _MAINLINE_LONG_VIDEO_MAX_ITEMS if duration >= _MAINLINE_LONG_VIDEO_SECONDS else _MAINLINE_MAX_ITEMS
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            continue
        key = _mainline_key(text)
        if not key or key in seen or _is_generic_mainline(key):
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _mainline_key(text: str) -> str:
    return re.sub(r"[\s，。！？!?、；;：:,.《》“”\"'（）()\[\]【】\-—_]+", "", text).lower()


def _is_generic_mainline(key: str) -> bool:
    generic_keys = {_mainline_key(item) for item in _GENERIC_MAINLINE_KEYS}
    if key in generic_keys:
        return True
    return any(generic in key and len(key) <= len(generic) + 4 for generic in generic_keys)


def _teaching_notes(ch: Any) -> list[str]:
    notes = [str(note).strip() for note in ch.teaching_notes if str(note).strip()]
    if sum(len(note) for note in notes) >= _MIN_TEACHING_NOTE_CHARS:
        return notes
    fallback = _chapter_fallback_note(ch, notes)
    if fallback:
        return [fallback]
    return notes


def _chapter_fallback_note(ch: Any, notes: list[str]) -> str:
    parts = []
    seen: set[str] = set()
    for part in [ch.learning_goal, *notes, ch.summary, *ch.key_takeaways]:
        text = str(part).strip()
        key = _mainline_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return " ".join(part for part in parts if part).strip()


def _visual_to_frame(v: Any) -> dict[str, Any]:
    return {
        "ts": v.ts,
        "path": v.path,
        "caption": v.caption,
        "ocr_text": v.ocr_text,
        "insight": v.selected_reason or v.caption,
        "visual_type": v.visual_type,
        "selected_reason": v.selected_reason,
        "importance_score": v.importance_score,
    }


def _units_to_glossary(ir: LectureIR) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in ir.knowledge_units:
        title = unit.title.strip()
        if not title or title in seen:
            continue
        seen.add(title)
        out.append({"term": title[:40], "ts": unit.ts, "explanation": unit.explanation[:160]})
        if len(out) >= 15:
            break
    return out


def _one_liner(ir: LectureIR) -> str:
    if ir.core_question:
        return _semantic_one_liner(ir.core_question)
    if ir.mainline:
        return _semantic_one_liner(ir.mainline[0])
    return _semantic_one_liner(ir.title or "视频讲义")


def _semantic_one_liner(text: str) -> str:
    body = re.sub(r"\s+", " ", str(text)).strip()
    body = body.rstrip("？?。!！")
    if not body:
        return "视频讲义"
    sentence = _question_to_topic_sentence(body)
    return _semantic_trim(sentence)


def _question_to_topic_sentence(text: str) -> str:
    if text.startswith("如何理解"):
        return _finish_sentence(f"本视频讲解{text.removeprefix('如何理解').strip()}")
    if text.startswith("如何"):
        return _finish_sentence(f"本视频讲解{text.removeprefix('如何').strip()}")
    if "如何" in text:
        left, right = text.split("如何", 1)
        return _finish_sentence(f"本视频讲解{left.strip()}{right.strip()}")
    if text.startswith("为什么"):
        return _finish_sentence(f"本视频解释{text.removeprefix('为什么').strip()}")
    if "为什么" in text:
        left, right = text.split("为什么", 1)
        return _finish_sentence(f"本视频解释{left.strip()}{right.strip()}")
    if text.startswith("什么是"):
        return _finish_sentence(f"本视频说明{text.removeprefix('什么是').strip()}")
    if "是什么" in text:
        return _finish_sentence(f"本视频说明{text}")
    return _finish_sentence(text)


def _semantic_trim(text: str) -> str:
    if len(text) <= _ONE_LINER_MAX_CHARS:
        return text
    for mark in _TERMINAL_PUNCTUATION:
        pos = text.rfind(mark, 0, _ONE_LINER_MAX_CHARS + 1)
        if pos >= _ONE_LINER_MIN_CUT_CHARS:
            return text[: pos + 1]
    for token in ("，并", "，同时", "，以及", "，从而", "，但", "；", ";", "，"):
        pos = text.rfind(token, 0, _ONE_LINER_MAX_CHARS + 1)
        if pos >= _ONE_LINER_MIN_CUT_CHARS:
            return _finish_sentence(text[:pos])
    return _finish_sentence(text[:_ONE_LINER_MAX_CHARS])


def _finish_sentence(text: str) -> str:
    text = text.strip().rstrip("，,；;：:、 ")
    if not text:
        return "视频讲义"
    if text[-1] in _TERMINAL_PUNCTUATION:
        return text
    return f"{text}。"
