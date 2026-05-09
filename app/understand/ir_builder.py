from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ._latex_repair import repair_obj as _repair_latex_escapes
from .agents import (
    CriticReviserAgent,
    CritiqueResult,
    StudyQuestionAgent,
    StudyQuestionsResult,
)
from .ir import LectureIR, hydrate_lecture_ir_data
from .lecturize import LecturizeContext, _format_frames, _format_segments
from .length_adapt import LengthBudget, mainline_max
from .prompts import LECTURE_IR_SYSTEM, LECTURE_IR_USER_TEMPLATE
from .schema import LectureJSON

logger = logging.getLogger(__name__)
_ONE_LINER_MAX_CHARS = 96
_ONE_LINER_MIN_CUT_CHARS = 40
_TERMINAL_PUNCTUATION = "。！？!?"
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
        self._settings = s
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
        # Optional sub-agents share the same client to reuse the http
        # connection pool. They are constructed lazily because not every
        # caller (e.g. v1 fallback path) needs them.
        self._study_agent: StudyQuestionAgent | None = None
        self._critic_agent: CriticReviserAgent | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(self, ctx: LecturizeContext) -> tuple[LectureIR, dict[str, Any]]:
        """Single-pass IR build with the new length-aware prompt.

        Kept as the smallest possible entry-point so existing callers
        (and tests that monkey-patch the LLM call) keep working.  See
        :meth:`build_with_agents` for the question-driven + critic path.
        """
        budget = LengthBudget.for_duration(
            ctx.duration,
            keyframe_base_min=self._settings.keyframe_min,
            keyframe_base_max=self._settings.keyframe_max,
        )
        user = self._user_message(ctx, budget=budget, study_questions=[])
        t_call = time.perf_counter()
        raw, usage = await self._call(user)
        initial_ir_sec = time.perf_counter() - t_call
        t_validate = time.perf_counter()
        ir = self._parse_and_validate(raw, ctx)
        validate_sec = time.perf_counter() - t_validate
        return ir, {
            "tokens": usage,
            "model": self._model,
            "stage": "lecture_ir",
            "budget": _budget_to_dict(budget),
            "agent_timing": {
                "study_question_sec": 0.0,
                "initial_ir_sec": round(initial_ir_sec, 3),
                "critic_sec": 0.0,
                "reviser_sec": 0.0,
                "validate_sec": round(validate_sec, 3),
            },
        }

    async def build_with_agents(
        self,
        ctx: LecturizeContext,
        *,
        question_driven: bool | None = None,
        critic_enabled: bool | None = None,
    ) -> tuple[LectureIR, dict[str, Any]]:
        """Production path: question-driven extraction + critic-reviser loop.

        Order:

        1. (optional) ``StudyQuestionAgent`` — produces ``study_questions``.
        2. ``LectureIRBuilder`` (this class) — same single-pass call as
           :meth:`build` but with both the length budget and the study
           questions injected into the user message.
        3. (optional) ``CriticReviserAgent`` — runs ≤ N rounds of
           critique-and-revise; if any round leaves the schema invalid
           we silently keep the previous valid IR.
        """
        s = self._settings
        if question_driven is None:
            question_driven = bool(getattr(s, "lecture_question_driven", True))
        if critic_enabled is None:
            critic_enabled = bool(getattr(s, "lecture_critic_enabled", True))

        budget = LengthBudget.for_duration(
            ctx.duration,
            keyframe_base_min=s.keyframe_min,
            keyframe_base_max=s.keyframe_max,
        )
        usage_aggregate: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # 1. Study questions
        questions: list[str] = []
        questions_usage: dict[str, Any] = {}
        study_question_sec = 0.0
        if question_driven:
            t_q = time.perf_counter()
            try:
                if self._study_agent is None:
                    self._study_agent = StudyQuestionAgent(client=self._client)
                qres: StudyQuestionsResult = await self._study_agent.generate(ctx)
                questions = qres.questions
                questions_usage = qres.usage
                _accumulate_usage(usage_aggregate, questions_usage)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Study question generation failed (skipping): %s", exc)
            finally:
                study_question_sec = time.perf_counter() - t_q

        # 2. Initial IR build with budget + questions
        user = self._user_message(ctx, budget=budget, study_questions=questions)
        t_initial = time.perf_counter()
        raw, usage = await self._call(user)
        initial_ir_sec = time.perf_counter() - t_initial
        _accumulate_usage(usage_aggregate, usage)
        t_validate = time.perf_counter()
        ir_json = self._parse_to_dict(raw, ctx)
        # Make sure study questions survive even if the LLM ignored them.
        if questions and not ir_json.get("study_questions"):
            ir_json["study_questions"] = list(questions)
        ir = self._validate_dict(ir_json, ctx, raw)
        validate_sec = time.perf_counter() - t_validate

        critique: CritiqueResult | None = None
        revise_rounds = 0
        critic_sec = 0.0
        reviser_sec = 0.0
        if critic_enabled and questions:  # only run when we have a goal
            critique, ir, revise_rounds, critic_usage, critic_timing = await self._run_critic_loop(
                ctx, ir, questions
            )
            for u in critic_usage:
                _accumulate_usage(usage_aggregate, u)
            critic_sec = critic_timing.get("critic_sec", 0.0)
            reviser_sec = critic_timing.get("reviser_sec", 0.0)

        return ir, {
            "tokens": usage_aggregate,
            "model": self._model,
            "stage": "lecture_ir_v3",
            "budget": _budget_to_dict(budget),
            "study_questions": questions,
            "critique": _critique_to_dict(critique) if critique else None,
            "revise_rounds": revise_rounds,
            "agent_timing": {
                "study_question_sec": round(study_question_sec, 3),
                "initial_ir_sec": round(initial_ir_sec, 3),
                "critic_sec": round(critic_sec, 3),
                "reviser_sec": round(reviser_sec, 3),
                "validate_sec": round(validate_sec, 3),
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _user_message(
        self,
        ctx: LecturizeContext,
        *,
        budget: LengthBudget,
        study_questions: list[str],
    ) -> str:
        questions_block = (
            "\n".join(f"- {q}" for q in study_questions)
            if study_questions
            else "(无)"
        )
        return LECTURE_IR_USER_TEMPLATE.format(
            bv_id=ctx.bv_id,
            title=ctx.title,
            author=ctx.author,
            duration=ctx.duration,
            minutes=max(1, int(ctx.duration // 60)),
            cover_url=ctx.cover_url,
            chapters_min=budget.chapters_min,
            chapters_max=budget.chapters_max,
            mainline_max=budget.mainline_max,
            glossary_max=budget.glossary_max,
            review_questions_max=budget.review_questions_max,
            density_hint=budget.density_hint(),
            study_questions_block=questions_block,
            subtitle_block=_format_segments(ctx.segments) or "(无)",
            frames_block=_format_frames(ctx.frame_descs) or "(无)",
        )

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

    def _parse_to_dict(self, raw: str, ctx: LecturizeContext) -> dict[str, Any]:
        try:
            data = _extract_json(raw)
        except ValueError as e:
            self._dump_raw(ctx.bv_id, raw, reason="ir-non-json")
            raise RuntimeError(f"LLM returned non-JSON LectureIR: {e}") from e
        hydrate_lecture_ir_data(data, ctx)
        return data

    def _validate_dict(self, data: dict[str, Any], ctx: LecturizeContext, raw: str) -> LectureIR:
        try:
            return LectureIR.model_validate(data)
        except ValidationError as e:
            debug_path = self._dump_raw(ctx.bv_id, raw, reason="ir-schema-invalid")
            logger.error("LectureIR validation failed: %s\nraw saved to %s", e, debug_path)
            raise RuntimeError(f"LectureIR schema invalid: {e}") from e

    def _parse_and_validate(self, raw: str, ctx: LecturizeContext) -> LectureIR:
        data = self._parse_to_dict(raw, ctx)
        return self._validate_dict(data, ctx, raw)

    async def _run_critic_loop(
        self,
        ctx: LecturizeContext,
        ir: LectureIR,
        study_questions: list[str],
    ) -> tuple[CritiqueResult, LectureIR, int, list[dict[str, Any]], dict[str, float]]:
        if self._critic_agent is None:
            self._critic_agent = CriticReviserAgent()
        max_rounds = int(getattr(self._settings, "lecture_critic_max_rounds", 1) or 0)
        usages: list[dict[str, Any]] = []
        latest_ir = ir
        latest_critique: CritiqueResult | None = None
        rounds = 0
        critic_sec = 0.0
        reviser_sec = 0.0
        for _ in range(max(1, max_rounds + 1)):
            t_c = time.perf_counter()
            critique = await self._critic_agent.critique(
                ctx, latest_ir.model_dump(mode="json"), study_questions
            )
            critic_sec += time.perf_counter() - t_c
            usages.append(critique.usage)
            latest_critique = critique
            if critique.verdict != "needs_revision" or not critique.issues or rounds >= max_rounds:
                break
            t_r = time.perf_counter()
            revised, rev_usage = await self._critic_agent.revise(
                ctx, latest_ir.model_dump(mode="json"), critique.issues
            )
            reviser_sec += time.perf_counter() - t_r
            usages.append(rev_usage)
            if not revised:
                break
            try:
                hydrate_lecture_ir_data(revised, ctx)
                latest_ir = LectureIR.model_validate(revised)
            except (ValidationError, RuntimeError, ValueError) as exc:
                logger.warning("Reviser produced invalid IR; keeping previous: %s", exc)
                break
            rounds += 1
        timing = {"critic_sec": critic_sec, "reviser_sec": reviser_sec}
        return (
            latest_critique or CritiqueResult("ok", "", [], {}),
            latest_ir,
            rounds,
            usages,
            timing,
        )

    def _dump_raw(self, bv_id: str, raw: str, *, reason: str) -> Path:
        s = get_settings()
        debug_dir = s.data_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        p = debug_dir / f"{bv_id}.lecture_ir.{reason}.txt"
        p.write_text(raw, encoding="utf-8")
        return p


def _accumulate_usage(target: dict[str, int], src: dict[str, Any] | None) -> None:
    if not src:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            target[key] = int(target.get(key, 0)) + int(src.get(key, 0) or 0)
        except (TypeError, ValueError):
            pass


def _budget_to_dict(budget: LengthBudget) -> dict[str, Any]:
    return {
        "duration_sec": budget.duration_sec,
        "chapters_min": budget.chapters_min,
        "chapters_max": budget.chapters_max,
        "mainline_max": budget.mainline_max,
        "global_visual_limit": budget.global_visual_limit,
        "glossary_max": budget.glossary_max,
        "review_questions_max": budget.review_questions_max,
        "keyframe_min": budget.keyframe_min,
        "keyframe_max": budget.keyframe_max,
        "study_questions_min": budget.study_questions_min,
        "study_questions_max": budget.study_questions_max,
    }


def _critique_to_dict(c: CritiqueResult) -> dict[str, Any]:
    return {
        "verdict": c.verdict,
        "summary": c.summary,
        "issues": [
            {
                "kind": i.kind,
                "severity": i.severity,
                "location": i.location,
                "evidence": i.evidence,
                "suggestion": i.suggestion,
            }
            for i in c.issues
        ],
    }


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
    # LLMs frequently emit unescaped LaTeX backslashes inside JSON strings
    # (e.g. ``"$\beta_1$"``).  ``json.loads`` honours JSON escape rules,
    # turning ``\b`` / ``\t`` / ``\f`` / ``\r`` into control characters and
    # silently corrupting math.  Repair the parsed tree so KaTeX sees real
    # LaTeX commands again.  See ``_latex_repair`` for the rationale.
    return _repair_latex_escapes(obj)


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
        "study_questions": list(getattr(ir, "study_questions", []) or []),
        "taxonomy": ir.taxonomy.model_dump(mode="json") if ir.taxonomy else None,
        "generation_mode": "lecture_ir_v2",
    }
    # Auto-flip render_plan flags whenever we actually have content for
    # them: otherwise the LLM-supplied flag and the populated lists
    # could disagree (flag false but data present, or vice versa).
    plan = data["render_plan"] or {}
    if any(ch.get("code_blocks") for ch in data["chapters"]):
        plan["code_section"] = True
    if any(ch.get("formula_blocks") for ch in data["chapters"]):
        plan["formula_section"] = True
    data["render_plan"] = plan
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
        "code_blocks": [
            cb.model_dump(mode="json")
            for cb in getattr(ch, "code_blocks", []) or []
            if getattr(cb, "code", "").strip()
        ],
        "formula_blocks": [
            fb.model_dump(mode="json")
            for fb in getattr(ch, "formula_blocks", []) or []
            if getattr(fb, "latex", "").strip()
        ],
    }


def _clean_mainline(items: list[str], duration: float) -> list[str]:
    """Dedup, drop generic items, and cap by length-aware budget.

    Replaces the legacy two-step ladder (6 / 8) with a continuous
    duration-driven cap from :func:`length_adapt.mainline_max`, so a
    44-minute lecture gets up to ~10 mainline steps instead of being
    truncated to 8.
    """
    out: list[str] = []
    seen: set[str] = set()
    max_items = mainline_max(duration)
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
    """Project ``knowledge_units`` into the lecture-level glossary list.

    Cap is duration-adaptive (``glossary_max(duration)``) so long
    lectures aren't truncated to the legacy 15 while short videos still
    stay tidy.
    """
    from .length_adapt import glossary_max

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    cap = glossary_max(ir.duration)
    for unit in ir.knowledge_units:
        title = unit.title.strip()
        if not title or title in seen:
            continue
        seen.add(title)
        out.append({"term": title[:40], "ts": unit.ts, "explanation": unit.explanation[:160]})
        if len(out) >= cap:
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
