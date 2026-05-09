"""Lightweight multi-agent helpers for the lecture pipeline.

Two callable agents share a single ``AsyncOpenAI`` client (DashScope
OpenAI-compatible mode) and a small JSON-extraction helper:

* :class:`StudyQuestionAgent` — pre-generates 5-10 study questions from
  the title, first few minutes of subtitle and a handful of frames.
  These questions are then echoed into the IR builder prompt to drive
  problem-first extraction (plan §10 N4).
* :class:`CriticReviserAgent` — runs a Critic + (optional) Reviser pass
  on a draft :class:`LectureIR`, looking for missing coverage / dubious
  quotes / un-extracted code/formula evidence.  Plan §9 M1.

Both agents are deliberately stateless and side-effect free; the
pipeline owns all I/O.  Failures degrade to **no-op** so the rest of
the pipeline never blocks on a flaky agent.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APITimeoutError, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ._latex_repair import repair_obj as _repair_latex_escapes
from .lecturize import LecturizeContext, _format_frames, _format_segments
from .length_adapt import LengthBudget, study_questions_target
from .prompts import (
    LECTURE_CRITIC_SYSTEM,
    LECTURE_CRITIC_USER_TEMPLATE,
    LECTURE_REVISER_SYSTEM,
    LECTURE_REVISER_USER_TEMPLATE,
    STUDY_QUESTIONS_SYSTEM,
    STUDY_QUESTIONS_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


_HEAD_MINUTES = 5
_MAX_HEAD_FRAMES = 6
_MAX_REVISER_SEGMENTS = 180
_MAX_REVISER_FRAMES = 24
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """Robust JSON extractor shared between the two agents."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJ_RE.search(text)
        if not match:
            raise ValueError("no JSON object found in response")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return _repair_latex_escapes(obj)


def _format_subtitle_head(ctx: LecturizeContext, minutes: int = _HEAD_MINUTES) -> str:
    cutoff = max(60.0, minutes * 60.0)
    head = [s for s in ctx.segments if s.start <= cutoff]
    if not head:
        head = ctx.segments[: min(40, len(ctx.segments))]
    return _format_segments(head)


def _format_frames_head(ctx: LecturizeContext, limit: int = _MAX_HEAD_FRAMES) -> str:
    head = ctx.frame_descs[:limit] if ctx.frame_descs else []
    return _format_frames(head)


def _domain_hint(ctx: LecturizeContext) -> str:
    """Cheap zero-LLM domain hint based on the title.

    Used to nudge the question-generator toward the right specifics
    (code/formula vs steps vs concepts) without spending another LLM
    round trip.
    """
    title = (ctx.title or "").lower()
    if any(k in title for k in ["代码", "代码实现", "python", "rust", "go ", "c++", "实现", "源码", "transformer", "attention", "算法"]):
        return "技术/编程类，请重点提问 API、复杂度、实现细节"
    if any(k in title for k in ["教程", "步骤", "操作", "实操", "做法", "怎么做", "搭建", "配置"]):
        return "操作/教程类，请重点提问步骤、检查点、常见失败"
    if any(k in title for k in ["公式", "推导", "证明", "数学", "物理", "化学"]):
        return "数理/公式类，请重点提问推导、参数、边界条件"
    if any(k in title for k in ["讲座", "认知", "观点", "方法论", "为什么", "理解", "思考"]):
        return "概念/思考类，请重点提问动机、反例、与已知方法的差异"
    return "通用讲解类，请按视频主题提问"


# ---------------------------------------------------------------------------
# Study questions (Question-Driven extraction)
# ---------------------------------------------------------------------------


@dataclass
class StudyQuestionsResult:
    questions: list[str]
    usage: dict[str, Any]


class StudyQuestionAgent:
    """Generate the 5-10 problem-first study questions for a video."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        s = get_settings()
        self._settings = s
        self._client = client or AsyncOpenAI(
            api_key=s.dashscope_api_key,
            base_url=s.dashscope_base_url,
            timeout=s.dashscope_request_timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=s.dashscope_request_timeout,
                trust_env=s.dashscope_trust_env,
            ),
        )
        # Use the same text model as the IR builder; questions are cheap
        # but accuracy matters more than latency.
        self._model = s.qwen_text_model
        self._timeout = s.dashscope_request_timeout
        self._strict = bool(s.lecture_strict_agents)
        self._enable_thinking = s.qwen_text_enable_thinking

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=1, max=8), reraise=True)
    async def generate(self, ctx: LecturizeContext) -> StudyQuestionsResult:
        q_min, q_max = study_questions_target(ctx.duration)
        user = STUDY_QUESTIONS_USER_TEMPLATE.format(
            bv_id=ctx.bv_id,
            title=ctx.title,
            author=ctx.author or "(未知)",
            duration=ctx.duration,
            minutes=max(1, int(ctx.duration // 60)),
            domain_hint=_domain_hint(ctx),
            q_min=q_min,
            q_max=q_max,
            head_minutes=_HEAD_MINUTES,
            subtitle_head=_format_subtitle_head(ctx) or "(无字幕节选)",
            frames_head=_format_frames_head(ctx) or "(无关键帧)",
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": STUDY_QUESTIONS_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
                timeout=self._timeout,
                extra_body={"enable_thinking": self._enable_thinking},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Study questions LLM call failed: %s", exc)
            if self._strict:
                raise
            return StudyQuestionsResult(questions=[], usage={})

        content = resp.choices[0].message.content or ""
        try:
            obj = _extract_json(content)
        except ValueError as exc:
            logger.warning("Study questions returned non-JSON: %s", exc)
            if self._strict:
                raise
            return StudyQuestionsResult(questions=[], usage=_usage(resp))

        raw_qs = obj.get("study_questions") or obj.get("questions") or []
        if not isinstance(raw_qs, list):
            return StudyQuestionsResult(questions=[], usage=_usage(resp))
        questions: list[str] = []
        seen: set[str] = set()
        for q in raw_qs:
            if isinstance(q, dict):
                q = q.get("question") or q.get("text") or ""
            text = str(q).strip()
            if not text:
                continue
            key = re.sub(r"\s+", "", text)
            if key in seen:
                continue
            seen.add(key)
            if not text.endswith(("?", "？")):
                text += "？"
            questions.append(text)
            if len(questions) >= q_max:
                break
        # Floor the count: even with q_min if the model produces fewer
        # we still pass through because forcing repetition would dilute
        # quality.
        return StudyQuestionsResult(questions=questions, usage=_usage(resp))


# ---------------------------------------------------------------------------
# Critic + Reviser
# ---------------------------------------------------------------------------


@dataclass
class CritiqueIssue:
    kind: str
    severity: str
    location: str
    evidence: str
    suggestion: str


@dataclass
class CritiqueResult:
    verdict: str  # ok | needs_revision
    summary: str
    issues: list[CritiqueIssue]
    usage: dict[str, Any]


class CriticReviserAgent:
    """Run the critic-reviser loop on a draft :class:`LectureIR` JSON."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        s = get_settings()
        self._settings = s
        self._client = client or AsyncOpenAI(
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
        self._critic_timeout = s.lecture_critic_timeout
        self._reviser_timeout = s.lecture_reviser_timeout
        self._strict = bool(s.lecture_strict_agents)
        self._enable_thinking = s.qwen_text_enable_thinking

    async def critique(
        self, ctx: LecturizeContext, ir_json: dict[str, Any], study_questions: list[str]
    ) -> CritiqueResult:
        user = LECTURE_CRITIC_USER_TEMPLATE.format(
            bv_id=ctx.bv_id,
            title=ctx.title,
            duration=ctx.duration,
            study_questions="\n".join(f"- {q}" for q in study_questions) or "(无)",
            lecture_ir_json=json.dumps(ir_json, ensure_ascii=False, indent=2),
            subtitle_block=_format_segments(ctx.segments) or "(无)",
            frames_block=_format_frames(ctx.frame_descs) or "(无)",
        )
        async def _do_call() -> Any:
            return await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": LECTURE_CRITIC_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=self._critic_timeout,
                extra_body={"enable_thinking": self._enable_thinking},
            )

        try:
            resp = await _do_call()
        except APITimeoutError as exc:
            # The Critic request is the heaviest in the pipeline (full draft
            # IR + subtitle + frames). Dashscope occasionally hangs on a
            # large prompt; one retry resolves the bulk of these in practice.
            logger.warning("Critic LLM call timed out (%s); retrying once", exc)
            try:
                resp = await _do_call()
            except Exception as exc2:  # noqa: BLE001
                logger.warning("Critic LLM retry also failed: %s", exc2)
                if self._strict:
                    raise
                return CritiqueResult(verdict="ok", summary="(critic skipped)", issues=[], usage={})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Critic LLM call failed: %s", exc)
            if self._strict:
                raise
            return CritiqueResult(verdict="ok", summary="(critic skipped)", issues=[], usage={})

        content = resp.choices[0].message.content or ""
        try:
            obj = _extract_json(content)
        except ValueError as exc:
            logger.warning("Critic returned non-JSON: %s", exc)
            if self._strict:
                raise
            return CritiqueResult(verdict="ok", summary="(critic non-json)", issues=[], usage=_usage(resp))

        raw_issues = obj.get("issues") or []
        issues: list[CritiqueIssue] = []
        if isinstance(raw_issues, list):
            for it in raw_issues:
                if not isinstance(it, dict):
                    continue
                issues.append(
                    CritiqueIssue(
                        kind=str(it.get("kind") or "other").strip(),
                        severity=str(it.get("severity") or "medium").strip(),
                        location=str(it.get("location") or "").strip(),
                        evidence=str(it.get("evidence") or "").strip(),
                        suggestion=str(it.get("suggestion") or "").strip(),
                    )
                )
        verdict = str(obj.get("verdict") or "").strip().lower()
        if verdict not in {"ok", "needs_revision"}:
            verdict = "needs_revision" if issues else "ok"
        summary = str(obj.get("summary") or "").strip()
        return CritiqueResult(verdict=verdict, summary=summary, issues=issues, usage=_usage(resp))

    async def revise(
        self,
        ctx: LecturizeContext,
        ir_json: dict[str, Any],
        issues: list[CritiqueIssue],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Apply critic feedback. Returns ``(revised_ir_json or None, usage)``."""
        if not issues:
            return None, {}
        user = LECTURE_REVISER_USER_TEMPLATE.format(
            lecture_ir_json=json.dumps(ir_json, ensure_ascii=False, indent=2),
            issues_json=json.dumps(
                [
                    {
                        "kind": i.kind,
                        "severity": i.severity,
                        "location": i.location,
                        "evidence": i.evidence,
                        "suggestion": i.suggestion,
                    }
                    for i in issues
                ],
                ensure_ascii=False,
                indent=2,
            ),
            subtitle_block=_format_reviser_segments(ctx, issues) or "(?)",
            frames_block=_format_reviser_frames(ctx, issues) or "(?)",
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": LECTURE_REVISER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=self._reviser_timeout,
                extra_body={"enable_thinking": self._enable_thinking},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reviser LLM call failed: %s", exc)
            if self._strict:
                raise
            return None, {}

        content = resp.choices[0].message.content or ""
        try:
            obj = _extract_json(content)
        except ValueError as exc:
            logger.warning("Reviser returned non-JSON: %s", exc)
            if self._strict:
                raise
            return None, _usage(resp)
        return obj, _usage(resp)


def _issue_timestamps(issues: list[CritiqueIssue]) -> list[float]:
    timestamps: list[float] = []
    for issue in issues:
        text = " ".join([issue.location, issue.evidence, issue.suggestion])
        for match in re.finditer(r"(?:t=|ts=|@)?(?:(\d{1,2}):)?(\d{1,2}):(\d{2})|(\d+(?:\.\d+)?)\s*(?:s|秒)", text):
            if match.group(4):
                timestamps.append(float(match.group(4)))
                continue
            hours_or_minutes = int(match.group(1) or 0)
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            timestamps.append(hours_or_minutes * 3600 + minutes * 60 + seconds)
    return timestamps


def _format_reviser_segments(ctx: LecturizeContext, issues: list[CritiqueIssue]) -> str:
    if not ctx.segments:
        return ""
    anchors = _issue_timestamps(issues)
    if not anchors:
        return _format_segments(ctx.segments[:_MAX_REVISER_SEGMENTS])
    selected = []
    seen: set[int] = set()
    for anchor in anchors:
        for idx, seg in enumerate(ctx.segments):
            if idx in seen:
                continue
            if abs(float(seg.start) - anchor) <= 90 or (float(seg.start) <= anchor <= float(seg.end)):
                selected.append(seg)
                seen.add(idx)
            if len(selected) >= _MAX_REVISER_SEGMENTS:
                break
        if len(selected) >= _MAX_REVISER_SEGMENTS:
            break
    if not selected:
        selected = ctx.segments[:_MAX_REVISER_SEGMENTS]
    selected.sort(key=lambda s: s.start)
    return _format_segments(selected[:_MAX_REVISER_SEGMENTS])


def _format_reviser_frames(ctx: LecturizeContext, issues: list[CritiqueIssue]) -> str:
    if not ctx.frame_descs:
        return ""
    anchors = _issue_timestamps(issues)
    frames = ctx.frame_descs
    if anchors:
        frames = sorted(
            frames,
            key=lambda f: min(abs(float(f.timestamp) - anchor) for anchor in anchors),
        )
    else:
        frames = sorted(
            frames,
            key=lambda f: (
                -float(getattr(f, "importance_score", 0.0) or 0.0),
                float(getattr(f, "timestamp", 0.0) or 0.0),
            ),
        )
    return _format_frames(sorted(frames[:_MAX_REVISER_FRAMES], key=lambda f: f.timestamp))


def _usage(resp: Any) -> dict[str, Any]:
    if not getattr(resp, "usage", None):
        return {}
    u = resp.usage
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


__all__ = [
    "StudyQuestionAgent",
    "StudyQuestionsResult",
    "CriticReviserAgent",
    "CritiqueIssue",
    "CritiqueResult",
    "LengthBudget",
]
