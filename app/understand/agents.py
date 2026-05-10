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
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import APITimeoutError, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..ingest.subtitle import SubtitleSegment
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
from .vlm import FrameDescription

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
    # Diagnostic counters set by ``CriticReviserAgent.critique`` so the
    # pipeline_stats / matrix report can attribute Critic cost to its
    # input projection. Populated on every exit path (including timeout
    # and non-JSON skip) so downstream readers do not need to special-
    # case missing keys.
    metrics: dict[str, Any] = field(default_factory=dict)


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
        self._critic_max_prompt_chars = int(
            getattr(s, "lecture_critic_max_prompt_chars", 0) or 0
        )

    async def critique(
        self, ctx: LecturizeContext, ir_json: dict[str, Any], study_questions: list[str]
    ) -> CritiqueResult:
        ir_json_str = json.dumps(ir_json, ensure_ascii=False, indent=2)
        sq_block = "\n".join(f"- {q}" for q in study_questions) or "(无)"

        max_chars = self._critic_max_prompt_chars
        if max_chars > 0:
            # Reserve room for the IR JSON, study questions block and the
            # template's own boilerplate before computing how much budget
            # we can give the subtitle block. ``+800`` is a deliberately
            # generous estimate of the LECTURE_CRITIC_USER_TEMPLATE
            # header / labels.
            overhead = len(ir_json_str) + len(sq_block) + 800
            seg_budget = max(1000, max_chars - overhead)
            crit_segments = _critic_segments(
                ctx, ir_json, max_chars=seg_budget
            )
            crit_frames = _critic_frames(ctx, ir_json)
        else:
            crit_segments = list(ctx.segments)
            crit_frames = list(ctx.frame_descs)

        user = LECTURE_CRITIC_USER_TEMPLATE.format(
            bv_id=ctx.bv_id,
            title=ctx.title,
            duration=ctx.duration,
            study_questions=sq_block,
            lecture_ir_json=ir_json_str,
            subtitle_block=_format_segments(crit_segments) or "(无)",
            frames_block=_format_frames(crit_frames) or "(无)",
        )

        metrics = {
            "critic_prompt_chars": len(user),
            "critic_segments_kept": len(crit_segments),
            "critic_segments_total": len(ctx.segments or []),
            "critic_frames_kept": len(crit_frames),
            "critic_frames_total": len(ctx.frame_descs or []),
            "critic_max_prompt_chars": max_chars,
            "critic_projection": "ir_anchored" if max_chars > 0 else "full",
        }

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
            # The Critic request is the heaviest remaining call in the
            # pipeline (full draft IR + projected subtitle + frames).
            # Dashscope occasionally hangs on a large prompt; one retry
            # resolves the bulk of these in practice.
            logger.warning("Critic LLM call timed out (%s); retrying once", exc)
            try:
                resp = await _do_call()
            except Exception as exc2:  # noqa: BLE001
                logger.warning("Critic LLM retry also failed: %s", exc2)
                if self._strict:
                    raise
                return CritiqueResult(
                    verdict="ok",
                    summary="(critic skipped)",
                    issues=[],
                    usage={},
                    metrics=metrics,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Critic LLM call failed: %s", exc)
            if self._strict:
                raise
            return CritiqueResult(
                verdict="ok",
                summary="(critic skipped)",
                issues=[],
                usage={},
                metrics=metrics,
            )

        content = resp.choices[0].message.content or ""
        try:
            obj = _extract_json(content)
        except ValueError as exc:
            logger.warning("Critic returned non-JSON: %s", exc)
            if self._strict:
                raise
            return CritiqueResult(
                verdict="ok",
                summary="(critic non-json)",
                issues=[],
                usage=_usage(resp),
                metrics=metrics,
            )

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
        return CritiqueResult(
            verdict=verdict,
            summary=summary,
            issues=issues,
            usage=_usage(resp),
            metrics=metrics,
        )

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


# ---------------------------------------------------------------------------
# Critic input projection (cost reduction)
# ---------------------------------------------------------------------------
#
# The Critic's prompt is dominated by three things: the full LectureIR JSON,
# the full subtitle, and the full set of frame descriptions. For long videos
# that easily exceeds 30k characters even before the model's own response.
# We never need every subtitle line / every frame to verify a draft IR; what
# matters is the neighborhood around every claim the IR makes (chapter
# windows, point/code/formula timestamps, knowledge unit anchors, visual
# evidence anchors). The helpers below project ``ctx.segments`` and
# ``ctx.frame_descs`` against the IR structure, then optionally downsample
# subtitles uniformly to fit a configurable character budget.
#
# Both helpers are pure and import-friendly so the unit tests can call them
# directly without instantiating an agent or hitting DashScope.


_CRITIC_POINT_WINDOW_SEC = 30.0
_CRITIC_FRAMES_LONG_VIDEO_SEC = 30 * 60
_CRITIC_FRAMES_SHORT_TARGET = 16
_CRITIC_FRAMES_LONG_TARGET = 24
_CRITIC_HIGH_VALUE_VISUAL_TYPES = {"code", "formula", "diagram"}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ir_anchor_intervals(
    ir_json: Any, *, duration: float, point_window: float = _CRITIC_POINT_WINDOW_SEC
) -> list[tuple[float, float]]:
    """Collect ``(start, end)`` intervals worth keeping subtitle lines for.

    Includes every chapter window (with a small 5% margin or 10s minimum)
    and a ``±point_window`` interval around every point/KU/code/formula/
    visual-evidence timestamp. Empty / malformed entries are silently
    skipped so a partially-hydrated IR still produces useful intervals.
    """
    if not isinstance(ir_json, dict):
        return []
    intervals: list[tuple[float, float]] = []
    duration = max(0.0, duration)

    def _push_anchor(ts: float) -> None:
        if ts <= 0:
            return
        a = max(0.0, ts - point_window)
        b = ts + point_window
        if duration > 0:
            b = min(b, duration)
        if b > a:
            intervals.append((a, b))

    for ch in ir_json.get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        start = _coerce_float(ch.get("start"))
        end = _coerce_float(ch.get("end"))
        if end > start:
            margin = max(10.0, (end - start) * 0.05)
            a = max(0.0, start - margin)
            b = end + margin
            if duration > 0:
                b = min(b, duration)
            intervals.append((a, b))
        for p in ch.get("points") or []:
            if isinstance(p, dict):
                _push_anchor(_coerce_float(p.get("ts")))
        for cb in ch.get("code_blocks") or []:
            if isinstance(cb, dict):
                _push_anchor(_coerce_float(cb.get("ts")))
        for fb in ch.get("formula_blocks") or []:
            if isinstance(fb, dict):
                _push_anchor(_coerce_float(fb.get("ts")))
    for ku in ir_json.get("knowledge_units") or []:
        if isinstance(ku, dict):
            _push_anchor(_coerce_float(ku.get("ts")))
    for ve in ir_json.get("visual_evidence") or []:
        if isinstance(ve, dict):
            _push_anchor(_coerce_float(ve.get("ts")))
    return intervals


def _ir_anchor_timestamps(ir_json: Any) -> list[float]:
    """Flat list of every IR timestamp worth probing visually.

    Used by :func:`_critic_frames` to bias frame selection toward the
    structures the IR already references (points, code/formula blocks,
    visual evidence anchors).
    """
    if not isinstance(ir_json, dict):
        return []
    out: list[float] = []
    for ch in ir_json.get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        for p in ch.get("points") or []:
            if isinstance(p, dict):
                ts = _coerce_float(p.get("ts"))
                if ts > 0:
                    out.append(ts)
        for cb in ch.get("code_blocks") or []:
            if isinstance(cb, dict):
                ts = _coerce_float(cb.get("ts"))
                if ts > 0:
                    out.append(ts)
        for fb in ch.get("formula_blocks") or []:
            if isinstance(fb, dict):
                ts = _coerce_float(fb.get("ts"))
                if ts > 0:
                    out.append(ts)
    for ku in ir_json.get("knowledge_units") or []:
        if isinstance(ku, dict):
            ts = _coerce_float(ku.get("ts"))
            if ts > 0:
                out.append(ts)
    for ve in ir_json.get("visual_evidence") or []:
        if isinstance(ve, dict):
            ts = _coerce_float(ve.get("ts"))
            if ts > 0:
                out.append(ts)
    return out


def _downsample_segments_to_char_budget(
    segments: list[SubtitleSegment], char_budget: int
) -> list[SubtitleSegment]:
    """Uniformly thin out ``segments`` until ``_format_segments`` fits.

    Picks evenly-spaced indices, never empty, deterministic for tests.
    Returns the input unchanged when already within budget or budget
    is non-positive.
    """
    if char_budget <= 0 or not segments:
        return list(segments)
    formatted = _format_segments(segments)
    if len(formatted) <= char_budget:
        return list(segments)
    n = len(segments)
    # Aim slightly under the raw ratio to absorb per-line overhead.
    ratio = (char_budget / max(1, len(formatted))) * 0.95
    target_n = max(1, int(n * ratio))
    if target_n >= n:
        return list(segments)
    step = n / target_n
    indices = sorted({min(n - 1, int(i * step)) for i in range(target_n)})
    return [segments[i] for i in indices]


def _critic_segments(
    ctx: LecturizeContext,
    ir_json: Any,
    *,
    max_chars: int = 0,
) -> list[SubtitleSegment]:
    """Project ``ctx.segments`` against the IR draft for the Critic.

    1. Build IR-anchored intervals (chapter windows + point/KU/code/
       formula/visual anchors with ``±30s`` neighborhoods).
    2. Keep every segment whose ``[start, end]`` overlaps any interval.
       Falls back to the full segment list if no anchors are available
       (e.g. a degenerate IR with no timestamps), which keeps the Critic
       useful even on the first iteration.
    3. If ``max_chars > 0`` and the formatted result still exceeds the
       budget, uniformly downsample segments by index.
    """
    segments = list(ctx.segments or [])
    if not segments:
        return []
    duration = _coerce_float(getattr(ctx, "duration", 0))
    intervals = _ir_anchor_intervals(ir_json, duration=duration)
    if intervals:
        kept: list[SubtitleSegment] = []
        for seg in segments:
            s_start = _coerce_float(getattr(seg, "start", 0))
            s_end = _coerce_float(getattr(seg, "end", s_start))
            for a, b in intervals:
                if s_end >= a and s_start <= b:
                    kept.append(seg)
                    break
        if kept:
            segments = kept
    segments = sorted(segments, key=lambda s: _coerce_float(getattr(s, "start", 0)))
    if max_chars > 0:
        segments = _downsample_segments_to_char_budget(segments, max_chars)
    return segments


def _critic_frames(
    ctx: LecturizeContext, ir_json: Any
) -> list[FrameDescription]:
    """Pick a Critic-friendly subset of ``ctx.frame_descs``.

    Always keeps every code/formula/diagram frame (high-value visual
    types the Critic must see to flag missing extraction). Then picks
    the frame nearest to each IR timestamp anchor, then fills the rest
    of the budget by ``importance_score``.

    The cap is duration-aware: long videos (>=30 min) can pack 24
    frames, shorter videos stick to 16 to keep the prompt tight.
    """
    frames = list(ctx.frame_descs or [])
    if not frames:
        return []
    duration = _coerce_float(getattr(ctx, "duration", 0))
    long_video = duration >= _CRITIC_FRAMES_LONG_VIDEO_SEC
    target = _CRITIC_FRAMES_LONG_TARGET if long_video else _CRITIC_FRAMES_SHORT_TARGET

    seen_paths: set[str] = set()
    selected: list[FrameDescription] = []

    def _key(frame: FrameDescription) -> str:
        return str(getattr(frame, "path", "") or id(frame))

    for f in frames:
        vt = (getattr(f, "visual_type", "") or "").lower()
        if vt in _CRITIC_HIGH_VALUE_VISUAL_TYPES:
            k = _key(f)
            if k not in seen_paths:
                seen_paths.add(k)
                selected.append(f)

    anchors = _ir_anchor_timestamps(ir_json)
    if anchors:
        for ts in anchors:
            nearest: FrameDescription | None = None
            best = float("inf")
            for f in frames:
                k = _key(f)
                if k in seen_paths:
                    continue
                gap = abs(_coerce_float(getattr(f, "timestamp", 0)) - ts)
                if gap < best:
                    best = gap
                    nearest = f
            if nearest is not None:
                seen_paths.add(_key(nearest))
                selected.append(nearest)
                if len(selected) >= target:
                    break

    if len(selected) < target:
        by_importance = sorted(
            frames,
            key=lambda f: -_coerce_float(getattr(f, "importance_score", 0.0)),
        )
        for f in by_importance:
            if len(selected) >= target:
                break
            k = _key(f)
            if k in seen_paths:
                continue
            seen_paths.add(k)
            selected.append(f)

    selected.sort(key=lambda f: _coerce_float(getattr(f, "timestamp", 0)))
    return selected[:target]


__all__ = [
    "StudyQuestionAgent",
    "StudyQuestionsResult",
    "CriticReviserAgent",
    "CritiqueIssue",
    "CritiqueResult",
    "LengthBudget",
]
