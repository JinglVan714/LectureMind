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


def _format_chapter_plan_block(chapter_plan: list[Any] | None) -> str:
    """Render a ChapterPlanner output for the ``{chapter_plan_block}``
    slot in :data:`LECTURE_IR_USER_TEMPLATE`.

    Empty / missing input collapses to an empty string so the prompt
    body matches the M1 layout byte-for-byte. When provided, a
    deterministic markdown block tags the suggestions explicitly as
    *hints* — the LLM is the final authority on chapter boundaries.

    The block carries a leading newline so that, when present, it
    visually separates from the preceding ``{study_questions_block}``;
    when absent, a single trailing newline keeps the surrounding
    template free of awkward double blank lines.
    """
    if not chapter_plan:
        return ""

    lines: list[str] = ["", "章节锚点（chapter_plan，由确定性规则抽取，仅作建议）："]
    for i, anchor in enumerate(chapter_plan):
        start = float(getattr(anchor, "start_sec", 0.0) or 0.0)
        end = float(getattr(anchor, "end_sec", 0.0) or 0.0)
        conf = float(getattr(anchor, "confidence", 0.0) or 0.0)
        source = str(getattr(anchor, "source", "") or "")
        text = str(getattr(anchor, "anchor_text", "") or "").strip()
        lines.append(
            f"- [{i+1}] {start:.1f}s → {end:.1f}s · "
            f"source={source} · conf={conf:.2f} · «{text[:60]}»"
        )
    lines.append(
        "可调整边界（合并/拆分/平移）以贴合实际语义；source=transition_phrase 的"
        "锚点最可信，equal_split 的最弱。"
    )
    lines.append("")
    return "\n".join(lines)


class LectureIRBuilder:
    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        self._client = AsyncOpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            timeout=s.deepseek_request_timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=s.deepseek_request_timeout,
                trust_env=s.deepseek_trust_env,
            ),
        )
        self._model = s.qwen_text_model
        self._timeout = s.deepseek_request_timeout
        self._extra_body = s.text_extra_body()
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
        ir = self._parse_and_validate(raw, ctx, finish_reason=str(usage.get("finish_reason", "")))
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
        reviser_enabled: bool | None = None,
        profile: Any = None,
        chapter_plan: list[Any] | None = None,
        chapter_cache: Any = None,
    ) -> tuple[LectureIR, dict[str, Any]]:
        """Production path: question-driven extraction + critic-reviser loop.

        Order:

        1. (optional) ``StudyQuestionAgent`` — produces ``study_questions``.
        2. IR build:

           * When ``profile.use_map_reduce`` is True (long / epic), call
             :class:`MapReduceIRBuilder` and feed it the
             ``chapter_plan`` (required) plus the same study questions
             so the global pass can synthesise a coherent mainline.
           * Otherwise (legacy / tiny / standard), do a single-pass
             :meth:`_call` with the budget, study questions and the
             optional chapter_plan rendered into the prompt.
        3. (optional) Critic + Reviser:

           * When ``profile`` is provided, dispatch through
             :meth:`_run_critic_loop_v2` (audit() + reviser_mode
             dispatch), so the per-profile patch / projected behaviour
             from P5 / P6 is honoured.
           * When ``profile`` is None, fall back to the M1-era
             :meth:`_run_critic_loop` so existing callers and tests
             keep working unchanged.
        """
        s = self._settings
        if question_driven is None:
            question_driven = bool(getattr(s, "lecture_question_driven", True))
        if critic_enabled is None:
            critic_enabled = bool(getattr(s, "lecture_critic_enabled", True))
        if reviser_enabled is None:
            reviser_enabled = bool(getattr(s, "lecture_reviser_enabled", False))

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
        study_question_warnings: list[str] = []
        study_question_metrics: dict[str, Any] = {}
        study_question_sec = 0.0
        if question_driven:
            t_q = time.perf_counter()
            try:
                if self._study_agent is None:
                    self._study_agent = StudyQuestionAgent(client=self._client)
                qres: StudyQuestionsResult = await self._study_agent.generate(ctx)
                questions = qres.questions
                questions_usage = qres.usage
                study_question_warnings = list(qres.warnings or [])
                study_question_metrics = dict(qres.metrics or {})
                _accumulate_usage(usage_aggregate, questions_usage)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Study question generation failed (skipping): %s", exc)
            finally:
                study_question_sec = time.perf_counter() - t_q

        # 2. Initial IR build — either map-reduce (long / epic) or
        # single-pass (tiny / standard / legacy). Both paths fill
        # ``ir`` and tally ``initial_ir_sec`` so downstream telemetry
        # stays comparable across profiles.
        map_reduce_stats: Any = None
        use_map_reduce = bool(profile is not None and getattr(profile, "use_map_reduce", False))
        validate_sec = 0.0
        if use_map_reduce:
            from .ir_map_reduce import MapReduceIRBuilder

            mr_builder = MapReduceIRBuilder(
                client=self._client,
                settings=s,
                profile=profile,
                chapter_cache=chapter_cache,
            )
            t_initial = time.perf_counter()
            ir, map_reduce_stats = await mr_builder.build(
                chapter_plan=list(chapter_plan or []),
                segments=ctx.segments,
                frames=ctx.frame_descs,
                meta=ctx,
                study_questions=questions,
            )
            initial_ir_sec = time.perf_counter() - t_initial
            # Map-reduce keeps its own usage telemetry separate; we do
            # not double-count here.
        else:
            user = self._user_message(
                ctx,
                budget=budget,
                study_questions=questions,
                chapter_plan=chapter_plan,
            )
            t_initial = time.perf_counter()
            raw, usage = await self._call(user)
            initial_ir_sec = time.perf_counter() - t_initial
            _accumulate_usage(usage_aggregate, usage)
            t_validate = time.perf_counter()
            ir_json = self._parse_to_dict(raw, ctx, finish_reason=str(usage.get("finish_reason", "")))
            # Make sure study questions survive even if the LLM ignored them.
            if questions and not ir_json.get("study_questions"):
                ir_json["study_questions"] = list(questions)
            ir = self._validate_dict(ir_json, ctx, raw)
            validate_sec = time.perf_counter() - t_validate

        # 3. Critic + Reviser
        critique: CritiqueResult | None = None
        revise_rounds = 0
        critic_sec = 0.0
        reviser_sec = 0.0
        reviser_telemetry: dict[str, Any] | None = None
        if profile is not None:
            critique, ir, revise_rounds, critic_usage, critic_timing = (
                await self._run_critic_loop_v2(
                    ctx, ir, questions, profile=profile
                )
            )
            for u in critic_usage:
                _accumulate_usage(usage_aggregate, u)
            critic_sec = critic_timing.get("critic_sec", 0.0)
            reviser_sec = critic_timing.get("reviser_sec", 0.0)
            reviser_telemetry = critic_timing.get("reviser")
        elif critic_enabled and questions:  # legacy path, only with study questions
            critique, ir, revise_rounds, critic_usage, critic_timing = await self._run_critic_loop(
                ctx, ir, questions, reviser_enabled=reviser_enabled
            )
            for u in critic_usage:
                _accumulate_usage(usage_aggregate, u)
            critic_sec = critic_timing.get("critic_sec", 0.0)
            reviser_sec = critic_timing.get("reviser_sec", 0.0)

        stats: dict[str, Any] = {
            "tokens": usage_aggregate,
            "model": self._model,
            "stage": "lecture_ir_v3",
            "budget": _budget_to_dict(budget),
            "study_questions": questions,
            "study_question_warnings": study_question_warnings,
            "study_question_metrics": study_question_metrics,
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
        if reviser_telemetry is not None:
            stats["reviser"] = reviser_telemetry
        if map_reduce_stats is not None:
            mr = map_reduce_stats
            stats["map_reduce"] = {
                "map_calls": getattr(mr, "map_calls", 0),
                "cache_hits": getattr(mr, "cache_hits", 0),
                "cache_writes": getattr(mr, "cache_writes", 0),
                "map_failures": list(getattr(mr, "map_failures", ()) or ()),
                "map_total_sec": round(float(getattr(mr, "map_total_sec", 0.0)), 3),
                "reduce_local_sec": round(float(getattr(mr, "reduce_local_sec", 0.0)), 3),
                "reduce_global_sec": round(float(getattr(mr, "reduce_global_sec", 0.0)), 3),
                "reduce_global_attempts": getattr(mr, "reduce_global_attempts", 0),
            }
        return ir, stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _user_message(
        self,
        ctx: LecturizeContext,
        *,
        budget: LengthBudget,
        study_questions: list[str],
        chapter_plan: list[Any] | None = None,
    ) -> str:
        """Build the user prompt for the IR builder LLM call.

        ``chapter_plan`` is an optional list of
        :class:`~app.understand.chapter_planner.ChapterAnchor` objects.
        When provided (and non-empty), they are rendered into the
        ``{chapter_plan_block}`` slot as a markdown bullet list of
        suggested chapter boundaries; the LLM is told to treat them as
        hints (not hard constraints). When ``None`` or empty, the slot
        collapses to an empty string and the prompt is byte-identical
        to the M1 layout — important so caches keyed on the prompt
        body do not invalidate spuriously when the planner is off.

        P2 only renders the block; the planner is wired into pipeline
        callers in P7.
        """
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
            chapter_plan_block=_format_chapter_plan_block(chapter_plan),
            subtitle_block=_format_segments(ctx.segments) or "(无)",
            frames_block=_format_frames(ctx.frame_descs) or "(无)",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=15), reraise=True)
    async def _call(self, user_msg: str) -> tuple[str, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": LECTURE_IR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.25,
            "response_format": {"type": "json_object"},
            "timeout": self._timeout,
            "extra_body": self._extra_body,
        }
        # Pin max_tokens to the configured ceiling so 20+ minute code-heavy
        # videos do not truncate JSON output at the backend's silent default
        # (DeepSeek defaults to 4096 per call). 0 means "let the backend
        # decide" — preserved as an escape hatch for non-DeepSeek backends.
        max_tokens = int(getattr(self._settings, "lecture_ir_max_tokens", 0) or 0)
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None) or ""
        usage: dict[str, Any] = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        if finish_reason:
            usage["finish_reason"] = finish_reason
        return content, usage

    def _parse_to_dict(
        self,
        raw: str,
        ctx: LecturizeContext,
        *,
        finish_reason: str = "",
    ) -> dict[str, Any]:
        try:
            data = _extract_json(raw)
        except ValueError as e:
            self._dump_raw(ctx.bv_id, raw, reason="ir-non-json")
            # When the backend reports finish_reason='length' the JSON tail
            # was clipped, so the parser sees an unterminated string. Surface
            # this distinct failure mode loudly so users know to raise
            # LECTURE_IR_MAX_TOKENS or route the video to map-reduce instead
            # of chasing a phantom prompt bug.
            if finish_reason == "length":
                raise RuntimeError(
                    "LLM truncated LectureIR JSON (finish_reason=length, "
                    f"raw {len(raw)} chars): {e}. Raise LECTURE_IR_MAX_TOKENS "
                    "or lower LECTURE_PROFILE_THRESHOLDS_SEC so this video "
                    "routes to the map-reduce builder."
                ) from e
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

    def _parse_and_validate(
        self,
        raw: str,
        ctx: LecturizeContext,
        *,
        finish_reason: str = "",
    ) -> LectureIR:
        data = self._parse_to_dict(raw, ctx, finish_reason=finish_reason)
        return self._validate_dict(data, ctx, raw)

    async def _run_critic_loop(
        self,
        ctx: LecturizeContext,
        ir: LectureIR,
        study_questions: list[str],
        *,
        reviser_enabled: bool = True,
    ) -> tuple[CritiqueResult, LectureIR, int, list[dict[str, Any]], dict[str, float]]:
        """Run the Critic and (optionally) the Reviser.

        When ``reviser_enabled`` is False the Critic still runs once as a
        quality audit (its issues end up in pipeline stats and the timing
        report) but the Reviser is never invoked, so the IR is returned
        unchanged. This is the recommended default for long videos where
        full-IR rewrite frequently times out; short / medium / code
        videos can opt in by setting ``LECTURE_REVISER_ENABLED=true``.
        """
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
            if (
                not reviser_enabled
                or critique.verdict != "needs_revision"
                or not critique.issues
                or rounds >= max_rounds
            ):
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
                revised_ir = LectureIR.model_validate(revised)
            except (ValidationError, RuntimeError, ValueError) as exc:
                logger.warning("Reviser produced invalid IR; keeping previous: %s", exc)
                break
            # Safety net: deepseek-flash sometimes silently drops high-value
            # arrays (code_blocks, formula_blocks, knowledge_units, …) when
            # revising unrelated fields. The Reviser system prompt forbids
            # this, but enforcing it post-hoc guarantees no regression.
            latest_ir = _restore_dropped_content(latest_ir, revised_ir)
            rounds += 1
        timing = {"critic_sec": critic_sec, "reviser_sec": reviser_sec}
        return (
            latest_critique or CritiqueResult("ok", "", [], {}),
            latest_ir,
            rounds,
            usages,
            timing,
        )

    async def _run_critic_loop_v2(
        self,
        ctx: LecturizeContext,
        ir: LectureIR,
        study_questions: list[str],
        *,
        profile: Any,
    ) -> tuple[
        "CritiqueResult | None",
        LectureIR,
        int,
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        """M2 P7 profile-driven Critic + Reviser dispatch.

        Branches:

        * ``profile.critic_mode == 'off'`` (tiny) → noop, returns the IR
          unchanged with an empty :class:`CritiqueResult`-like shape.
        * Otherwise build a :class:`CriticContext` (full or projected per
          ``profile.critic_mode``), call
          :meth:`CriticReviserAgent.audit`, and dispatch by
          ``profile.reviser_mode``:

          * ``off``: do not invoke the Reviser; the Critic's findings
            still surface in the report but the IR is returned as-is.
          * ``patch``: call
            :meth:`CriticReviserAgent.revise_patch` and apply the
            whitelisted patches via :func:`ir_patches.apply_patches`.
          * ``full``: fall back to the legacy
            :meth:`CriticReviserAgent.revise` rewrite path so existing
            integrators (and the ``LECTURE_REVISER_MODE=full`` opt-in)
            keep working.

        Returns ``(critique, latest_ir, rounds, usages, telemetry)``
        where ``telemetry`` carries ``critic_sec`` /
        ``reviser_sec`` (always) plus a ``reviser`` dict
        (mode/proposed/applied/rejected/unfixable) when the Reviser
        actually ran.
        """
        from .critic_context import build_critic_context
        from .ir_patches import apply_patches

        empty = CritiqueResult("ok", "(critic off)", [], {})
        telemetry: dict[str, Any] = {
            "critic_sec": 0.0,
            "reviser_sec": 0.0,
        }
        if profile is None or getattr(profile, "critic_mode", "off") == "off":
            return empty, ir, 0, [], telemetry

        if self._critic_agent is None:
            self._critic_agent = CriticReviserAgent()

        max_rounds = int(getattr(self._settings, "lecture_critic_max_rounds", 1) or 0)
        reviser_mode = str(getattr(profile, "reviser_mode", "off") or "off").lower()
        latest_ir = ir
        latest_critique: CritiqueResult | None = None
        usages: list[dict[str, Any]] = []
        rounds = 0
        critic_sec = 0.0
        reviser_sec = 0.0
        reviser_summary: dict[str, Any] = {
            "mode": reviser_mode,
            "proposed": 0,
            "applied": 0,
            "rejected": 0,
            "rejected_reasons": [],
            "unfixable": 0,
            "schema_rollbacks": 0,
        }

        for _ in range(max(1, max_rounds + 1)):
            context = build_critic_context(
                profile,
                ir=latest_ir,
                segments=ctx.segments,
                frames=ctx.frame_descs,
                study_questions=study_questions,
            )
            t_c = time.perf_counter()
            critique = await self._critic_agent.audit(context)
            critic_sec += time.perf_counter() - t_c
            usages.append(critique.usage or {})
            latest_critique = critique

            if (
                reviser_mode == "off"
                or critique.verdict != "needs_revision"
                or not critique.issues
                or rounds >= max_rounds
            ):
                break

            if reviser_mode == "patch":
                t_r = time.perf_counter()
                result = await self._critic_agent.revise_patch(
                    context=context, issues=critique.issues
                )
                reviser_sec += time.perf_counter() - t_r
                usages.append(result.usage or {})
                proposed = list(result.patches or [])
                rejected_pre = list(result.rejected or [])
                reviser_summary["proposed"] += len(proposed)
                reviser_summary["unfixable"] += len(result.unfixable_issues or [])
                if not proposed:
                    reviser_summary["rejected"] += len(rejected_pre)
                    for rp in rejected_pre:
                        reviser_summary["rejected_reasons"].append(rp.reason)
                    break
                next_ir, rejected_app = apply_patches(latest_ir, proposed)
                applied = max(0, len(proposed) - len(rejected_app))
                # apply_patches signals schema rollback by appending a
                # RejectedPatch with patch=None and a 'schema_violation:'
                # prefix; keep that as the visible rollback counter.
                rollbacks = sum(
                    1
                    for rp in rejected_app
                    if rp.patch is None
                    and (rp.reason or "").startswith("schema_violation")
                )
                reviser_summary["applied"] += applied
                reviser_summary["rejected"] += len(rejected_pre) + len(rejected_app)
                reviser_summary["schema_rollbacks"] += rollbacks
                for rp in rejected_pre + rejected_app:
                    reviser_summary["rejected_reasons"].append(rp.reason)
                if next_ir is latest_ir or applied == 0:
                    # Either fully rolled back (schema violation) or every
                    # patch rejected — no productive change, stop the loop.
                    break
                latest_ir = next_ir
                rounds += 1
                continue

            if reviser_mode == "full":
                t_r = time.perf_counter()
                revised, rev_usage = await self._critic_agent.revise(
                    ctx, latest_ir.model_dump(mode="json"), critique.issues
                )
                reviser_sec += time.perf_counter() - t_r
                usages.append(rev_usage or {})
                if not revised:
                    break
                try:
                    hydrate_lecture_ir_data(revised, ctx)
                    revised_ir = LectureIR.model_validate(revised)
                except (ValidationError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Reviser produced invalid IR; keeping previous: %s", exc
                    )
                    break
                latest_ir = _restore_dropped_content(latest_ir, revised_ir)
                rounds += 1
                continue

            logger.warning(
                "Unknown reviser_mode=%r; treating as off", reviser_mode
            )
            break

        telemetry["critic_sec"] = critic_sec
        telemetry["reviser_sec"] = reviser_sec
        if reviser_mode != "off":
            telemetry["reviser"] = reviser_summary
        return (latest_critique or empty, latest_ir, rounds, usages, telemetry)

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
        # Diagnostic counters from CriticReviserAgent.critique describing
        # how aggressively the Critic prompt was pruned. Always a dict
        # (possibly empty) so matrix readers don't need to special-case
        # missing keys.
        "metrics": dict(getattr(c, "metrics", {}) or {}),
    }


# Top-level arrays that the Reviser must never silently shrink. If a round
# of revision removes items from these, we treat it as a model slip and
# restore the previous version. ``points`` and chapter ``frames`` are
# intentionally excluded because the Critic legitimately asks the Reviser
# to refine, merge or split those.
_RESTORE_TOP_LEVEL_ARRAYS = (
    "knowledge_units",
    "visual_evidence",
    "study_questions",
    "review_questions",
    "glossary",
)
_RESTORE_CHAPTER_ARRAYS = (
    "code_blocks",
    "formula_blocks",
    "process_steps",
    "pitfalls",
    "key_takeaways",
)


def _restore_dropped_content(prev_ir: LectureIR, revised_ir: LectureIR) -> LectureIR:
    """Backfill arrays the Reviser dropped, returning a merged ``LectureIR``.

    For each protected array the previous IR is the safety baseline. If the
    revised IR has *fewer* items than the previous one we conclude the
    Reviser silently dropped content (the system prompt forbids this) and
    restore the previous list verbatim. The revised IR keeps every other
    field, so legitimate modifications still flow through.
    """
    revised_data = revised_ir.model_dump(mode="json")
    prev_data = prev_ir.model_dump(mode="json")

    # Top-level arrays.
    for field in _RESTORE_TOP_LEVEL_ARRAYS:
        prev_arr = prev_data.get(field) or []
        rev_arr = revised_data.get(field) or []
        if len(prev_arr) > len(rev_arr):
            logger.warning(
                "Reviser dropped %d %s items (had %d, kept %d); restoring from previous IR",
                len(prev_arr) - len(rev_arr),
                field,
                len(prev_arr),
                len(rev_arr),
            )
            revised_data[field] = prev_arr

    # Chapter-level arrays, matched by chapter index when possible.
    prev_chapters = prev_data.get("chapters") or []
    rev_chapters = revised_data.get("chapters") or []
    prev_by_idx = {ch.get("index"): ch for ch in prev_chapters if isinstance(ch, dict)}
    for ch in rev_chapters:
        if not isinstance(ch, dict):
            continue
        prev_ch = prev_by_idx.get(ch.get("index"))
        if not prev_ch:
            continue
        for field in _RESTORE_CHAPTER_ARRAYS:
            prev_arr = prev_ch.get(field) or []
            rev_arr = ch.get(field) or []
            if len(prev_arr) > len(rev_arr):
                logger.warning(
                    "Reviser dropped %d ch%s.%s items (had %d, kept %d); restoring",
                    len(prev_arr) - len(rev_arr),
                    ch.get("index"),
                    field,
                    len(prev_arr),
                    len(rev_arr),
                )
                ch[field] = prev_arr

    try:
        return LectureIR.model_validate(revised_data)
    except ValidationError as exc:
        logger.warning(
            "Restored IR failed schema validation (%s); keeping the revised IR",
            exc,
        )
        return revised_ir


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
