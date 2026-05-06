"""Structured lecture generation via Qwen3-Max (DashScope OpenAI-compatible)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..ingest.subtitle import SubtitleSegment
from .prompts import LECTURIZER_SYSTEM, LECTURIZER_USER_TEMPLATE
from .schema import LectureJSON
from .vlm import FrameDescription

logger = logging.getLogger(__name__)


@dataclass
class LecturizeContext:
    bv_id: str
    url: str
    title: str
    author: str
    duration: int
    cover_url: str
    segments: list[SubtitleSegment]
    frame_descs: list[FrameDescription]


class Lecturizer:
    """Turn subtitle + frame descriptions into a validated LectureJSON."""

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

    async def lecturize(self, ctx: LecturizeContext) -> tuple[LectureJSON, dict[str, Any]]:
        """Returns (validated lecture, raw stats dict).

        Stats include token usage (when available) and verification info.
        """
        subtitle_block = _format_segments(ctx.segments)
        frames_block = _format_frames(ctx.frame_descs)
        user = LECTURIZER_USER_TEMPLATE.format(
            bv_id=ctx.bv_id,
            title=ctx.title,
            author=ctx.author,
            duration=ctx.duration,
            cover_url=ctx.cover_url,
            subtitle_block=subtitle_block or "(无)",
            frames_block=frames_block or "(无)",
        )

        raw, usage = await self._call(user)
        lecture = self._parse_and_validate(raw, ctx)

        # Hydrate fields LLM might not set
        if not lecture.bv_id:
            lecture.bv_id = ctx.bv_id
        if not lecture.url:
            lecture.url = ctx.url
        if not lecture.cover_url:
            lecture.cover_url = ctx.cover_url
        if not lecture.author:
            lecture.author = ctx.author
        if not lecture.title:
            lecture.title = ctx.title
        if not lecture.duration:
            lecture.duration = float(ctx.duration)

        # Verify quotes against transcript and mark dubious ones
        transcript = " ".join(s.text for s in ctx.segments)
        marked = lecture.mark_unverified_points(transcript)
        stats = {
            "tokens": usage,
            "marked_unverified_points": marked,
            "model": self._model,
        }
        return lecture, stats

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        reraise=True,
    )
    async def _call(self, user_msg: str) -> tuple[str, dict[str, Any]]:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": LECTURIZER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
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

    def _parse_and_validate(self, raw: str, ctx: LecturizeContext) -> LectureJSON:
        try:
            data = _extract_json(raw)
        except ValueError as e:
            self._dump_raw(ctx.bv_id, raw, reason="non-json")
            raise RuntimeError(f"LLM returned non-JSON: {e}") from e

        # Inject context defaults so a sloppy LLM still validates
        data.setdefault("bv_id", ctx.bv_id)
        data.setdefault("url", ctx.url)
        data.setdefault("title", ctx.title)
        data.setdefault("author", ctx.author)
        data.setdefault("duration", ctx.duration)
        data.setdefault("cover_url", ctx.cover_url)

        _hydrate_lecture_data(data, ctx)

        try:
            return LectureJSON.model_validate(data)
        except ValidationError as e:
            debug_path = self._dump_raw(ctx.bv_id, raw, reason="schema-invalid")
            logger.error(
                "LectureJSON validation failed: %s\nraw saved to %s",
                e, debug_path,
            )
            raise RuntimeError(f"LectureJSON schema invalid: {e}") from e

    def _dump_raw(self, bv_id: str, raw: str, *, reason: str) -> Path:
        s = get_settings()
        debug_dir = s.data_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        p = debug_dir / f"{bv_id}.lecturize.{reason}.txt"
        p.write_text(raw, encoding="utf-8")
        return p


# ---------------- helpers ----------------


def _format_segments(segments: list[SubtitleSegment]) -> str:
    return "\n".join(
        f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.text}"
        for s in segments
    )


def _format_frames(frames: list[FrameDescription]) -> str:
    out: list[str] = []
    for f in frames:
        line = (
            f"[{int(f.timestamp // 60):02d}:{int(f.timestamp % 60):02d}] "
            f"path={f.path} · {f.caption}"
        )
        if f.ocr_text:
            line += f"  · OCR: {f.ocr_text[:200]}"
        if getattr(f, "visual_type", ""):
            line += f"  · type={f.visual_type}"
        if getattr(f, "importance_score", 0):
            line += f"  · score={f.importance_score:.2f}"
        if getattr(f, "why_useful", ""):
            line += f"  · useful={f.why_useful[:120]}"
        out.append(line)
    return "\n".join(out)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty response")
    # Strip code fences if any
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        raise ValueError("no JSON object found in response")
    return json.loads(m.group(0))


def _hydrate_lecture_data(data: dict[str, Any], ctx: LecturizeContext) -> None:
    data.setdefault("one_liner", (ctx.title or "本视频的结构化讲义")[:30])
    data.setdefault("category", "lecture")
    data.setdefault("domain_tags", [])

    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        chapters = [_fallback_chapter(ctx)]
        data["chapters"] = chapters

    normalised: list[dict[str, Any]] = []
    for idx, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict):
            chapter = {"summary": str(chapter)}
        chapter["index"] = _coerce_int(chapter.get("index"), idx)
        chapter["title"] = str(chapter.get("title") or f"第 {idx} 章").strip()
        chapter["summary"] = str(chapter.get("summary") or chapter["title"]).strip()
        if "start" not in chapter:
            chapter["start"] = _first_present(
                chapter, ["start_time", "start_sec", "from", "begin"]
            )
        if "end" not in chapter:
            chapter["end"] = _first_present(chapter, ["end_time", "end_sec", "to", "finish"])
        chapter["points"] = _normalise_points(chapter.get("points"), chapter, ctx)
        chapter["frames"] = _normalise_frames(chapter.get("frames"), ctx)
        chapter["learning_goal"] = str(
            chapter.get("learning_goal")
            or chapter.get("goal")
            or chapter.get("question")
            or f"理解「{chapter['title']}」的核心内容"
        ).strip()
        chapter["teaching_notes"] = (
            _normalise_string_list(chapter.get("teaching_notes"))
            or _normalise_string_list(chapter.get("explanations"))
            or [chapter["summary"]]
        )
        chapter["process_steps"] = _normalise_string_list(
            chapter.get("process_steps")
            or chapter.get("steps")
            or chapter.get("workflow")
        )
        chapter["pitfalls"] = _normalise_string_list(
            chapter.get("pitfalls")
            or chapter.get("warnings")
            or chapter.get("misunderstandings")
        )
        chapter["key_takeaways"] = _normalise_string_list(
            chapter.get("key_takeaways")
            or chapter.get("takeaways")
            or chapter.get("conclusions")
        ) or [p["text"] for p in chapter["points"][:3] if isinstance(p, dict) and p.get("text")]
        normalised.append(chapter)

    _fill_chapter_ranges(normalised, ctx.duration)
    data["chapters"] = normalised
    data["learning_path"] = (
        _normalise_string_list(data.get("learning_path"))
        or _fallback_learning_path(normalised)
    )
    data["final_synthesis"] = str(
        data.get("final_synthesis")
        or data.get("synthesis")
        or _fallback_final_synthesis(normalised, ctx)
    ).strip()
    data["highlights"] = _normalise_highlights(data.get("highlights"), ctx)
    data["glossary"] = _normalise_glossary(data.get("glossary"), ctx.duration)
    data["review_questions"] = _normalise_string_list(data.get("review_questions"))[:8]


def _fallback_chapter(ctx: LecturizeContext) -> dict[str, Any]:
    summary = " ".join(s.text for s in ctx.segments[:8]).strip() or ctx.title
    return {
        "index": 1,
        "title": ctx.title or "视频讲义",
        "start": 0,
        "end": ctx.duration,
        "summary": summary,
        "learning_goal": f"理解「{ctx.title or '视频'}」的核心内容",
        "teaching_notes": [summary] if summary else [],
        "process_steps": [],
        "points": [
            {
                "text": (ctx.segments[0].text if ctx.segments else ctx.title)[:80],
                "ts": ctx.segments[0].start if ctx.segments else 0,
                "quote": (ctx.segments[0].text if ctx.segments else ctx.title)[:80],
            }
        ],
        "frames": [],
        "pitfalls": [],
        "key_takeaways": [],
    }


def _normalise_points(
    raw_points: Any,
    chapter: dict[str, Any],
    ctx: LecturizeContext,
) -> list[dict[str, Any]]:
    if not isinstance(raw_points, list):
        raw_points = []
    points: list[dict[str, Any]] = []
    default_ts = _coerce_float(chapter.get("start"), 0.0)
    for raw in raw_points:
        if not isinstance(raw, dict):
            raw = {"text": str(raw)}
        ts = _coerce_float(_first_present(raw, ["ts", "timestamp", "time", "second"]), default_ts)
        ts = _clamp(ts, 0.0, float(ctx.duration))
        quote = str(
            raw.get("quote")
            or raw.get("evidence")
            or raw.get("subtitle_quote")
            or raw.get("source")
            or ""
        ).strip()
        if not quote:
            quote = _nearest_quote(ctx.segments, ts)
        text = str(
            raw.get("text")
            or raw.get("point")
            or raw.get("claim")
            or raw.get("summary")
            or quote
            or chapter.get("title")
            or ""
        ).strip()
        if text:
            points.append({"text": text, "ts": ts, "quote": quote or text})
    if not points:
        ts = _coerce_float(chapter.get("start"), 0.0)
        quote = _nearest_quote(ctx.segments, ts)
        points.append(
            {
                "text": quote or chapter["title"],
                "ts": ts,
                "quote": quote or chapter["title"],
            }
        )
    return points


def _normalise_frames(raw_frames: Any, ctx: LecturizeContext) -> list[dict[str, Any]]:
    if not isinstance(raw_frames, list):
        return []
    frames: list[dict[str, Any]] = []
    for raw in raw_frames:
        if not isinstance(raw, dict):
            continue
        ts = _coerce_float(_first_present(raw, ["ts", "timestamp", "time", "second"]), 0.0)
        ts = _clamp(ts, 0.0, float(ctx.duration))
        nearest = _nearest_frame(ctx.frame_descs, ts)
        path = str(raw.get("path") or raw.get("image_path") or raw.get("file") or "")
        if not path and nearest:
            path = str(nearest.path)
        caption = str(
            raw.get("caption")
            or raw.get("description")
            or raw.get("text")
            or (nearest.caption if nearest else "")
        ).strip()
        ocr_text = str(
            raw.get("ocr_text")
            or raw.get("ocr")
            or raw.get("ocrText")
            or (nearest.ocr_text if nearest else "")
        ).strip()
        insight = str(
            raw.get("insight")
            or raw.get("why_it_matters")
            or raw.get("reason")
            or raw.get("analysis")
            or ""
        ).strip()
        if not insight:
            insight = caption or ocr_text[:120]
        if path:
            frames.append(
                {
                    "ts": ts,
                    "path": path,
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "insight": insight,
                    "visual_type": str(
                        raw.get("visual_type")
                        or (nearest.visual_type if nearest and hasattr(nearest, "visual_type") else "")
                    ),
                    "selected_reason": str(
                        raw.get("selected_reason")
                        or raw.get("why_useful")
                        or (nearest.why_useful if nearest and hasattr(nearest, "why_useful") else "")
                    ),
                    "importance_score": _clamp(
                        _coerce_float(
                            raw.get("importance_score")
                            or (nearest.importance_score if nearest and hasattr(nearest, "importance_score") else 0.0),
                            0.0,
                        ),
                        0.0,
                        1.0,
                    ),
                }
            )
    return frames


def _fill_chapter_ranges(chapters: list[dict[str, Any]], duration: int) -> None:
    n = max(len(chapters), 1)
    for idx, chapter in enumerate(chapters):
        if chapter.get("start") is None:
            timestamps = _chapter_timestamps(chapter)
            fallback = duration * idx / n if duration else 0.0
            chapter["start"] = min(timestamps) if timestamps else fallback
        chapter["start"] = _clamp(
            _coerce_float(chapter.get("start"), 0.0),
            0.0,
            float(duration),
        )

    for idx, chapter in enumerate(chapters):
        if chapter.get("end") is None:
            timestamps = _chapter_timestamps(chapter)
            next_start = chapters[idx + 1]["start"] if idx + 1 < len(chapters) else None
            if next_start is not None and next_start > chapter["start"]:
                chapter["end"] = next_start
            elif timestamps:
                chapter["end"] = min(float(duration), max(timestamps) + 15.0)
            else:
                chapter["end"] = duration * (idx + 1) / n if duration else chapter["start"]
        chapter["end"] = _clamp(
            _coerce_float(chapter.get("end"), chapter["start"]),
            chapter["start"],
            float(duration),
        )


def _chapter_timestamps(chapter: dict[str, Any]) -> list[float]:
    timestamps: list[float] = []
    for point in chapter.get("points", []):
        if isinstance(point, dict):
            timestamps.append(_coerce_float(point.get("ts"), 0.0))
    for frame in chapter.get("frames", []):
        if isinstance(frame, dict):
            timestamps.append(_coerce_float(frame.get("ts"), 0.0))
    return timestamps


def _normalise_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if not isinstance(raw, list):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("title")
                or item.get("summary")
                or item.get("question")
                or item.get("step")
                or item.get("content")
                or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def _fallback_learning_path(chapters: list[dict[str, Any]]) -> list[str]:
    path: list[str] = []
    for chapter in chapters[:6]:
        title = str(chapter.get("title") or "").strip()
        if title:
            path.append(f"先理解：{title}")
    return path


def _fallback_final_synthesis(chapters: list[dict[str, Any]], ctx: LecturizeContext) -> str:
    summaries = [
        str(ch.get("summary") or "").strip()
        for ch in chapters[:4]
        if str(ch.get("summary") or "").strip()
    ]
    if summaries:
        return " ".join(summaries)[:500]
    return ctx.title or "本视频围绕核心主题展开讲解。"


def _normalise_highlights(raw_highlights: Any, ctx: LecturizeContext) -> list[dict[str, Any]]:
    if not isinstance(raw_highlights, list):
        return []
    highlights: list[dict[str, Any]] = []
    for raw in raw_highlights:
        if isinstance(raw, str):
            raw = {"text": raw, "ts": 0}
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or raw.get("quote") or "").strip()
        if not text:
            continue
        ts = _clamp(
            _coerce_float(_first_present(raw, ["ts", "timestamp", "time", "second"]), 0.0),
            0.0,
            float(ctx.duration),
        )
        highlights.append({"text": text, "ts": ts})
    return highlights


def _normalise_glossary(raw_glossary: Any, duration: int) -> list[dict[str, Any]]:
    if not isinstance(raw_glossary, list):
        return []
    glossary: list[dict[str, Any]] = []
    for raw in raw_glossary:
        if isinstance(raw, str):
            raw = {"term": raw}
        if not isinstance(raw, dict):
            continue
        term = str(raw.get("term") or raw.get("name") or "").strip()
        if not term:
            continue
        ts = _clamp(
            _coerce_float(_first_present(raw, ["ts", "timestamp", "time", "second"]), 0.0),
            0.0,
            float(duration),
        )
        glossary.append(
            {
                "term": term,
                "ts": ts,
                "explanation": str(raw.get("explanation") or raw.get("definition") or "").strip(),
            }
        )
    return glossary


def _nearest_quote(segments: list[SubtitleSegment], ts: float) -> str:
    if not segments:
        return ""
    segment = min(segments, key=lambda s: abs(float(s.start) - ts))
    return segment.text.strip()[:80]


def _nearest_frame(frames: list[FrameDescription], ts: float) -> FrameDescription | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(float(f.timestamp) - ts))


def _first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if maximum < minimum:
        return minimum
    return max(minimum, min(maximum, value))
