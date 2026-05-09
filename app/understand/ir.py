from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .length_adapt import (
    chapter_frame_max,
    points_per_chapter_max,
)
from .schema import Taxonomy

PrimaryType = Literal["technical_formula", "conceptual_talk", "procedural_tutorial", "generic_lecture"]
Density = Literal["brief", "standard", "dense"]
KnowledgeType = Literal[
    "concept",
    "mechanism",
    "formula",
    "code",
    "procedure",
    "example",
    "boundary",
    "pitfall",
]
VisualType = Literal["diagram", "formula", "code", "table", "ui", "slide_text", "person", "other"]


class LectureProfile(BaseModel):
    primary_type: PrimaryType = "generic_lecture"
    secondary_types: list[str] = Field(default_factory=list)
    density: Density = "standard"
    required_sections: list[str] = Field(default_factory=list)
    optional_sections: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)


class IRPoint(BaseModel):
    text: str = ""
    ts: float = Field(default=0.0, ge=0)
    quote: str = ""


class IRFrame(BaseModel):
    ts: float = Field(default=0.0, ge=0)
    path: str = ""
    caption: str = ""
    ocr_text: str = ""
    insight: str = ""
    visual_type: VisualType = "other"
    selected_reason: str = ""
    importance_score: float = Field(default=0.0, ge=0, le=1)


class IRCodeBlock(BaseModel):
    """First-class code excerpt (mirrors :class:`schema.CodeBlock`)."""

    language: str = ""
    code: str = ""
    ts: float = Field(default=0.0, ge=0)
    chapter_index: int = Field(default=1, ge=1)
    source: str = "ocr"
    explanation: str = ""
    related_frame_paths: list[str] = Field(default_factory=list)


class IRFormulaBlock(BaseModel):
    """First-class LaTeX formula (mirrors :class:`schema.FormulaBlock`)."""

    latex: str = ""
    ts: float = Field(default=0.0, ge=0)
    chapter_index: int = Field(default=1, ge=1)
    explanation: str = ""
    related_frame_paths: list[str] = Field(default_factory=list)


class IRChapter(BaseModel):
    index: int = Field(default=1, ge=1)
    title: str = ""
    start: float = Field(default=0.0, ge=0)
    end: float = Field(default=0.0, ge=0)
    summary: str = ""
    learning_goal: str = ""
    teaching_notes: list[str] = Field(default_factory=list)
    process_steps: list[str] = Field(default_factory=list)
    points: list[IRPoint] = Field(default_factory=list)
    frames: list[IRFrame] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    knowledge_unit_ids: list[str] = Field(default_factory=list)
    code_blocks: list[IRCodeBlock] = Field(default_factory=list)
    formula_blocks: list[IRFormulaBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "IRChapter":
        if self.end < self.start:
            raise ValueError(f"chapter {self.index}: end {self.end} < start {self.start}")
        return self


class KnowledgeUnit(BaseModel):
    id: str = ""
    type: KnowledgeType = "concept"
    title: str = ""
    explanation: str = ""
    ts: float = Field(default=0.0, ge=0)
    quote: str = ""
    chapter_index: int = Field(default=1, ge=1)
    related_frames: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)


class Timeline(BaseModel):
    turning_points: list[str] = Field(default_factory=list)
    argument_path: list[str] = Field(default_factory=list)
    state_evolution: list[str] = Field(default_factory=list)
    chapter_boundaries: list[str] = Field(default_factory=list)


class VisualEvidence(BaseModel):
    ts: float = Field(default=0.0, ge=0)
    path: str = ""
    caption: str = ""
    ocr_text: str = ""
    visual_type: VisualType = "other"
    importance_score: float = Field(default=0.0, ge=0, le=1)
    ocr_density: float = Field(default=0.0, ge=0, le=1)
    novelty_score: float = Field(default=0.0, ge=0, le=1)
    selected: bool = False
    selected_reason: str = ""
    linked_knowledge_unit: str = ""


class Completeness(BaseModel):
    mainline_closed: bool = False
    missing_prerequisites: list[str] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)
    missing_examples: list[str] = Field(default_factory=list)
    missing_boundaries: list[str] = Field(default_factory=list)
    visual_coverage: str = ""
    topic_reusability: bool = False
    reader_can_understand_without_video: bool = False
    notes: str = ""


class RenderPlan(BaseModel):
    hero: bool = True
    learning_map: bool = True
    completeness_card: bool = True
    timeline_path: bool = True
    chapter_notes: bool = True
    formula_section: bool = False
    code_section: bool = False
    procedure_section: bool = False
    visual_evidence_section: bool = True
    final_synthesis: bool = True
    review_questions: bool = True
    glossary: bool = True


class LectureIR(BaseModel):
    bv_id: str = ""
    url: str = ""
    title: str = ""
    author: str = ""
    duration: float = Field(default=0.0, ge=0)
    cover: str = ""
    profile: LectureProfile = Field(default_factory=LectureProfile)
    core_question: str = ""
    mainline: list[str] = Field(default_factory=list)
    knowledge_units: list[KnowledgeUnit] = Field(default_factory=list)
    timeline: Timeline = Field(default_factory=Timeline)
    visual_evidence: list[VisualEvidence] = Field(default_factory=list)
    chapters: list[IRChapter] = Field(default_factory=list)
    completeness: Completeness = Field(default_factory=Completeness)
    render_plan: RenderPlan = Field(default_factory=RenderPlan)
    final_synthesis: str = ""
    review_questions: list[str] = Field(default_factory=list)
    study_questions: list[str] = Field(
        default_factory=list,
        description="Question-Driven 抽取阶段产出的核心学习问题",
    )
    taxonomy: Taxonomy | None = None

    @model_validator(mode="after")
    def _check_timestamps(self) -> "LectureIR":
        for ch in self.chapters:
            if ch.end > self.duration + 1.0:
                raise ValueError(f"chapter {ch.index} end {ch.end} > duration {self.duration}")
        return self


def hydrate_lecture_ir_data(data: dict[str, Any], ctx: Any) -> None:
    transcript = " ".join(getattr(s, "text", "") for s in getattr(ctx, "segments", []))
    frame_text = " ".join(
        f"{getattr(f, 'caption', '')} {getattr(f, 'ocr_text', '')}"
        for f in getattr(ctx, "frame_descs", [])
    )
    data.setdefault("bv_id", getattr(ctx, "bv_id", ""))
    data.setdefault("url", getattr(ctx, "url", ""))
    data.setdefault("title", getattr(ctx, "title", ""))
    data.setdefault("author", getattr(ctx, "author", ""))
    data.setdefault("duration", getattr(ctx, "duration", 0))
    data.setdefault("cover", getattr(ctx, "cover_url", ""))
    data["profile"] = _hydrate_profile(data.get("profile"), data["title"], transcript, frame_text)
    data["core_question"] = str(data.get("core_question") or f"这段视频要解决什么问题：{data['title']}").strip()
    data["chapters"] = _normalise_chapters(data.get("chapters"), ctx)
    data["mainline"] = _normalise_string_list(data.get("mainline")) or _fallback_mainline(data["chapters"])
    data["visual_evidence"] = _normalise_visuals(data.get("visual_evidence"), ctx)
    _attach_visuals_to_chapters(data["chapters"], data["visual_evidence"])
    data["knowledge_units"] = _normalise_units(data.get("knowledge_units"), data["chapters"], data["profile"], ctx)
    data["timeline"] = _hydrate_timeline(data.get("timeline"), data["chapters"], data["mainline"])
    data["completeness"] = _hydrate_completeness(data.get("completeness"), data)
    data["render_plan"] = _hydrate_render_plan(data.get("render_plan"), data["profile"])
    data["final_synthesis"] = str(
        data.get("final_synthesis") or data.get("synthesis") or " → ".join(data["mainline"][:6])
    ).strip()
    data["review_questions"] = _normalise_string_list(data.get("review_questions")) or [
        f"为什么「{ch.get('title', '本章')}」是主线中的必要环节？"
        for ch in data["chapters"][:5]
    ]
    data["study_questions"] = _normalise_string_list(data.get("study_questions"))


def infer_primary_type(title: str, transcript: str = "", frame_text: str = "") -> PrimaryType:
    text = f"{title} {transcript[:3000]} {frame_text[:2000]}".lower()
    technical = ["代码", "公式", "算法", "模型", "矩阵", "张量", "维度", "推导", "attention", "transformer", "python", "mask", "多头", "神经网络"]
    procedural = ["教程", "步骤", "操作", "安装", "配置", "实操", "流程", "做法", "怎么做", "排错", "检查点"]
    conceptual = ["讲座", "认知", "观点", "框架", "方法论", "为什么", "理解", "思考", "边界", "误解"]
    if any(k in text for k in technical):
        return "technical_formula"
    if any(k in text for k in procedural):
        return "procedural_tutorial"
    if any(k in text for k in conceptual):
        return "conceptual_talk"
    return "generic_lecture"


def infer_visual_type(caption: str, ocr_text: str = "") -> VisualType:
    text = f"{caption} {ocr_text}".lower()
    if any(k in text for k in ["def ", "class ", "import ", "代码", "code", "函数", "return", "python"]):
        return "code"
    if any(k in text for k in ["公式", "=", "∑", "矩阵", "向量", "维度", "qkv", "softmax"]):
        return "formula"
    if any(k in text for k in ["流程", "结构", "架构", "图", "箭头", "框图", "diagram"]):
        return "diagram"
    if any(k in text for k in ["表格", "table", "列", "行"]):
        return "table"
    if any(k in text for k in ["界面", "按钮", "菜单", "窗口", "ui"]):
        return "ui"
    if len(ocr_text.strip()) >= 12:
        return "slide_text"
    if any(k in text for k in ["人物", "人像", "up主", "讲者"]):
        return "person"
    return "other"


def _hydrate_profile(raw: Any, title: str, transcript: str, frame_text: str) -> dict[str, Any]:
    profile = raw if isinstance(raw, dict) else {}
    primary = profile.get("primary_type") or infer_primary_type(title, transcript, frame_text)
    if primary not in PrimaryType.__args__:
        primary = infer_primary_type(title, transcript, frame_text)
    density = profile.get("density") or ("dense" if primary == "technical_formula" else "standard")
    if density not in Density.__args__:
        density = "dense" if primary == "technical_formula" else "standard"
    required = _normalise_string_list(profile.get("required_sections")) or [
        "learning_map", "completeness_card", "timeline_path", "chapter_notes", "final_synthesis"
    ]
    optional = _normalise_string_list(profile.get("optional_sections")) or [
        "visual_evidence_section", "review_questions", "glossary"
    ]
    return {
        "primary_type": primary,
        "secondary_types": _normalise_string_list(profile.get("secondary_types")),
        "density": density,
        "required_sections": required,
        "optional_sections": optional,
        "domain_tags": _normalise_string_list(profile.get("domain_tags")),
    }


def _normalise_chapters(raw: Any, ctx: Any) -> list[dict[str, Any]]:
    chapters = raw if isinstance(raw, list) and raw else [_fallback_chapter(ctx)]
    duration = int(getattr(ctx, "duration", 0) or 0)
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(chapters, start=1):
        ch = item if isinstance(item, dict) else {"summary": str(item)}
        ch["index"] = _coerce_int(ch.get("index"), idx)
        ch["title"] = str(ch.get("title") or f"第 {idx} 章").strip()
        ch["summary"] = str(ch.get("summary") or ch["title"]).strip()
        ch["start"] = _coerce_float(_first_present(ch, ["start", "start_time", "begin"]), None)
        ch["end"] = _coerce_float(_first_present(ch, ["end", "end_time", "finish"]), None)
        ch["learning_goal"] = str(ch.get("learning_goal") or f"理解「{ch['title']}」如何推进主线").strip()
        ch["teaching_notes"] = _normalise_string_list(ch.get("teaching_notes") or ch.get("explanations")) or [ch["summary"]]
        ch["process_steps"] = _normalise_string_list(ch.get("process_steps") or ch.get("steps") or ch.get("workflow"))
        ch["points"] = _normalise_points(ch.get("points"), ch, ctx)
        ch["frames"] = _normalise_chapter_frames(ch.get("frames"), ctx)
        ch["pitfalls"] = _normalise_string_list(ch.get("pitfalls") or ch.get("warnings"))
        ch["key_takeaways"] = _normalise_string_list(ch.get("key_takeaways") or ch.get("takeaways")) or [p["text"] for p in ch["points"][:2] if p.get("text")]
        ch["knowledge_unit_ids"] = _normalise_string_list(ch.get("knowledge_unit_ids"))
        ch["code_blocks"] = _normalise_code_blocks(
            ch.get("code_blocks") or ch.get("code") or ch.get("snippets"),
            ch,
            duration,
        )
        ch["formula_blocks"] = _normalise_formula_blocks(
            ch.get("formula_blocks") or ch.get("formulas"),
            ch,
            duration,
        )
        # Length-adaptive cap on points: prevents long chapters from
        # being truncated to the legacy hidden ceiling of 5 while still
        # bounding the total HTML size.
        ch_seconds = max(0.0, _coerce_float(ch.get("end"), 0.0) - _coerce_float(ch.get("start"), 0.0))
        ch["points"] = ch["points"][: points_per_chapter_max(ch_seconds)]
        out.append(ch)
    _fill_chapter_ranges(out, duration)
    _repair_point_timestamps(out, ctx, duration)
    return out


def _normalise_visuals(raw: Any, ctx: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list) and raw:
        items = [x for x in raw if isinstance(x, dict)]
    else:
        items = []
    by_path = {str(x.get("path")): x for x in items if x.get("path")}
    visuals: list[dict[str, Any]] = []
    for fd in getattr(ctx, "frame_descs", []):
        path = str(getattr(fd, "path", ""))
        item = by_path.get(path, {})
        caption = str(item.get("caption") or getattr(fd, "caption", "")).strip()
        ocr = str(item.get("ocr_text") or getattr(fd, "ocr_text", "")).strip()
        visual_type = str(item.get("visual_type") or getattr(fd, "visual_type", "") or infer_visual_type(caption, ocr))
        score = _coerce_float(item.get("importance_score") or getattr(fd, "importance_score", 0.0), 0.0)
        if not score:
            score = _visual_score(visual_type, ocr)
        selected = bool(item.get("selected", score >= 0.55 and visual_type != "person"))
        reason = str(item.get("selected_reason") or item.get("why_useful") or getattr(fd, "why_useful", "") or "这张图补充了字幕中难以完整表达的结构信息。").strip()
        visuals.append({
            # Trust the path-joined VLM cache timestamp; LLM is unreliable for ts
            # (commonly extracts only the minute part from "[MM:SS]" prompt formatting).
            "ts": _coerce_float(getattr(fd, "timestamp", 0.0), 0.0),
            "path": path,
            "caption": caption,
            "ocr_text": ocr,
            "visual_type": visual_type if visual_type in VisualType.__args__ else "other",
            "importance_score": _clamp(score, 0.0, 1.0),
            "ocr_density": _clamp(_coerce_float(item.get("ocr_density") or getattr(fd, "ocr_density", 0.0), len(ocr) / 500), 0.0, 1.0),
            "novelty_score": _clamp(_coerce_float(item.get("novelty_score") or getattr(fd, "novelty_score", 0.0), score), 0.0, 1.0),
            "selected": selected,
            "selected_reason": reason,
            "linked_knowledge_unit": str(item.get("linked_knowledge_unit") or ""),
        })
    if visuals:
        return visuals
    return [x for x in items if x.get("path")]


def _attach_visuals_to_chapters(chapters: list[dict[str, Any]], visuals: list[dict[str, Any]]) -> None:
    selected = [v for v in visuals if v.get("selected") and v.get("path")]
    for ch in chapters:
        # Respect frames the LLM already attached to THIS chapter, but still
        # auto-distribute for chapters that came back empty. The previous early
        # `return` left every empty chapter without visual support whenever any
        # other chapter had even one LLM-attached frame.
        if ch.get("frames"):
            continue
        start = _coerce_float(ch.get("start"), 0.0)
        end = _coerce_float(ch.get("end"), start)
        ch_seconds = max(0.0, end - start)
        cap = chapter_frame_max(ch_seconds)
        local = [v for v in selected if start <= _coerce_float(v.get("ts"), 0.0) <= end]
        ch["frames"] = [
            {
                "ts": v.get("ts", 0),
                "path": v.get("path", ""),
                "caption": v.get("caption", ""),
                "ocr_text": v.get("ocr_text", ""),
                "insight": v.get("selected_reason", ""),
                "visual_type": v.get("visual_type", "other"),
                "selected_reason": v.get("selected_reason", ""),
                "importance_score": v.get("importance_score", 0),
            }
            for v in sorted(local, key=lambda x: x.get("importance_score", 0), reverse=True)[:cap]
        ]


def _normalise_units(raw: Any, chapters: list[dict[str, Any]], profile: dict[str, Any], ctx: Any | None = None) -> list[dict[str, Any]]:
    units = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
    if not units:
        for ch in chapters:
            for i, text in enumerate(ch.get("key_takeaways", [])[:2], start=1):
                units.append({
                    "id": f"ch{ch['index']}-u{i}",
                    "type": "mechanism" if profile.get("primary_type") == "technical_formula" else "concept",
                    "title": text[:40],
                    "explanation": text,
                    "ts": ch.get("start", 0),
                    "quote": ch.get("points", [{}])[0].get("quote", "") if ch.get("points") else "",
                    "chapter_index": ch.get("index", 1),
                    "domain_tags": profile.get("domain_tags", []),
                })
    chapters_by_index = {int(ch.get("index") or 0): ch for ch in chapters}
    fallback_chapter = chapters[0] if chapters else {"start": 0.0, "end": 0.0, "title": "", "index": 1}
    out: list[dict[str, Any]] = []
    allowed = KnowledgeType.__args__
    for idx, unit in enumerate(units, start=1):
        kind = str(unit.get("type") or "concept")
        ch_idx = _coerce_int(unit.get("chapter_index"), 1)
        owning_chapter = chapters_by_index.get(int(ch_idx) or 0, fallback_chapter)
        ts_value = _locate_anchor_ts(unit, owning_chapter, ctx) if ctx is not None else _coerce_float(unit.get("ts") or unit.get("timestamp"), 0.0)
        out.append({
            "id": str(unit.get("id") or f"ku-{idx}"),
            "type": kind if kind in allowed else "concept",
            "title": str(unit.get("title") or unit.get("term") or f"知识点 {idx}").strip(),
            "explanation": str(unit.get("explanation") or unit.get("summary") or "").strip(),
            "ts": ts_value,
            "quote": str(unit.get("quote") or unit.get("evidence") or "").strip(),
            "chapter_index": ch_idx,
            "related_frames": _normalise_string_list(unit.get("related_frames")),
            "prerequisites": _normalise_string_list(unit.get("prerequisites")),
            "depends_on": _normalise_string_list(unit.get("depends_on")),
            "domain_tags": _normalise_string_list(unit.get("domain_tags")),
        })
    return out


def _normalise_points(raw: Any, chapter: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
    points = [x if isinstance(x, dict) else {"text": str(x)} for x in raw] if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    for p in points:
        ts = _locate_anchor_ts(p, chapter, ctx)
        quote = str(p.get("quote") or p.get("evidence") or _nearest_quote(getattr(ctx, "segments", []), ts)).strip()
        text = str(p.get("text") or p.get("point") or p.get("claim") or quote or chapter.get("title") or "").strip()
        if text:
            out.append({"text": text, "ts": ts, "quote": quote or text})
    if out:
        return out
    ts = _coerce_float(chapter.get("start"), 0.0)
    quote = _nearest_quote(getattr(ctx, "segments", []), ts)
    return [{"text": quote or chapter.get("title", ""), "ts": ts, "quote": quote or chapter.get("title", "")}]


def _locate_anchor_ts(item: dict[str, Any], chapter: dict[str, Any], ctx: Any) -> float:
    """Pick a plausible anchor ts for a point/knowledge-unit.

    The LLM frequently writes ``ts=0`` even when the chapter starts deep into
    the video, or echoes back the minute portion of ``[MM:SS]`` prompt
    formatting. We trust an LLM ts only when it lands inside the chapter
    range; otherwise we look up the matching subtitle by quote/text and fall
    back to the chapter midpoint.
    """
    duration = float(getattr(ctx, "duration", 0) or 0)
    ch_start = _coerce_float(chapter.get("start"), 0.0) or 0.0
    ch_end = _coerce_float(chapter.get("end"), ch_start) or ch_start
    raw_ts = _first_present(item, ["ts", "timestamp", "time"])
    ts = _coerce_float(raw_ts, None)
    plausible = (
        ts is not None
        and ch_start - 5 <= ts <= ch_end + 5
        and not (ts == 0.0 and ch_start > 5.0)
    )
    if plausible:
        return _clamp(float(ts), 0.0, duration)
    needle = str(item.get("quote") or item.get("text") or item.get("title") or "").strip()
    found = _find_subtitle_ts(getattr(ctx, "segments", []), needle, ch_start, ch_end)
    if found is not None:
        return _clamp(found, 0.0, duration)
    midpoint = (ch_start + ch_end) / 2 if ch_end > ch_start else ch_start
    return _clamp(midpoint, 0.0, duration)


def _repair_point_timestamps(chapters: list[dict[str, Any]], ctx: Any, duration: int) -> None:
    for chapter in chapters:
        ch_start = _coerce_float(chapter.get("start"), 0.0) or 0.0
        ch_end = _coerce_float(chapter.get("end"), ch_start) or ch_start
        if ch_end < ch_start:
            ch_end = ch_start
        for point in chapter.get("points", []):
            if not isinstance(point, dict):
                continue
            ts = _coerce_float(point.get("ts"), ch_start)
            if ch_start <= ts <= ch_end:
                point["ts"] = round(_clamp(ts, 0.0, float(duration or 0)), 3)
                continue
            repaired = _locate_anchor_ts(point, chapter, ctx)
            point["ts"] = round(_clamp(repaired, ch_start, ch_end), 3)


def _find_subtitle_ts(
    segments: list[Any], needle: str, ch_start: float, ch_end: float
) -> float | None:
    """Find a subtitle segment ts within ``[ch_start-5, ch_end+5]`` whose
    text matches ``needle``.

    Out-of-range matches are intentionally rejected: when the LLM mis-files
    a point under the wrong chapter, jumping the user to a video moment in
    a *different* chapter is more confusing than landing inside the
    declared chapter (caller falls back to the chapter midpoint). Returns
    ``None`` when no in-range fragment of the needle is found.
    """
    if not needle or not segments:
        return None
    head = needle[:24].strip()
    if not head:
        return None
    for s in segments:
        s_start = float(getattr(s, "start", 0) or 0)
        if not (ch_start - 5 <= s_start <= ch_end + 5):
            continue
        text = str(getattr(s, "text", "") or "")
        if not text:
            continue
        if head in text or text[:24] in needle:
            return s_start
    return None


_FENCED_CODE_RE = re.compile(r"```([A-Za-z0-9_+\-.#]*)?[ \t]*\n(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> tuple[str, str]:
    """Return ``(language, body)`` if ``text`` is a fenced block, else ('', text).

    LLMs often output ``code`` already wrapped in ``` ```python\n…\n``` ```.
    We unwrap so the renderer can apply its own ``<pre><code>`` wrapper
    consistently and ``<pre>`` doesn't end up nested inside a ``<pre>``.
    """
    text = text or ""
    match = _FENCED_CODE_RE.search(text)
    if not match:
        return "", text
    return (match.group(1) or "").strip().lower(), match.group(2).strip("\n")


def _normalise_code_blocks(raw: Any, chapter: dict[str, Any], duration: int) -> list[dict[str, Any]]:
    """Hydrate ``code_blocks`` from any reasonable LLM payload.

    Tolerates:
    * ``[{"language": "python", "code": "...", "ts": 12, "explanation": "..."}]``
    * ``[{"lang": "...", "code": "..."}]`` (alternate key names)
    * Plain strings (treated as ``code``; language probed from a fence)
    * ``None`` / non-list (returns ``[]``)
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    chapter_index = _coerce_int(chapter.get("index"), 1)
    default_ts = _coerce_float(chapter.get("start"), 0.0)
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            language, body = _strip_code_fence(item)
            code = body
        elif isinstance(item, dict):
            raw_code = item.get("code") or item.get("snippet") or item.get("text") or ""
            fence_lang, body = _strip_code_fence(str(raw_code))
            language = str(item.get("language") or item.get("lang") or fence_lang or "").strip().lower()
            code = body if body else str(raw_code)
        else:
            continue
        # Tolerate single-line code that arrived as JSON-escaped \n; the
        # ``_latex_repair`` pass usually fixes this, but a belt-and-braces
        # decode never hurts.
        code = (code or "").replace("\r\n", "\n").rstrip("\n")
        if not code.strip():
            continue
        ts = _clamp(
            _coerce_float(_first_present(item if isinstance(item, dict) else {}, ["ts", "timestamp", "time"]), default_ts),
            0.0,
            float(duration or 0),
        )
        explanation = (
            str(item.get("explanation") or item.get("note") or "")
            if isinstance(item, dict)
            else ""
        ).strip()
        source = (
            str(item.get("source") or "ocr")
            if isinstance(item, dict)
            else "ocr"
        ).strip().lower()
        if source not in {"ocr", "narrated", "reconstructed"}:
            source = "ocr"
        related = (
            _normalise_string_list(item.get("related_frame_paths") or item.get("frames"))
            if isinstance(item, dict)
            else []
        )
        out.append(
            {
                "language": _sanitize_language(language),
                "code": code,
                "ts": ts,
                "chapter_index": _coerce_int(item.get("chapter_index") if isinstance(item, dict) else None, chapter_index),
                "source": source,
                "explanation": explanation,
                "related_frame_paths": related,
            }
        )
    return out


def _normalise_formula_blocks(raw: Any, chapter: dict[str, Any], duration: int) -> list[dict[str, Any]]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    chapter_index = _coerce_int(chapter.get("index"), 1)
    default_ts = _coerce_float(chapter.get("start"), 0.0)
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            latex = item
            payload = {}
        elif isinstance(item, dict):
            payload = item
            latex = str(item.get("latex") or item.get("formula") or item.get("text") or "").strip()
        else:
            continue
        if not latex:
            continue
        ts = _clamp(
            _coerce_float(_first_present(payload, ["ts", "timestamp", "time"]), default_ts),
            0.0,
            float(duration or 0),
        )
        explanation = str(payload.get("explanation") or payload.get("note") or "").strip()
        related = _normalise_string_list(payload.get("related_frame_paths") or payload.get("frames"))
        out.append(
            {
                "latex": latex,
                "ts": ts,
                "chapter_index": _coerce_int(payload.get("chapter_index"), chapter_index),
                "explanation": explanation,
                "related_frame_paths": related,
            }
        )
    return out


def _sanitize_language(language: str) -> str:
    """Restrict to a safe charset for use as a CSS class / hljs language."""
    return re.sub(r"[^A-Za-z0-9_+\-.#]", "", str(language or "")).lower()[:32]


def _normalise_chapter_frames(raw: Any, ctx: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    # Map keyframe path -> authoritative ts (from VLM cache via ctx.frame_descs).
    cache_ts: dict[str, float] = {}
    for fd in getattr(ctx, "frame_descs", []) or []:
        p = str(getattr(fd, "path", ""))
        if p:
            cache_ts[p] = float(getattr(fd, "timestamp", 0.0) or 0.0)
    frames: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("image_path") or "")
        # Path-joined cache ts is the source of truth; only fall back to the
        # LLM-supplied ts if the path is unknown to the cache.
        if path in cache_ts:
            ts = cache_ts[path]
        else:
            ts = _coerce_float(_first_present(item, ["ts", "timestamp", "time"]), 0.0)
        caption = str(item.get("caption") or item.get("description") or "").strip()
        ocr = str(item.get("ocr_text") or item.get("ocr") or "").strip()
        visual_type = str(item.get("visual_type") or infer_visual_type(caption, ocr))
        frames.append({
            "ts": _clamp(ts, 0.0, float(getattr(ctx, "duration", 0) or 0)),
            "path": path,
            "caption": caption,
            "ocr_text": ocr,
            "insight": str(item.get("insight") or item.get("selected_reason") or "").strip(),
            "visual_type": visual_type if visual_type in VisualType.__args__ else "other",
            "selected_reason": str(item.get("selected_reason") or item.get("why_useful") or "").strip(),
            "importance_score": _clamp(_coerce_float(item.get("importance_score"), 0.0), 0.0, 1.0),
        })
    return frames


def _hydrate_timeline(raw: Any, chapters: list[dict[str, Any]], mainline: list[str]) -> dict[str, Any]:
    tl = raw if isinstance(raw, dict) else {}
    boundaries = _normalise_string_list(tl.get("chapter_boundaries")) or [
        f"{ch.get('start', 0):.0f}s：进入「{ch.get('title', '')}」，作用是{ch.get('learning_goal', '')}"
        for ch in chapters
    ]
    return {
        "turning_points": _normalise_string_list(tl.get("turning_points")) or boundaries[:3],
        "argument_path": _normalise_string_list(tl.get("argument_path")) or mainline,
        "state_evolution": _normalise_string_list(tl.get("state_evolution")) or mainline[:4],
        "chapter_boundaries": boundaries,
    }


def _hydrate_completeness(raw: Any, data: dict[str, Any]) -> dict[str, Any]:
    comp = raw if isinstance(raw, dict) else {}
    has_visuals = any(v.get("selected") for v in data.get("visual_evidence", []))
    has_units = bool(data.get("knowledge_units"))
    mainline_closed = bool(comp.get("mainline_closed", data.get("chapters") and data.get("mainline")))
    return {
        "mainline_closed": bool(mainline_closed),
        "missing_prerequisites": _normalise_string_list(comp.get("missing_prerequisites")),
        "missing_steps": _normalise_string_list(comp.get("missing_steps")),
        "missing_examples": _normalise_string_list(comp.get("missing_examples")),
        "missing_boundaries": _normalise_string_list(comp.get("missing_boundaries")),
        "visual_coverage": str(comp.get("visual_coverage") or ("已覆盖关键结构画面" if has_visuals else "视觉证据不足或未识别到高价值画面")),
        "topic_reusability": bool(comp.get("topic_reusability", has_units)),
        "reader_can_understand_without_video": bool(comp.get("reader_can_understand_without_video", bool(data.get("mainline") and data.get("chapters")))),
        "notes": str(comp.get("notes") or "已检查主线、章节、视觉证据和可复用知识单元。").strip(),
    }


def _hydrate_render_plan(raw: Any, profile: dict[str, Any]) -> dict[str, Any]:
    plan = raw if isinstance(raw, dict) else {}
    primary = profile.get("primary_type")
    plan.setdefault("formula_section", primary == "technical_formula")
    plan.setdefault("code_section", primary == "technical_formula")
    plan.setdefault("procedure_section", primary == "procedural_tutorial")
    return plan


def _fallback_chapter(ctx: Any) -> dict[str, Any]:
    segments = getattr(ctx, "segments", [])
    summary = " ".join(getattr(s, "text", "") for s in segments[:8]).strip() or getattr(ctx, "title", "")
    return {"index": 1, "title": getattr(ctx, "title", "视频讲义"), "start": 0, "end": getattr(ctx, "duration", 0), "summary": summary}


def _fallback_mainline(chapters: list[dict[str, Any]]) -> list[str]:
    return [f"先理解：{ch.get('title', '')}" for ch in chapters[:6] if ch.get("title")]


def _fill_chapter_ranges(chapters: list[dict[str, Any]], duration: int) -> None:
    n = max(len(chapters), 1)
    for idx, ch in enumerate(chapters):
        if ch.get("start") is None:
            ch["start"] = duration * idx / n if duration else 0.0
        ch["start"] = _clamp(_coerce_float(ch.get("start"), 0.0), 0.0, float(duration))
    for idx, ch in enumerate(chapters):
        if ch.get("end") is None:
            next_start = chapters[idx + 1]["start"] if idx + 1 < len(chapters) else None
            ch["end"] = next_start if next_start and next_start > ch["start"] else duration * (idx + 1) / n
        ch["end"] = _clamp(_coerce_float(ch.get("end"), ch["start"]), ch["start"], float(duration))
    # LLMs sometimes truncate chapter coverage to round numbers (e.g. 600s
    # for an 889s video). If the last chapter falls more than ~10% short,
    # stretch it to the full duration so trailing content stays attributable.
    if chapters and duration > 0:
        last = chapters[-1]
        last_end = _coerce_float(last.get("end"), 0.0) or 0.0
        if last_end < float(duration) * 0.9:
            last["end"] = float(duration)


def _nearest_quote(segments: list[Any], ts: float) -> str:
    if not segments:
        return ""
    nearest = min(segments, key=lambda s: abs(float(getattr(s, "start", 0.0)) - ts))
    return str(getattr(nearest, "text", ""))[:80]


def _visual_score(visual_type: str, ocr_text: str) -> float:
    base = {"diagram": 0.86, "formula": 0.88, "code": 0.86, "table": 0.72, "ui": 0.66, "slide_text": 0.62, "person": 0.18, "other": 0.35}.get(visual_type, 0.35)
    return min(1.0, base + min(len(ocr_text) / 1000, 0.18))


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
            text = str(item.get("text") or item.get("title") or item.get("summary") or item.get("content") or "").strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def _first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _coerce_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float | None, low: float, high: float) -> float:
    if value is None:
        value = low
    return max(low, min(float(value), high))


def compact_text(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]
