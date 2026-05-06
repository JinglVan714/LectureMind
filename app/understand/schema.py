"""LectureJSON Pydantic models with validation."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _normalise(text: str) -> str:
    """Strip whitespace + Chinese/English punctuation for fuzzy matching."""
    return re.sub(r"[\s。，、；：！？,.;:!?\"'`~“”‘’（）()【】\[\]<>《》—\-—_…]+", "", text)


class Point(BaseModel):
    text: str = Field(..., description="精炼论点（一句话）")
    ts: float = Field(..., ge=0, description="时间戳（秒）")
    quote: str = Field(..., description="原字幕中支撑该论点的一句话")


class Frame(BaseModel):
    ts: float = Field(..., ge=0)
    path: str = Field(..., description="相对路径或绝对路径")
    caption: str = Field(default="")
    ocr_text: str = Field(default="")
    insight: str = Field(default="")
    visual_type: str = Field(default="")
    selected_reason: str = Field(default="")
    importance_score: float = Field(default=0.0, ge=0, le=1)


class Chapter(BaseModel):
    index: int = Field(..., ge=1)
    title: str
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    summary: str = Field(..., description="一段完整的中文段落，不是 bullet")
    learning_goal: str = Field(default="")
    teaching_notes: list[str] = Field(default_factory=list)
    process_steps: list[str] = Field(default_factory=list)
    points: list[Point] = Field(default_factory=list)
    frames: list[Frame] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "Chapter":
        if self.end < self.start:
            raise ValueError(f"chapter {self.index}: end {self.end} < start {self.start}")
        return self


class Highlight(BaseModel):
    text: str
    ts: float = Field(..., ge=0)


class GlossaryItem(BaseModel):
    term: str
    ts: float = Field(..., ge=0)
    explanation: str = Field(default="")


class LectureProfileView(BaseModel):
    primary_type: str = "generic_lecture"
    density: str = "standard"
    required_sections: list[str] = Field(default_factory=list)
    optional_sections: list[str] = Field(default_factory=list)


class TimelineView(BaseModel):
    turning_points: list[str] = Field(default_factory=list)
    argument_path: list[str] = Field(default_factory=list)
    state_evolution: list[str] = Field(default_factory=list)
    chapter_boundaries: list[str] = Field(default_factory=list)


class CompletenessView(BaseModel):
    mainline_closed: bool = False
    missing_prerequisites: list[str] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)
    missing_examples: list[str] = Field(default_factory=list)
    missing_boundaries: list[str] = Field(default_factory=list)
    visual_coverage: str = ""
    topic_reusability: bool = False
    reader_can_understand_without_video: bool = False
    notes: str = ""


class RenderPlanView(BaseModel):
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


class KnowledgeUnitView(BaseModel):
    id: str = ""
    type: str = "concept"
    title: str = ""
    explanation: str = ""
    ts: float = Field(default=0.0, ge=0)
    quote: str = ""
    chapter_index: int = Field(default=1, ge=1)
    related_frames: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    """Auto-tagging output produced alongside LectureIR.

    All fields default to safe empty values so that legacy LectureIR debug
    dumps (which never emitted ``taxonomy``) replay without validation
    errors.  Real-world classification is performed by the LLM and then
    repaired by ``app.copilot.taxonomy.normalize`` (white-list, suffix
    stripping, tag dedupe).
    """

    domain: str = Field(default="", description="主领域；归一化后必须落在 DOMAIN_WHITELIST")
    direction: str = Field(default="", description="该领域下的具体方向短语")
    tags: list[str] = Field(default_factory=list, description="3-8 个关键词标签")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LectureJSON(BaseModel):
    bv_id: str
    url: str = ""
    title: str
    author: str = ""
    duration: float = Field(..., ge=0)
    cover_url: str = ""
    one_liner: str = Field(..., description="一句话总结")
    category: str = "lecture"
    domain_tags: list[str] = Field(default_factory=list)
    learning_path: list[str] = Field(default_factory=list)
    chapters: list[Chapter]
    final_synthesis: str = Field(default="")
    highlights: list[Highlight] = Field(default_factory=list)
    glossary: list[GlossaryItem] = Field(default_factory=list)
    review_questions: list[str] = Field(default_factory=list)
    profile: LectureProfileView = Field(default_factory=LectureProfileView)
    core_question: str = ""
    mainline: list[str] = Field(default_factory=list)
    timeline: TimelineView = Field(default_factory=TimelineView)
    completeness: CompletenessView = Field(default_factory=CompletenessView)
    render_plan: RenderPlanView = Field(default_factory=RenderPlanView)
    knowledge_units: list[KnowledgeUnitView] = Field(default_factory=list)
    visual_evidence: list[Frame] = Field(default_factory=list)
    taxonomy: Taxonomy | None = None
    generation_mode: str = "v1"

    @field_validator("chapters")
    @classmethod
    def _has_at_least_one(cls, v: list[Chapter]) -> list[Chapter]:
        if not v:
            raise ValueError("must have at least one chapter")
        return v

    @model_validator(mode="after")
    def _check_timestamps(self) -> "LectureJSON":
        for ch in self.chapters:
            if ch.end > self.duration + 1.0:  # tolerance 1s
                raise ValueError(
                    f"chapter {ch.index} end {ch.end} > duration {self.duration}"
                )
            for p in ch.points:
                if p.ts > self.duration + 1.0:
                    raise ValueError(
                        f"point ts {p.ts} > duration {self.duration}"
                    )
            for f in ch.frames:
                if f.ts > self.duration + 1.0:
                    raise ValueError(
                        f"frame ts {f.ts} > duration {self.duration}"
                    )
        return self

    def validate_quotes(self, transcript: str) -> list[str]:
        """Return list of error messages: any quote that cannot be found
        in the transcript (after normalisation) is flagged.
        """
        errors: list[str] = []
        norm_full = _normalise(transcript)
        for ch in self.chapters:
            for p in ch.points:
                if not p.quote:
                    errors.append(f"chapter {ch.index}: point at {p.ts:.0f}s has empty quote")
                    continue
                if _normalise(p.quote) not in norm_full:
                    errors.append(
                        f"chapter {ch.index}: quote at {p.ts:.0f}s not found in transcript"
                    )
        return errors

    def mark_unverified_points(self, transcript: str) -> int:
        """Mark each point that fails quote-in-transcript with a ⚠️ prefix.
        Returns the number of points marked.
        """
        norm_full = _normalise(transcript)
        marked = 0
        for ch in self.chapters:
            for p in ch.points:
                if p.quote and _normalise(p.quote) not in norm_full:
                    if not p.text.startswith("⚠️"):
                        p.text = "⚠️ " + p.text
                    marked += 1
        return marked


def lecture_json_schema() -> dict[str, Any]:
    """Return the JSON Schema dict useful for prompt injection."""
    return LectureJSON.model_json_schema()
