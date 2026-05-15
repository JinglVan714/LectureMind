"""LectureJSON Pydantic models with validation."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Minimum quote length (after normalisation) for which we trust the
# fuzzy n-gram check to avoid both false-positives (random short
# substring matches) and false-negatives (a 4-char quote with one
# missing char failing every 4-gram). For shorter quotes we fall back
# to plain substring matching.
_FUZZY_MIN_LEN = 6
_FUZZY_NGRAM = 4
_FUZZY_THRESHOLD = 0.6


def _normalise(text: str) -> str:
    """Strip whitespace + Chinese/English punctuation for fuzzy matching."""
    return re.sub(r"[\s。，、；：！？,.;:!?\"'`~“”‘’（）()【】\[\]<>《》—\-—_…]+", "", text)


def _ngrams(text: str, n: int = _FUZZY_NGRAM) -> set[str]:
    """Return the multiset (as a plain set, dedup) of character n-grams.

    Works for both Chinese (each char ≈ token) and English / mixed
    content. Sets are intentional — repeating n-grams should not
    inflate the overlap ratio for short quotes.
    """
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _quote_in_transcript(quote: str, transcript: str, transcript_norm: str) -> bool:
    """Membership check that tolerates light paraphrase.

    1. Normalise (strip whitespace + punct).
    2. Fast path: exact substring (handles correctly-quoted CC).
    3. For quotes longer than ``_FUZZY_MIN_LEN``, accept when the
       4-gram overlap ratio with the transcript is ≥ ``_FUZZY_THRESHOLD``.
       This catches LLM paraphrases that reorder a few words or drop
       filler particles ("嗯/呃/那么/我们") without inviting random
       substring noise.

    ``transcript_norm`` is precomputed by the caller to avoid O(N²)
    work when validating dozens of quotes.
    """
    quote_norm = _normalise(quote)
    if not quote_norm:
        return False
    if quote_norm in transcript_norm:
        return True
    if len(quote_norm) < _FUZZY_MIN_LEN:
        return False
    quote_grams = _ngrams(quote_norm)
    transcript_grams = _ngrams(transcript_norm)
    if not quote_grams:
        return False
    overlap = len(quote_grams & transcript_grams) / len(quote_grams)
    return overlap >= _FUZZY_THRESHOLD


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


class CodeBlock(BaseModel):
    """A code excerpt extracted from a frame OCR or narrated by the speaker.

    Kept first-class (instead of being squashed into a teaching-note
    string) so:

    * Indentation / newlines survive a JSON round-trip.
    * The renderer can put each block in its own ``<pre><code>`` panel
      with syntax highlighting.
    * The RAG indexer can give code its own chunk kind.
    """

    language: str = Field(default="", description="python / rust / cpp / shell / sql / yaml ...")
    code: str = Field(..., description="原样保留缩进与换行的代码")
    ts: float = Field(default=0.0, ge=0)
    chapter_index: int = Field(default=1, ge=1)
    source: str = Field(default="ocr", description="ocr | narrated | reconstructed")
    explanation: str = Field(default="", description="一句话说明这段代码做了什么")
    related_frame_paths: list[str] = Field(default_factory=list)


class FormulaBlock(BaseModel):
    """A LaTeX-rendered formula that deserves its own callout."""

    latex: str = Field(..., description="必须使用 KaTeX 兼容的 LaTeX 语法")
    ts: float = Field(default=0.0, ge=0)
    chapter_index: int = Field(default=1, ge=1)
    explanation: str = Field(default="")
    related_frame_paths: list[str] = Field(default_factory=list)


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
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    formula_blocks: list[FormulaBlock] = Field(default_factory=list)

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


class SupportingVisualView(BaseModel):
    visual_role: str = "keyframe"
    source_mode: str = "video_frame"
    path: str = ""
    caption: str = ""
    ts: float = Field(default=0.0, ge=0)


class OutlineBlockView(BaseModel):
    id: str = ""
    ordinal: int = Field(..., ge=1)
    title: str = ""
    lead: str = ""
    paragraphs: list[str] = Field(default_factory=list)
    source_chapter_refs: list[int] = Field(default_factory=list)
    source_timestamps: list[float] = Field(default_factory=list)
    block_role: str = ""
    topic_hint: str = ""
    source_hint: str = ""


class BodySectionView(BaseModel):
    id: str
    title: str
    section_role: str = "concept"
    summary: str = ""
    paragraphs: list[str] = Field(default_factory=list)
    outline_blocks: list[OutlineBlockView] = Field(default_factory=list)
    supporting_visuals: list[SupportingVisualView] = Field(default_factory=list)
    source_chapter_refs: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_outline_blocks(self) -> "BodySectionView":
        if not self.outline_blocks and self.paragraphs:
            self.outline_blocks = [
                OutlineBlockView(
                    id=f"{self.id}-block-1" if self.id else "block-1",
                    ordinal=1,
                    title=self.title,
                    paragraphs=list(self.paragraphs),
                    source_chapter_refs=list(self.source_chapter_refs),
                )
            ]
        if not self.paragraphs and self.outline_blocks:
            flattened: list[str] = []
            for block in self.outline_blocks:
                if block.lead:
                    flattened.append(block.lead)
                flattened.extend(block.paragraphs)
            self.paragraphs = flattened
        for index, block in enumerate(self.outline_blocks, start=1):
            if not block.id:
                self.outline_blocks[index - 1] = block.model_copy(
                    update={"id": f"{self.id}-block-{index}" if self.id else f"block-{index}"}
                )
        return self


class CompositionView(BaseModel):
    burden_score: float = Field(default=0.0, ge=0.0, le=1.0)
    burden_signals: dict[str, Any] = Field(default_factory=dict)
    summary_mode: str = "note"
    semantic_profile: str = ""
    composition_profile: str = "conceptual"
    reorder_strength: str = "light"
    reading_goal: str = ""
    hero_summary: str = ""
    key_takeaways_top: list[str] = Field(default_factory=list)
    audience_fit: str = ""
    body_sections: list[BodySectionView] = Field(default_factory=list)
    source_index: dict[str, Any] = Field(default_factory=dict)
    visual_plan: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class LectureFrontMatterPlan(BaseModel):
    one_sentence_claim: str = ""
    reader_orientation: str = ""
    takeaways_top: list[str] = Field(default_factory=list)
    reading_map: list[str] = Field(default_factory=list)
    reader_prerequisites: list[str] = Field(default_factory=list)
    suitable_for: list[str] = Field(default_factory=list)
    not_suitable_for: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class BodyUnitPlan(BaseModel):
    unit_id: str
    title: str = ""
    teaching_goal: str = ""
    unit_role: str = "claim"
    core_message: str = ""
    transition_from_previous: str = ""
    source_chapter_refs: list[int] = Field(default_factory=list)
    visual_needs: list[dict[str, Any]] = Field(default_factory=list)
    appendix_candidates: list[str] = Field(default_factory=list)


class LectureBackMatterPlan(BaseModel):
    boundary_and_risks: list[str] = Field(default_factory=list)
    source_index_entrypoints: list[str] = Field(default_factory=list)
    appendices: list[dict[str, Any]] = Field(default_factory=list)
    transfer_and_next_steps: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class LectureBlueprint(BaseModel):
    front_matter_plan: LectureFrontMatterPlan = Field(default_factory=LectureFrontMatterPlan)
    body_unit_plan: list[BodyUnitPlan] = Field(default_factory=list)
    back_matter_plan: LectureBackMatterPlan = Field(default_factory=LectureBackMatterPlan)
    reorder_strength: str = "light"
    visual_needs: list[dict[str, Any]] = Field(default_factory=list)
    appendix_candidates: list[dict[str, Any]] = Field(default_factory=list)
    semantic_portrait: str = ""
    reading_burden: str = ""
    teaching_center_of_gravity: str = ""


class VisualSlot(BaseModel):
    slot_id: str
    visual_role: str = "keyframe_explainer"
    title: str = ""
    caption: str = ""
    source_paths: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    ts: float | None = Field(default=None, ge=0)


class ContentBlock(BaseModel):
    block_id: str
    title: str = ""
    block_role: str = ""
    lead: str = ""
    paragraphs: list[str] = Field(default_factory=list)
    transition: str = ""
    source_chapter_refs: list[int] = Field(default_factory=list)
    source_timestamps: list[float] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class TeachingUnit(BaseModel):
    unit_id: str
    ordinal: str = ""
    title: str = ""
    teaching_goal: str = ""
    unit_role: str = "claim"
    core_message: str = ""
    transition_from_previous: str = ""
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    visual_slots: list[VisualSlot] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source_chapter_refs: list[int] = Field(default_factory=list)


class LectureNoteFrontMatter(BaseModel):
    one_sentence_claim: str = ""
    reader_orientation: str = ""
    takeaways_top: list[str] = Field(default_factory=list)
    reading_map: list[str] = Field(default_factory=list)
    reader_prerequisites: list[str] = Field(default_factory=list)
    suitable_for: list[str] = Field(default_factory=list)
    not_suitable_for: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class LectureNoteBody(BaseModel):
    teaching_units: list[TeachingUnit] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class LectureNoteBackMatter(BaseModel):
    boundary_and_risks: list[str] = Field(default_factory=list)
    term_quick_ref: list[dict[str, Any]] = Field(default_factory=list)
    source_index_entrypoints: list[str] = Field(default_factory=list)
    appendices: list[dict[str, Any]] = Field(default_factory=list)
    transfer_and_next_steps: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class LectureNoteIR(BaseModel):
    front_matter: LectureNoteFrontMatter = Field(default_factory=LectureNoteFrontMatter)
    body: LectureNoteBody = Field(default_factory=LectureNoteBody)
    back_matter: LectureNoteBackMatter = Field(default_factory=LectureNoteBackMatter)


class RagChunk(BaseModel):
    kind: str = "teaching_note"
    chapter_idx: int = Field(default=1, ge=1)
    t_start: float = Field(default=0.0, ge=0)
    t_end: float = Field(default=0.0, ge=0)
    text: str = ""
    note_node_id: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class EvidenceObject(BaseModel):
    evidence_id: str
    kind: str
    title: str = ""
    summary: str = ""
    chapter_index: int | None = Field(default=None, ge=1)
    quote: str = ""
    text: str = ""
    ts: float | None = Field(default=None, ge=0)
    t_end: float | None = Field(default=None, ge=0)
    path: str = ""
    note_node_ids: list[str] = Field(default_factory=list)
    anchors: dict[str, Any] = Field(default_factory=dict)
    rag_chunks: list[RagChunk] = Field(default_factory=list)
    source_payload: dict[str, Any] = Field(default_factory=dict)


class EvidenceRelation(BaseModel):
    relation: str
    from_id: str
    to_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceIndex(BaseModel):
    evidence_objects: list[EvidenceObject] = Field(default_factory=list)
    evidence_relations: list[EvidenceRelation] = Field(default_factory=list)
    anchor_map: dict[str, Any] = Field(default_factory=dict)
    projection_views: dict[str, Any] = Field(default_factory=dict)


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
    composition: CompositionView = Field(default_factory=CompositionView)
    lecture_blueprint: LectureBlueprint | None = None
    lecture_note_ir: LectureNoteIR | None = None
    evidence_index: EvidenceIndex | None = None
    study_questions: list[str] = Field(
        default_factory=list,
        description="Question-Driven 抽取阶段产出的学习问题；驱动结构化抽取并展示在 HTML",
    )
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
        in the transcript (after fuzzy n-gram matching) is flagged.

        Uses :func:`_quote_in_transcript`, which accepts either an exact
        substring (post-normalisation) or a 4-gram overlap ratio ≥ 0.6.
        That stops the LLM-paraphrase false-positives that earlier
        flooded long videos with ⚠️.
        """
        errors: list[str] = []
        norm_full = _normalise(transcript)
        for ch in self.chapters:
            for p in ch.points:
                if not p.quote:
                    errors.append(f"chapter {ch.index}: point at {p.ts:.0f}s has empty quote")
                    continue
                if not _quote_in_transcript(p.quote, transcript, norm_full):
                    errors.append(
                        f"chapter {ch.index}: quote at {p.ts:.0f}s not found in transcript"
                    )
        return errors

    def mark_unverified_points(self, transcript: str) -> int:
        """Mark each point that fails fuzzy quote-in-transcript with a ⚠️ prefix.

        Returns the number of points marked.
        """
        norm_full = _normalise(transcript)
        marked = 0
        for ch in self.chapters:
            for p in ch.points:
                if not p.quote:
                    continue
                if not _quote_in_transcript(p.quote, transcript, norm_full):
                    if not p.text.startswith("⚠️"):
                        p.text = "⚠️ " + p.text
                    marked += 1
        return marked


def lecture_json_schema() -> dict[str, Any]:
    """Return the JSON Schema dict useful for prompt injection."""
    return LectureJSON.model_json_schema()
