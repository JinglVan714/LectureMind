"""Centralised settings loader. Reads from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Allow constructor kwargs by field name (in addition to alias)
        # so tests and internal callers can build Settings without
        # remembering every uppercase env-var name.
        populate_by_name=True,
    )

    # ---- Models ----
    # We run a split stack:
    #   * Text + Copilot  -> DeepSeek official API (OpenAI-compatible).
    #   * Vision (VLM)    -> DashScope Qwen native multimodal.
    #   * Embeddings      -> DashScope tongyi-embedding (native SDK).
    # This lets us cut text cost with DeepSeek V4 pricing while keeping
    # Qwen's solid vision + multi-modal embedding quality.

    # -- DeepSeek (text + copilot) --
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_request_timeout: float = Field(
        default=180.0, alias="DEEPSEEK_REQUEST_TIMEOUT"
    )
    deepseek_trust_env: bool = Field(default=False, alias="DEEPSEEK_TRUST_ENV")

    # -- DashScope (VLM + embeddings) --
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    # NOTE: alias kept as QWEN_TEXT_MODEL for backward compatibility even
    # though the default now points at DeepSeek. The Critic / Reviser /
    # StudyQuestion / IR builder all consume this name against the DeepSeek
    # endpoint (see ``deepseek_*`` above). deepseek-v4-flash is the chosen
    # baseline: it is dramatically cheaper than v4-pro and, per user
    # regression on BV1NM1tY3Eu5, delivers acceptable IR density when paired
    # with the existing Critic + coverage warnings.
    qwen_text_model: str = Field(default="deepseek-v4-flash", alias="QWEN_TEXT_MODEL")
    # Vision model used for keyframe captioning + board-OCR. Stays on
    # DashScope because DeepSeek V4 does not (officially) expose a vision
    # endpoint yet. qwen3.5-omni-plus is the native-multimodal Qwen model
    # the user selected for this migration.
    qwen_vl_model: str = Field(default="qwen3.5-omni-plus", alias="QWEN_VL_MODEL")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    dashscope_request_timeout: float = Field(default=180.0, alias="DASHSCOPE_REQUEST_TIMEOUT")
    dashscope_trust_env: bool = Field(default=False, alias="DASHSCOPE_TRUST_ENV")
    # Legacy flag — only honoured when the text model is clearly a Qwen
    # model (base URL contains ``dashscope``). DeepSeek uses its own
    # thinking-mode mechanism (model suffix) and would 400 on this extra
    # body key, so it is suppressed automatically for DeepSeek traffic.
    qwen_text_enable_thinking: bool = Field(default=False, alias="QWEN_TEXT_ENABLE_THINKING")
    qwen_vl_concurrency: int = Field(
        default=4,
        alias="QWEN_VL_CONCURRENCY",
        ge=1,
        le=16,
        description="Concurrency for VLM frame description calls.",
    )

    # ---- VLM tiering & cache (M3) ----
    # Two-tier prompting: HIGH-tier frames get the full caption+OCR+
    # scoring prompt (~1k input + ~200 output tokens). LOW-tier frames
    # — repeated slides, talking heads, near-blank transitions —
    # collapse to a much cheaper OCR-only prompt. The mixed strategy
    # below combines an absolute junk filter (drop near-duplicates and
    # near-blank frames outright) with a percentage floor (always keep
    # at least HIGH_FLOOR_RATIO of survivors as HIGH so visually-rich
    # videos do not lose information just because every frame "looks
    # similar"). Defaults match the M3 spec; ``VLM_TIERING_ENABLED=
    # false`` is the emergency escape hatch back to legacy behaviour.
    vlm_tiering_enabled: bool = Field(default=True, alias="VLM_TIERING_ENABLED")
    vlm_tiering_high_floor_ratio: float = Field(
        default=0.5,
        alias="VLM_TIERING_HIGH_FLOOR_RATIO",
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of non-junk frames that must remain HIGH tier "
            "regardless of score. Set to 1.0 to disable downgrading."
        ),
    )
    vlm_junk_sim_threshold: float = Field(
        default=0.93,
        alias="VLM_JUNK_SIM_THRESHOLD",
        ge=0.0,
        le=1.0,
        description=(
            "If a frame's dHash similarity to the previous frame is "
            "at or above this value, it is forced into LOW tier."
        ),
    )
    vlm_junk_entropy_threshold: float = Field(
        default=2.0,
        alias="VLM_JUNK_ENTROPY_THRESHOLD",
        ge=0.0,
        description=(
            "Frames whose grayscale Shannon entropy is below this "
            "value (range [0, 8]) are forced into LOW tier."
        ),
    )
    # SQLite-backed content-addressed cache. Key is
    # (sha256(image_bytes), model, tier) so repeated slides across
    # videos hit. Setting ``VLM_CACHE_ENABLED=false`` bypasses the
    # cache entirely (useful for A/B). ``VLM_CACHE_PATH`` defaults to
    # ``<DATA_DIR>/vlm_cache.sqlite`` when blank.
    vlm_cache_enabled: bool = Field(default=True, alias="VLM_CACHE_ENABLED")
    vlm_cache_path: str = Field(default="", alias="VLM_CACHE_PATH")

    # ---- Whisper fallback ----
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")

    # ---- Auth ----
    basic_auth_user: str = Field(default="admin", alias="BASIC_AUTH_USER")
    basic_auth_password: str = Field(default="change-me", alias="BASIC_AUTH_PASSWORD")

    # ---- Service ----
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    max_concurrent_jobs: int = Field(default=2, alias="MAX_CONCURRENT_JOBS")

    # ---- Ingest ----
    keyframe_min: int = Field(default=8, alias="KEYFRAME_MIN")
    keyframe_max: int = Field(default=20, alias="KEYFRAME_MAX")
    keyframe_threshold: float = Field(default=27.0, alias="KEYFRAME_THRESHOLD")
    keyframe_length_adapt: bool = Field(
        default=True,
        alias="KEYFRAME_LENGTH_ADAPT",
        description="Scale keyframe min/max with video duration (recommended).",
    )
    bilibili_cookie_file: Path | None = Field(
        default=Path("./data/cookies/bilibili.txt"), alias="BILIBILI_COOKIE_FILE"
    )

    # ---- Lecture generation v2 (multi-agent + question-driven) ----
    lecture_question_driven: bool = Field(
        default=True,
        alias="LECTURE_QUESTION_DRIVEN",
        description="Pre-generate study questions to drive structured extraction.",
    )
    lecture_study_question_multi_window: bool = Field(
        default=True,
        alias="LECTURE_STUDY_QUESTION_MULTI_WINDOW",
        description=(
            "When true (default), the StudyQuestionAgent samples "
            "subtitles and frames from multiple windows (head + mid + "
            "tail) instead of only the first ~5 minutes, so questions "
            "for long videos cover the whole timeline. Set false to "
            "recover the legacy head-only sampling for A/B testing."
        ),
    )
    lecture_critic_enabled: bool = Field(
        default=True,
        alias="LECTURE_CRITIC_ENABLED",
        description=(
            "Run the Critic stage after the initial IR build. The Critic "
            "audits coverage, missing code/formula and dubious quotes and "
            "writes its findings into pipeline_stats.critique. It does NOT "
            "rewrite the IR by itself — see lecture_reviser_enabled."
        ),
    )
    lecture_reviser_enabled: bool = Field(
        default=False,
        alias="LECTURE_REVISER_ENABLED",
        description=(
            "Legacy on/off switch for the Critic-Reviser loop. Kept for "
            "back-compat — prefer ``LECTURE_REVISER_MODE``. When "
            "``LECTURE_REVISER_MODE`` is *not* explicitly set, this flag "
            "is mapped at load-time to ``mode='full'`` (true) or "
            "``mode='off'`` (false). When ``LECTURE_REVISER_MODE`` is "
            "set explicitly it always wins."
        ),
    )
    lecture_reviser_mode: Literal["off", "patch", "full"] = Field(
        default="off",
        alias="LECTURE_REVISER_MODE",
        description=(
            "Reviser dispatch mode (M2). 'off' disables the Reviser "
            "while still letting the Critic audit the IR. 'patch' "
            "applies a whitelist of structured edits returned by the "
            "Reviser (introduced in M2.P6). 'full' rewrites the entire "
            "IR (legacy behaviour, slowest path). Default 'off' "
            "preserves M1 behaviour; M2.P6 will flip the recommended "
            "default to 'patch'."
        ),
    )

    # ---- M2 length-aware routing ----
    # Boundary policy: thresholds are exclusive on the lower side, so a
    # video with duration_sec == tiny_th maps to ``standard``. Set via
    # CSV string in .env (e.g. ``LECTURE_PROFILE_THRESHOLDS_SEC=180,1500,3600``).
    # ``NoDecode`` disables pydantic-settings' default JSON-decoding for
    # this complex type so the raw CSV string from .env reaches our
    # ``_parse_profile_thresholds`` validator intact (otherwise
    # ``"180,1500,3600"`` would explode in ``json.loads``).
    lecture_profile_thresholds_sec: Annotated[
        tuple[int, int, int], NoDecode
    ] = Field(
        default=(180, 1500, 3600),
        alias="LECTURE_PROFILE_THRESHOLDS_SEC",
        description=(
            "Three ascending integers (seconds) splitting tiny / "
            "standard / long / epic profiles. Default (180, 1500, 3600) "
            "= (<3min / <25min / <60min / 60+min). Override via CSV "
            "string."
        ),
    )

    # ---- M2.P2 ChapterPlanner ----
    # Deterministic chapter-anchor extraction. The planner is on by
    # default but only fires for videos >= ``lecture_chapter_planner_min_sec``;
    # below that floor a single-chapter lecture is the right answer.
    lecture_chapter_planner_enabled: bool = Field(
        default=True,
        alias="LECTURE_CHAPTER_PLANNER_ENABLED",
        description=(
            "Master switch for the deterministic ChapterPlanner. "
            "When False, ``plan_chapters`` always returns ``[]`` and "
            "downstream stages fall back to LLM-only chapter inference."
        ),
    )
    lecture_chapter_planner_min_sec: int = Field(
        default=60,
        alias="LECTURE_CHAPTER_PLANNER_MIN_SEC",
        ge=0,
        description=(
            "Minimum video duration (seconds) required to run the "
            "ChapterPlanner. Videos shorter than this skip planning "
            "and ship a single implicit chapter."
        ),
    )
    lecture_planner_silence_sec: float = Field(
        default=6.0,
        alias="LECTURE_PLANNER_SILENCE_SEC",
        gt=0.0,
        description=(
            "Inter-segment gap (seconds) at or above which a silent "
            "transition is treated as a chapter-break candidate."
        ),
    )
    lecture_planner_transition_phrase_confidence: float = Field(
        default=0.85,
        alias="LECTURE_PLANNER_TRANSITION_PHRASE_CONFIDENCE",
        ge=0.0,
        le=1.0,
        description=(
            "Default confidence for chapter anchors found via the "
            "transition-phrase regex (e.g. '接下来我们看', '总结'). "
            "Strongest of the 3 deterministic signals."
        ),
    )
    lecture_planner_silence_confidence: float = Field(
        default=0.6,
        alias="LECTURE_PLANNER_SILENCE_CONFIDENCE",
        ge=0.0,
        le=1.0,
        description=(
            "Default confidence for chapter anchors found via silent "
            "gaps in the subtitle stream."
        ),
    )
    lecture_planner_visual_shift_confidence: float = Field(
        default=0.55,
        alias="LECTURE_PLANNER_VISUAL_SHIFT_CONFIDENCE",
        ge=0.0,
        le=1.0,
        description=(
            "Default confidence for chapter anchors found via visual "
            "type changes in adjacent keyframes (weakest of the 3 "
            "signals; cross-checked against subtitle boundaries)."
        ),
    )

    # ---- M2.P3 ChapterCache (SQLite, prompt_hash keyed) ----
    # Per-chapter LLM result cache used by ``MapReduceIRBuilder`` (P4).
    # Key is a SHA-256 of (segments + frames + boundaries + model + prompt
    # version), so editing a single subtitle line invalidates only the
    # affected chapter, while bumping ``LECTURE_MAP_PROMPT_VERSION``
    # invalidates the whole table at once. Disabling the cache turns
    # ``ChapterCache.put/get`` into no-ops without touching the filesystem
    # — useful for A/B and emergency disable.
    lecture_chapter_cache_enabled: bool = Field(
        default=True,
        alias="LECTURE_CHAPTER_CACHE_ENABLED",
        description=(
            "Master switch for the per-chapter SQLite cache used by "
            "the map-reduce IR builder. Default on; off makes "
            "ChapterCache a no-op without touching disk."
        ),
    )
    lecture_chapter_cache_path: str = Field(
        default="",
        alias="LECTURE_CHAPTER_CACHE_PATH",
        description=(
            "Override the chapter-cache SQLite path. Defaults to "
            "<DATA_DIR>/chapter_cache.sqlite when blank."
        ),
    )
    lecture_map_prompt_version: str = Field(
        default="m2-map-v1",
        alias="LECTURE_MAP_PROMPT_VERSION",
        description=(
            "Version tag mixed into every chapter prompt_hash. Bump "
            "this whenever the map-chapter prompt template is "
            "edited so all stale rows fall through to a miss."
        ),
    )

    lecture_critic_max_rounds: int = Field(
        default=1,
        alias="LECTURE_CRITIC_MAX_ROUNDS",
        ge=0,
        le=3,
    )
    lecture_critic_timeout: float = Field(
        default=300.0,
        alias="LECTURE_CRITIC_TIMEOUT",
        description=(
            "Timeout for the Critic LLM call. The full subtitle + frames + "
            "draft IR can be large for long videos, so this defaults to a "
            "generous 5 minutes (vs the global 180s default). Override "
            "with LECTURE_CRITIC_TIMEOUT."
        ),
    )
    # M2.P5 per-profile Critic timeouts. Standard profile keeps using
    # the legacy ``lecture_critic_timeout`` (300s) for backward compat
    # — the M1 critique() path is unchanged in P5. The two new knobs
    # below are consumed by the upcoming :func:`audit` entrypoint
    # wired from P7 onwards: full-mode tracks the legacy budget, while
    # projected-mode is leaner because the prompt drops segments /
    # frames / point-quote bodies (spec §3.5).
    lecture_critic_full_timeout: float = Field(
        default=180.0,
        alias="LECTURE_CRITIC_FULL_TIMEOUT",
        description=(
            "Timeout for the full-mode Critic LLM call (standard "
            "profile). Spec §3.5 default 180s; consumed by P7 "
            "wiring, not the legacy critique() path."
        ),
    )
    lecture_critic_projected_timeout: float = Field(
        default=120.0,
        alias="LECTURE_CRITIC_PROJECTED_TIMEOUT",
        description=(
            "Timeout for the projected-mode Critic LLM call "
            "(long / epic profiles). Spec §3.5 default 120s; the "
            "projection drops segments / frames / point quotes so "
            "the budget is smaller than full-mode by design."
        ),
    )
    lecture_critic_max_prompt_chars: int = Field(
        default=24000,
        alias="LECTURE_CRITIC_MAX_PROMPT_CHARS",
        ge=0,
        description=(
            "Soft cap on the Critic **subtitle excerpt** size in "
            "characters (NOT the total prompt — the IR JSON is the "
            "artefact under audit and is irreducible). When > 0 the "
            "Critic projects subtitles against the IR structure "
            "(chapter windows + point/code/formula/visual anchor "
            "neighborhoods) and uniformly downsamples segments only "
            "when the formatted subtitle block would still exceed this "
            "budget. Frames are independently capped at 16 (short / "
            "medium) or 24 (long) and prioritise high-value visual "
            "types. Set to 0 to disable both projection and trimming "
            "and recover the legacy full-subtitle behaviour."
        ),
    )
    lecture_reviser_timeout: float = Field(default=600.0, alias="LECTURE_REVISER_TIMEOUT")
    lecture_strict_agents: bool = Field(default=False, alias="LECTURE_STRICT_AGENTS")
    lecture_code_highlighter: str = Field(
        default="highlight.js",
        alias="LECTURE_CODE_HIGHLIGHTER",
        description="'highlight.js', 'prism' or '' to disable JS highlighting.",
    )
    lecture_map_reduce_threshold_sec: int = Field(
        default=1200,
        alias="LECTURE_MAP_REDUCE_THRESHOLD_SEC",
        description="Above this duration, future map-reduce stages activate.",
    )
    pipeline_wait_rag: bool = Field(
        default=True,
        alias="PIPELINE_WAIT_RAG",
        description=(
            "When true (default), pipeline.run() awaits RAG indexing before "
            "returning. Set false to make RAG indexing a fire-and-forget "
            "background task so the report path returns sooner."
        ),
    )

    # ---- Logging ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ---- Copilot + MCP (v1) ----
    # Copilot shares the DeepSeek endpoint with the text pipeline unless
    # explicitly overridden. deepseek-v4-flash is a good default: cheap,
    # low latency, and tool-use capable per DeepSeek V4 release notes.
    qwen_copilot_model: str = Field(default="deepseek-v4-flash", alias="QWEN_COPILOT_MODEL")
    qwen_embedding_model: str = Field(
        default="tongyi-embedding-vision-flash-2026-03-06",
        alias="QWEN_EMBEDDING_MODEL",
    )
    copilot_frame_vision_embed: bool = Field(
        default=False, alias="COPILOT_FRAME_VISION_EMBED"
    )
    copilot_max_tool_calls: int = Field(default=6, alias="COPILOT_MAX_TOOL_CALLS")
    copilot_max_concurrent: int = Field(default=4, alias="COPILOT_MAX_CONCURRENT")
    bocha_api_key: str = Field(default="", alias="BOCHA_API_KEY")
    bocha_base_url: str = Field(default="https://api.bochaai.com/v1/web-search", alias="BOCHA_BASE_URL")
    mcp_server_token: str = Field(default="", alias="MCP_SERVER_TOKEN")
    mcp_expose_summarize: bool = Field(default=False, alias="MCP_EXPOSE_SUMMARIZE")

    # ---- Validators ----
    @field_validator("lecture_profile_thresholds_sec", mode="before")
    @classmethod
    def _parse_profile_thresholds(cls, v):
        """Accept either a CSV string from .env or a native tuple/list.

        Pydantic-settings would otherwise try to JSON-decode tuples and
        reject ``"180,1500,3600"`` outright. We normalise to a 3-tuple
        of strictly ascending ints here so downstream callers can rely
        on the invariant.
        """
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if len(parts) != 3:
                raise ValueError(
                    "LECTURE_PROFILE_THRESHOLDS_SEC must be 3 comma-separated"
                    f" ints, got {v!r}"
                )
            try:
                v = tuple(int(p) for p in parts)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "LECTURE_PROFILE_THRESHOLDS_SEC contains a non-integer"
                    f" segment: {v!r}"
                ) from exc
        if isinstance(v, (list, tuple)):
            if len(v) != 3:
                raise ValueError(
                    "LECTURE_PROFILE_THRESHOLDS_SEC must contain exactly 3"
                    f" values, got {len(v)}"
                )
            ints = tuple(int(x) for x in v)
            if not (ints[0] < ints[1] < ints[2]):
                raise ValueError(
                    "LECTURE_PROFILE_THRESHOLDS_SEC must be strictly"
                    f" ascending, got {ints}"
                )
            return ints
        return v

    @model_validator(mode="after")
    def _apply_reviser_mode_compat(self) -> "Settings":
        """Map legacy ``LECTURE_REVISER_ENABLED`` to ``LECTURE_REVISER_MODE``.

        Only applies when ``lecture_reviser_mode`` was *not* explicitly
        set (env var or constructor kwarg). When the user has set the
        new field directly, it always wins. The mapping is::

            enabled=True  & mode unset → mode='full'  (legacy behaviour)
            enabled=False & mode unset → mode='off'   (legacy behaviour)
        """
        if "lecture_reviser_mode" not in self.model_fields_set:
            new_mode = "full" if self.lecture_reviser_enabled else "off"
            # Settings is mutable (BaseSettings is not frozen) so a plain
            # attribute assignment is enough; ``object.__setattr__`` keeps
            # us safe even if someone freezes the model later.
            object.__setattr__(self, "lecture_reviser_mode", new_mode)
        return self

    # ---- Derived helpers ----
    def _is_dashscope_url(self, url: str) -> bool:
        return "dashscope" in (url or "").lower()

    def text_extra_body(self) -> dict[str, bool]:
        """Extra body for text LLM calls.

        Only emits ``{"enable_thinking": ...}`` when the **text client** is
        pointing at a DashScope Qwen endpoint — DeepSeek toggles thinking
        mode via the model name suffix and rejects this field, so we must
        not send it on DeepSeek traffic.
        """
        if self._is_dashscope_url(self.deepseek_base_url):
            return {"enable_thinking": self.qwen_text_enable_thinking}
        return {}

    # ---- Derived paths ----
    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def keyframes_dir(self) -> Path:
        return self.data_dir / "keyframes"

    @property
    def subtitles_dir(self) -> Path:
        return self.data_dir / "subtitles"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def cookies_dir(self) -> Path:
        return self.data_dir / "cookies"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.reports_dir,
            self.keyframes_dir,
            self.subtitles_dir,
            self.audio_dir,
            self.cookies_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
