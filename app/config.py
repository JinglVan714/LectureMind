"""Centralised settings loader. Reads from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Models (Qwen / DashScope) ----
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    # NOTE: alias kept as QWEN_TEXT_MODEL for backward compatibility. The
    # recommended default is deepseek-v4-pro on DashScope: it produces the
    # most stable LectureIR (denser chapters / code blocks / learning paths)
    # and is the only model we currently rely on for cross-duration
    # validation. deepseek-v4-flash is acceptable as a fallback when pro
    # quota is exhausted but tends to drop high-value arrays.
    qwen_text_model: str = Field(default="deepseek-v4-pro", alias="QWEN_TEXT_MODEL")
    qwen_vl_model: str = Field(default="qwen3.6-plus", alias="QWEN_VL_MODEL")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    dashscope_request_timeout: float = Field(default=180.0, alias="DASHSCOPE_REQUEST_TIMEOUT")
    dashscope_trust_env: bool = Field(default=False, alias="DASHSCOPE_TRUST_ENV")
    qwen_text_enable_thinking: bool = Field(default=False, alias="QWEN_TEXT_ENABLE_THINKING")
    qwen_vl_concurrency: int = Field(
        default=4,
        alias="QWEN_VL_CONCURRENCY",
        ge=1,
        le=16,
        description="Concurrency for VLM frame description calls.",
    )

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
            "When true, allow the Critic-Reviser loop to actually call the "
            "Reviser to rewrite the IR based on critic issues. Defaults to "
            "false because the full-IR rewrite is the slowest stage and is "
            "the primary cause of long-video timeouts. With this off the "
            "Critic still runs and its issues are recorded for later use; "
            "turn it on for short / medium / code videos when latency is "
            "acceptable in exchange for higher coverage."
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
    qwen_copilot_model: str = Field(default="qwen3.6-plus", alias="QWEN_COPILOT_MODEL")
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
