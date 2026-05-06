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
    qwen_text_model: str = Field(default="qwen3-max", alias="QWEN_TEXT_MODEL")
    qwen_vl_model: str = Field(default="qwen-vl-max-latest", alias="QWEN_VL_MODEL")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    dashscope_request_timeout: float = Field(default=180.0, alias="DASHSCOPE_REQUEST_TIMEOUT")
    dashscope_trust_env: bool = Field(default=False, alias="DASHSCOPE_TRUST_ENV")
    qwen_text_enable_thinking: bool = Field(default=False, alias="QWEN_TEXT_ENABLE_THINKING")

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
    bilibili_cookie_file: Path | None = Field(
        default=Path("./data/cookies/bilibili.txt"), alias="BILIBILI_COOKIE_FILE"
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
