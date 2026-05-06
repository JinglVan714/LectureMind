"""Subtitle extraction. CC subtitle preferred; faster-whisper fallback."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import get_settings
from .cookies import cookie_file_path, load_cookie_jar

logger = logging.getLogger(__name__)


def _local_faster_whisper_base_snapshot(home: Path | None = None) -> str | None:
    snapshots_dir = (
        (home or Path.home())
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Systran--faster-whisper-base"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if not snapshots:
            return None
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    except OSError:
        return None


@dataclass
class SubtitleSegment:
    start: float  # seconds
    end: float
    text: str


@dataclass
class SubtitleResult:
    source: str  # "cc" | "whisper"
    language: str
    segments: list[SubtitleSegment]
    raw_path: Path  # where we cached the raw JSON / SRT


class SubtitleExtractor:
    """Get a normalised subtitle list for a Bilibili video.

    Strategy:
      1. Try the player-info endpoint to find an official CC subtitle URL.
         If found, fetch the JSON directly.
      2. Otherwise download the audio with yt-dlp + ffmpeg, then transcribe
         with faster-whisper. Whisper is imported lazily so users without
         the optional dep can still run pure-CC mode.
    """

    PLAYER_INFO_URL = "https://api.bilibili.com/x/player/wbi/v2"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        }
        self._cookies = load_cookie_jar(self._settings.bilibili_cookie_file)
        self._cookie_file = cookie_file_path(self._settings.bilibili_cookie_file)

    async def extract(self, bv_id: str, aid: int, cid: int) -> SubtitleResult:
        """Try CC then fallback to Whisper."""
        try:
            cc = await self._fetch_cc(bv_id, aid, cid)
            if cc:
                return cc
        except Exception as exc:  # noqa: BLE001
            logger.warning("CC subtitle fetch failed for %s: %s", bv_id, exc)

        cached = self._load_cached_whisper(bv_id)
        if cached:
            logger.info("Reusing cached Whisper subtitle for %s", bv_id)
            return cached

        logger.info("Falling back to Whisper for %s", bv_id)
        return await self._whisper_fallback(bv_id)

    # ---------------- CC path ----------------

    async def _fetch_cc(self, bv_id: str, aid: int, cid: int) -> SubtitleResult | None:
        params = {"aid": aid, "cid": cid, "bvid": bv_id}
        async with httpx.AsyncClient(
            timeout=15.0, headers=self._headers, cookies=self._cookies
        ) as client:
            resp = await client.get(self.PLAYER_INFO_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("code") != 0:
                return None
            subs = payload.get("data", {}).get("subtitle", {}).get("subtitles", []) or []
            if not subs:
                return None

            # Prefer Chinese, otherwise the first available track
            chosen = next(
                (s for s in subs if "zh" in (s.get("lan") or "").lower()),
                subs[0],
            )
            sub_url = chosen.get("subtitle_url") or ""
            if sub_url.startswith("//"):
                sub_url = "https:" + sub_url
            if not sub_url:
                return None

            sub_resp = await client.get(sub_url)
            sub_resp.raise_for_status()
            sub_json = sub_resp.json()

        body = sub_json.get("body") or []
        segments = [
            SubtitleSegment(
                start=float(item.get("from", 0.0)),
                end=float(item.get("to", 0.0)),
                text=(item.get("content") or "").strip(),
            )
            for item in body
            if (item.get("content") or "").strip()
        ]
        if not segments:
            return None

        raw_path = self._settings.subtitles_dir / f"{bv_id}.cc.json"
        raw_path.write_text(json.dumps(sub_json, ensure_ascii=False), encoding="utf-8")
        return SubtitleResult(
            source="cc",
            language=chosen.get("lan", "unknown"),
            segments=segments,
            raw_path=raw_path,
        )

    # ---------------- Whisper path ----------------

    def _load_cached_whisper(self, bv_id: str) -> SubtitleResult | None:
        raw_path = self._settings.subtitles_dir / f"{bv_id}.whisper.json"
        if not raw_path.exists():
            return None
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid cached Whisper subtitle %s: %s", raw_path, exc)
            return None
        if not isinstance(raw, list):
            return None
        segments = [
            SubtitleSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=str(item.get("text") or "").strip(),
            )
            for item in raw
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        if not segments:
            return None
        return SubtitleResult(
            source="whisper",
            language="auto",
            segments=segments,
            raw_path=raw_path,
        )

    async def _whisper_fallback(self, bv_id: str) -> SubtitleResult:
        audio_path = self._settings.audio_dir / f"{bv_id}.m4a"
        if not audio_path.exists():
            await self._download_audio(bv_id, audio_path)

        # Lazy import to keep optional dep optional
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Either install it "
                "(`pip install lecturemind[whisper]`) or use videos with CC subtitles."
            ) from exc

        # Run sync transcription in a thread to avoid blocking the loop
        def _transcribe() -> list[SubtitleSegment]:
            model_name = str(self._settings.whisper_model).strip()
            local_snapshot = _local_faster_whisper_base_snapshot()
            if model_name == "base" and local_snapshot:
                logger.info("Using local faster-whisper-base snapshot: %s", local_snapshot)
                model_name = local_snapshot
            model = WhisperModel(
                model_name,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
            segments_iter, _info = model.transcribe(
                str(audio_path),
                language=None,  # auto-detect
                vad_filter=True,
            )
            return [
                SubtitleSegment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in segments_iter
                if s.text and s.text.strip()
            ]

        segments = await asyncio.to_thread(_transcribe)

        raw_path = self._settings.subtitles_dir / f"{bv_id}.whisper.json"
        raw_path.write_text(
            json.dumps(
                [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return SubtitleResult(
            source="whisper",
            language="auto",
            segments=segments,
            raw_path=raw_path,
        )

    async def _download_audio(self, bv_id: str, out_path: Path) -> None:
        """Use yt-dlp to grab the bestaudio of the BV."""
        if shutil.which("yt-dlp") is None:
            raise RuntimeError(
                "yt-dlp executable not found in PATH. "
                "Install it (`pip install yt-dlp`) and ensure ffmpeg is available."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "yt-dlp",
            "-f",
            "bestaudio",
            "-o",
            str(out_path),
            f"https://www.bilibili.com/video/{bv_id}",
            "--no-playlist",
            "--no-warnings",
            "--retries",
            "5",
            "--fragment-retries",
            "5",
            "--extractor-retries",
            "5",
            "--socket-timeout",
            "30",
            "--force-ipv4",
            "--user-agent",
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "--referer",
            "https://www.bilibili.com/",
        ]
        if self._cookie_file:
            cmd.extend(["--cookies", str(self._cookie_file)])
        fallback_cmd = [
            *cmd,
            "--no-check-certificates",
            "--socket-timeout",
            "60",
            "--sleep-requests",
            "1",
        ]
        logger.info("Running yt-dlp: %s", " ".join(cmd))

        def _run() -> None:
            last_stderr = ""
            for current_cmd in (cmd, fallback_cmd):
                for attempt in range(1, 4):
                    proc = subprocess.run(current_cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        return
                    last_stderr = proc.stderr.strip()
                    logger.warning("yt-dlp audio attempt %d failed: %s", attempt, last_stderr[:300])
                    time.sleep(2 * attempt)
            raise RuntimeError(
                "yt-dlp failed after retries. "
                f"stderr={last_stderr[:500]}"
            )

        await asyncio.to_thread(_run)
        if not out_path.exists():
            raise RuntimeError(f"yt-dlp completed but {out_path} is missing")
