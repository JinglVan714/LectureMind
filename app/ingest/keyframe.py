"""Keyframe extraction via PySceneDetect + ffmpeg.

Strategy: detect scene boundaries with content detector, then sample one
representative frame per scene (the middle of each scene). If too few
scenes are found we fall back to uniform sampling.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from .cookies import cookie_file_path

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    timestamp: float  # seconds
    path: Path


class KeyframeExtractor:
    """Download the lowest-acceptable video stream, then extract N keyframes."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cookie_file = cookie_file_path(self._settings.bilibili_cookie_file)

    async def extract(self, bv_id: str, duration: int) -> list[Keyframe]:
        out_dir = self._settings.keyframes_dir / bv_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already extracted
        existing = sorted(out_dir.glob("*.jpg"))
        if existing:
            logger.info("Reusing %d cached keyframes for %s", len(existing), bv_id)
            return [self._frame_from_path(p) for p in existing]

        try:
            video_path = await self._download_video(bv_id, out_dir.parent / f"{bv_id}.mp4")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Video download for keyframes failed for %s; continuing without frames: %s", bv_id, exc)
            return []
        try:
            timestamps = await asyncio.to_thread(
                self._detect_scene_timestamps, video_path, duration
            )
            frames = await asyncio.to_thread(self._extract_frames, video_path, timestamps, out_dir)
        finally:
            # Save disk: drop the source video once frames are out
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                pass
        return frames

    # ---------------- video download ----------------

    async def _download_video(self, bv_id: str, out_path: Path) -> Path:
        if out_path.exists():
            return out_path
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("yt-dlp not found in PATH")
        # Pick a low-cost stream: <= 480p height when available
        cmd = [
            "yt-dlp",
            "-f",
            "bv*[height<=480]+ba/b[height<=480]/bv*+ba/b",
            "--merge-output-format",
            "mp4",
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
        logger.info("Downloading video for keyframes: %s", bv_id)

        def _run() -> None:
            last_stderr = ""
            for current_cmd in (cmd, fallback_cmd):
                for attempt in range(1, 4):
                    proc = subprocess.run(current_cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        return
                    last_stderr = proc.stderr.strip()
                    logger.warning("yt-dlp video attempt %d failed: %s", attempt, last_stderr[:300])
                    time.sleep(2 * attempt)
            raise RuntimeError(
                "yt-dlp video download failed after retries. "
                f"stderr={last_stderr[:500]}"
            )

        await asyncio.to_thread(_run)
        if not out_path.exists():
            raise RuntimeError(f"video download finished but {out_path} missing")
        return out_path

    # ---------------- scene detection ----------------

    def _detect_scene_timestamps(self, video_path: Path, duration: int) -> list[float]:
        """Return a list of timestamps (seconds) within [keyframe_min, keyframe_max]."""
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError as exc:
            raise RuntimeError("PySceneDetect is required for keyframe extraction") from exc

        timestamps: list[float] = []
        try:
            video = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=self._settings.keyframe_threshold))
            sm.detect_scenes(video=video, show_progress=False)
            scenes = sm.get_scene_list()
            for start, end in scenes:
                mid = (start.get_seconds() + end.get_seconds()) / 2.0
                timestamps.append(mid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene detection failed (%s); falling back to uniform sampling", exc)

        # Cap to keyframe_max
        if len(timestamps) > self._settings.keyframe_max:
            step = len(timestamps) / self._settings.keyframe_max
            timestamps = [timestamps[int(i * step)] for i in range(self._settings.keyframe_max)]

        # Pad to keyframe_min via uniform sampling
        if len(timestamps) < self._settings.keyframe_min and duration > 0:
            need = self._settings.keyframe_min - len(timestamps)
            extra = [
                duration * (i + 1) / (need + 1)
                for i in range(need)
            ]
            timestamps = sorted(set(round(t, 1) for t in timestamps + extra))

        # Edge guard: at least one frame
        if not timestamps and duration > 0:
            timestamps = [duration / 2.0]
        return sorted(timestamps)

    # ---------------- frame extraction ----------------

    def _extract_frames(
        self, video_path: Path, timestamps: list[float], out_dir: Path
    ) -> list[Keyframe]:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")
        frames: list[Keyframe] = []
        for ts in timestamps:
            fname = f"{int(ts * 1000):08d}.jpg"
            fpath = out_dir / fname
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-vf",
                "scale='min(1280,iw)':-2",
                str(fpath),
                "-loglevel",
                "error",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and fpath.exists():
                frames.append(Keyframe(timestamp=ts, path=fpath))
            else:
                logger.warning("ffmpeg failed at ts=%.2f: %s", ts, proc.stderr[:200])
        return frames

    @staticmethod
    def _frame_from_path(p: Path) -> Keyframe:
        try:
            ts_ms = int(p.stem)
            return Keyframe(timestamp=ts_ms / 1000.0, path=p)
        except ValueError:
            return Keyframe(timestamp=0.0, path=p)
