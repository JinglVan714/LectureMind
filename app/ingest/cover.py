from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class CoverCache:
    def __init__(self, timeout: float = 15.0) -> None:
        self._settings = get_settings()
        self._timeout = timeout
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        }

    async def fetch(self, bv_id: str, cover_url: str) -> Path | None:
        if not cover_url:
            return None
        out_dir = self._settings.data_dir / "covers"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = _suffix_from_url(cover_url)
        out_path = out_dir / f"{bv_id}{suffix}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=self._headers
        ) as client:
            last_error = ""
            for url in _candidate_urls(cover_url):
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    if "image" not in content_type.lower() and not resp.content.startswith((b"\xff\xd8", b"\x89PNG", b"RIFF")):
                        logger.warning("Cover response for %s is not an image: %s", bv_id, content_type)
                        continue
                    out_path.write_bytes(resp.content)
                    return out_path
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    logger.warning("Failed cover candidate for %s (%s): %s", bv_id, url, exc)
            logger.warning("Failed to cache cover for %s: %s", bv_id, last_error)
            return None


def _suffix_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(suffix):
            return suffix
    return ".jpg"


def _candidate_urls(url: str) -> list[str]:
    urls: list[str] = []
    if url.startswith("http://"):
        urls.append("https://" + url.removeprefix("http://"))
    urls.append(url)
    deduped: list[str] = []
    for candidate in urls:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped
