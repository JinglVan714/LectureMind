"""Bilibili metadata fetching: BV parsing, video info, page list."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from .cookies import load_cookie_jar

logger = logging.getLogger(__name__)


_BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")


@dataclass
class VideoMeta:
    bv_id: str
    aid: int
    title: str
    author: str
    duration: int  # seconds
    cover_url: str
    description: str
    pages: list[dict[str, Any]]  # [{cid, page, part, duration}]

    @property
    def primary_cid(self) -> int:
        return self.pages[0]["cid"] if self.pages else 0


class BilibiliIngest:
    """Fetch video metadata via the public web API.

    We deliberately use the lightweight /x/web-interface/view endpoint
    rather than the full bilibili-api-python Video class, to keep
    dependency surface and risk of API drift small.
    """

    INFO_URL = "https://api.bilibili.com/x/web-interface/view"

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        }
        self._cookies = load_cookie_jar(get_settings().bilibili_cookie_file)

    @staticmethod
    def parse_bv(url_or_bv: str) -> str:
        """Extract a BV id from any URL or raw BV string."""
        m = _BV_RE.search(url_or_bv)
        if not m:
            raise ValueError(f"Cannot find BV id in: {url_or_bv!r}")
        return m.group(0)

    async def fetch_meta(self, url_or_bv: str) -> VideoMeta:
        bv_id = self.parse_bv(url_or_bv)
        try:
            payload = await self._fetch_payload_httpx(bv_id)
        except httpx.HTTPError as exc:
            logger.warning("Bilibili API fetch failed for %s via httpx: %s", bv_id, exc)
            payload = await asyncio.to_thread(self._fetch_payload_curl, bv_id)
        return self._payload_to_meta(bv_id, payload)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _fetch_payload_httpx(self, bv_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=self._timeout, headers=self._headers, cookies=self._cookies
        ) as client:
            resp = await client.get(self.INFO_URL, params={"bvid": bv_id})
            resp.raise_for_status()
            return resp.json()

    def _fetch_payload_curl(self, bv_id: str) -> dict[str, Any]:
        headers: list[str] = []
        for name, value in self._headers.items():
            headers.extend(["-H", f"{name}: {value}"])
        if self._cookies:
            cookie = "; ".join(f"{k}={v}" for k, v in self._cookies.items())
            headers.extend(["-H", f"Cookie: {cookie}"])
        completed = subprocess.run(
            [
                "curl.exe",
                "-L",
                "--silent",
                "--show-error",
                *headers,
                f"{self.INFO_URL}?bvid={bv_id}",
            ],
            capture_output=True,
            check=True,
        )
        text = completed.stdout.decode("utf-8", errors="replace")
        start = text.find("{")
        if start < 0:
            raise RuntimeError(f"curl returned no JSON for {bv_id}: {text[:200]}")
        return json.loads(text[start:])

    def _payload_to_meta(self, bv_id: str, payload: dict[str, Any]) -> VideoMeta:
        if payload.get("code") != 0:
            raise RuntimeError(
                f"Bilibili API error for {bv_id}: code={payload.get('code')} "
                f"msg={payload.get('message')}"
            )
        data = payload["data"]
        pages = [
            {
                "cid": p["cid"],
                "page": p["page"],
                "part": p.get("part", ""),
                "duration": p.get("duration", 0),
            }
            for p in data.get("pages", [])
        ]
        return VideoMeta(
            bv_id=bv_id,
            aid=data["aid"],
            title=data["title"],
            author=data.get("owner", {}).get("name", ""),
            duration=data.get("duration", 0),
            cover_url=data.get("pic", ""),
            description=data.get("desc", "") or "",
            pages=pages,
        )
