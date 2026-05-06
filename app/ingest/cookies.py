"""Bilibili cookie utilities.

Reads a Netscape-format cookies.txt (the kind yt-dlp itself emits with
`--cookies-from-browser`) and exposes:

  - the raw file path (so yt-dlp can use it via `--cookies`)
  - a dict[str, str] of cookie name → value, for httpx clients

Why we need cookies:
  Many Bilibili views/subtitles are gated behind the user's SESSDATA.
  Without it, /x/web-interface/view returns a stripped payload and
  /x/player/wbi/v2 returns no subtitle URLs.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_cookie_jar(path: Path | str | None) -> dict[str, str]:
    """Parse a Netscape cookies.txt into a name→value dict.

    Returns an empty dict if the file does not exist or is empty,
    which lets the rest of the pipeline gracefully fall back to
    anonymous mode.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("Cookie file not found at %s, running anonymously", p)
        return {}
    cookies: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Netscape format:  domain  flag  path  secure  expiry  name  value
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        name, value = parts[5], parts[6]
        if name and value:
            cookies[name] = value
    if not cookies:
        logger.warning("Cookie file %s parsed to 0 cookies", p)
    else:
        logger.debug("Loaded %d cookies from %s", len(cookies), p)
    return cookies


def cookie_file_path(path: Path | str | None) -> Path | None:
    """Return the cookie file path if it exists; otherwise None.
    Used by yt-dlp callers to decide whether to pass --cookies.
    """
    if not path:
        return None
    p = Path(path)
    return p if p.exists() else None
