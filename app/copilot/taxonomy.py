"""Auto-tagging normalisation for ``LectureIR.taxonomy``.

The LLM is prompted to emit a ``taxonomy`` block with a ``domain`` chosen
from a fixed white-list, a free-form ``direction`` phrase, ``tags`` and a
``confidence``.  This module post-processes that raw output so that:

* ``domain`` is forced into :data:`DOMAIN_WHITELIST` (anything else falls
  back to ``"其他"`` with confidence clamped to ``0.3``).
* ``direction`` has stage suffixes (``教程`` / ``入门`` / ``详解`` / ...)
  stripped and inner whitespace collapsed.
* ``tags`` are deduplicated case-insensitively and capped at 5 entries
  while preserving the LLM's original casing for display.
* ``confidence`` is clamped to ``[0, 1]``.

The companion :func:`needs_review` predicate is consumed by the index
page (Stage 8) to decide which lectures land in the *待归类* bucket.

Stage 1 only exposes the pure functions; Stage 2 / Stage 8 will reuse
them from the indexer and the manual re-grouping endpoint respectively.
"""
from __future__ import annotations

import re

from app.understand.schema import Taxonomy

# Keep this in sync with the white-list in ``app/understand/prompts.py``
# rule 16.  Any change requires a re-run of the LLM (otherwise existing
# rows would stale-fall into "其他").
DOMAIN_WHITELIST: tuple[str, ...] = (
    "AI 技术",
    "编程开发",
    "数据科学",
    "硬件与系统",
    "数学",
    "物理",
    "化学生物",
    "医学",
    "烹饪",
    "健身运动",
    "金融投资",
    "人文社科",
    "艺术设计",
    "工程实务",
    "其他",
)

_DOMAIN_SET: frozenset[str] = frozenset(DOMAIN_WHITELIST)

# Suffix tokens that signal "stage" rather than topic; we strip them so
# "Rust 并发模型 教程" / "Rust 并发模型 入门" / "Rust 并发模型 详解" all
# collapse to "Rust 并发模型" for grouping purposes.
_DIRECTION_SUFFIX_RE = re.compile(
    r"\s*(?:教程|入门|进阶|详解|教学|讲解|课程|课|实战|手把手|从零|一文)+\s*$"
)
_WHITESPACE_RE = re.compile(r"\s+")

# Maximum tag count; mirrors prompt rule 18 "3-5 个".
_MAX_TAGS = 5

# Anything below this confidence (or domain == "其他") is surfaced as
# *needs_review* on the index page.
_NEEDS_REVIEW_THRESHOLD = 0.4

# When the LLM drifts off-list we keep at most this much confidence so the
# UI never shows an unverified "其他" with a deceptively high score.
_FALLBACK_CONFIDENCE_CEILING = 0.3


def normalize(taxonomy: Taxonomy | None) -> Taxonomy | None:
    """Return a repaired copy of ``taxonomy``.

    ``None`` in, ``None`` out — callers can feed it the raw IR field
    directly without having to short-circuit themselves.
    """
    if taxonomy is None:
        return None

    domain_raw = (taxonomy.domain or "").strip()
    confidence = float(taxonomy.confidence)
    if domain_raw in _DOMAIN_SET:
        domain = domain_raw
    else:
        domain = "其他"
        confidence = min(confidence, _FALLBACK_CONFIDENCE_CEILING)

    direction = _strip_direction(taxonomy.direction or "")
    tags = _dedupe_tags(taxonomy.tags or [])
    confidence = max(0.0, min(1.0, confidence))

    return Taxonomy(
        domain=domain,
        direction=direction,
        tags=tags,
        confidence=confidence,
    )


def needs_review(taxonomy: Taxonomy | None) -> bool:
    """Whether the lecture should land in the *待归类* bucket.

    Returns ``True`` when ``taxonomy`` is missing entirely, when the LLM
    fell back to ``"其他"``, when ``direction`` is empty, or when
    confidence dropped below :data:`_NEEDS_REVIEW_THRESHOLD`.
    """
    if taxonomy is None:
        return True
    if not taxonomy.direction.strip():
        return True
    if taxonomy.domain == "其他":
        return True
    if taxonomy.confidence < _NEEDS_REVIEW_THRESHOLD:
        return True
    return False


# ---------------------------------------------------------------------------
# helpers


def _strip_direction(text: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    # Repeatedly peel suffixes so "Rust 并发 教程详解" → "Rust 并发".
    while cleaned:
        new = _DIRECTION_SUFFIX_RE.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned


def _dedupe_tags(raw_tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        text = _WHITESPACE_RE.sub(" ", tag).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= _MAX_TAGS:
            break
    return out
