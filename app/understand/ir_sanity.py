"""Post-IR sanity validator and cleaners.

Operates on the normalised LectureIR dict (after
:func:`ir.hydrate_lecture_ir_data` in the single-call path, or after
``reduce_local`` + ``reduce_global`` in the map-reduce path) and applies
**in-place** fixes that defend against three common LLM output
regressions observed on real BV runs (see ``handoff_closeout.md``
Q1-Q4):

1. ``process_steps`` items emitted with their own leading numbering
   (``"1. "``, ``"Step1:"``, ``"①"``…). The lecture template wraps
   ``process_steps`` in an auto-numbered ``<ol>``, so leaving the
   prefix produces ``1. 1. xxx`` / ``2. Step2 xxx`` ("1.1 / 2.2") in
   the rendered HTML.
2. ``mainline`` degraded into a chapter-title list
   (``["第 1 章 · …", "第 2 章 · …"]``). This duplicates
   ``chapter-nav`` and — because :func:`ir_builder._one_liner` falls
   back to ``mainline[0]`` when ``core_question`` is empty — surfaces a
   chapter title as the hero one-liner.
3. ``core_question`` that echoes the first chapter title instead of a
   genuine question.
4. Chapter index gaps / non-monotonic indices (e.g. the LLM skips the
   first chapter or restarts numbering mid-list), which mis-attribute
   knowledge_units and break in-page anchors.

The module is side-effect free aside from mutating the supplied dict.
It returns a telemetry dict suitable for
``pipeline_stats['ir_sanity']`` so the pipeline can surface warnings
without forcing a hard failure.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["sanitize_ir_data", "strip_list_prefix"]


# --------------------------------------------------------------------------- #
# Regex toolbox                                                                #
# --------------------------------------------------------------------------- #

# Match a leading "Nth chapter" header in mainline / one_liner.
# Covers half-width and full-width digits as well as 一二三...十 numerals.
_CHAPTER_HEADER_RE = re.compile(
    r"^\s*第\s*[一二三四五六七八九十百零0-9０-９]+\s*[章节回讲集篇]"
)

# Match a leading list-number prefix that an LLM may have inlined into a
# process_steps item.  Each alternative consumes a separator (`.` `、` `:`
# `)` …) and optional whitespace; a trailing **lookahead** then requires
# the next visible character to NOT be a digit.  That single lookahead is
# what keeps real numbers ("1.5x speed", "v1.0", "Step10" as a literal
# step count) from being mistaken for list prefixes — the separator can
# match, but the following digit fails the lookahead so no strip happens.
_LIST_PREFIX_RE = re.compile(
    r"""^
    (?:
        # Step1:, STEP 2., 步骤3、, 阶段4) — keyword + digit + optional sep
        (?:Step|STEP|step|阶段|步骤)\s*\d+\s*[:：.．、)\uFF09]?
      | # 1.  1．  1、  1)  1）  1:  1： — digit + separator
        \d+\s*[\.．、)\uFF09:：]
      | # (1)  （1）
        [(\uFF08]\s*\d+\s*[)\uFF09]
      | # ①②③ … (Unicode circled digits up to 50)
        [\u2460-\u2473\u2776-\u2793\u24EB-\u24FF]
    )
    \s*
    (?=[^\d\s]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalise_title_key(text: str) -> str:
    """Cheap fuzzy-equality key for comparing chapter titles to mainline /
    core_question entries.  Strips whitespace, punctuation and Chinese
    enumeration prefixes so ``"第 1 章 · 开场与问题引入。"`` and ``"开场与
    问题引入"`` collapse to the same key.
    """
    cleaned = _CHAPTER_HEADER_RE.sub("", text or "")
    cleaned = re.sub(
        r"[\s，。！？!?、；;：:,.·•《》“”\"'（）()\[\]【】\-—_/]+",
        "",
        cleaned,
    )
    return cleaned.strip().lower()


def strip_list_prefix(text: str) -> tuple[str, bool]:
    """Strip a single leading list-number prefix from ``text``.

    Returns ``(cleaned, stripped)``. Conservative by design: only one
    prefix is removed per call so legitimate content like
    ``"第 1 步：操作 …"`` doesn't get cascaded into nothing.
    """
    if not isinstance(text, str) or not text:
        return text or "", False
    new = _LIST_PREFIX_RE.sub("", text, count=1)
    if new == text:
        return text, False
    return new.lstrip(), True


# --------------------------------------------------------------------------- #
# Individual fixers                                                            #
# --------------------------------------------------------------------------- #


def _sanitize_process_steps(chapters: list[dict[str, Any]]) -> int:
    """Strip leading list-number prefixes from every ``process_steps`` item.

    Returns the number of items that had a prefix removed.
    """
    patched = 0
    for ch in chapters:
        steps = ch.get("process_steps")
        if not isinstance(steps, list) or not steps:
            continue
        new_steps: list[str] = []
        for step in steps:
            text = step if isinstance(step, str) else str(step)
            cleaned, stripped = strip_list_prefix(text)
            if stripped:
                patched += 1
            new_steps.append(cleaned)
        ch["process_steps"] = new_steps
    return patched


def _is_chapter_title_echo(text: str, chapter_keys: set[str]) -> bool:
    """Return True when ``text`` is effectively a restatement of a chapter
    title (or carries a ``"第 N 章 …"`` header).
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if _CHAPTER_HEADER_RE.search(text):
        return True
    key = _normalise_title_key(text)
    return bool(key) and key in chapter_keys


def _sanitize_mainline(
    data: dict[str, Any], chapter_keys: set[str]
) -> bool:
    """Clear ``data['mainline']`` when it has degraded into a chapter-title list.

    Trigger: at least ``ceil(N/2)`` items (and ``≥ 2``) match a chapter
    title or carry a ``"第 N 章"`` header. We don't try to repair the
    individual items because the LLM was clearly summarising the
    table-of-contents instead of the cognitive path; keeping a partial
    list would still mislead the reader. Downstream
    :func:`ir_builder._one_liner` will then fall back to the lecture
    title, and the renderer hides the empty ``mainline-section``.
    """
    items = data.get("mainline")
    if not isinstance(items, list) or not items:
        return False
    bad = sum(1 for it in items if _is_chapter_title_echo(it, chapter_keys))
    if bad < 2 or bad * 2 < len(items):
        return False
    data["mainline"] = []
    return True


def _sanitize_core_question(
    data: dict[str, Any], chapter_keys: set[str], title: str
) -> bool:
    """Clear ``data['core_question']`` when it echoes a chapter title or the
    lecture title.

    Returning ``True`` lets the renderer hide the ``.hero-question`` block
    instead of displaying noise; the downstream one-liner fallback
    cascades to the lecture title.
    """
    text = data.get("core_question")
    if not isinstance(text, str) or not text.strip():
        return False
    if _is_chapter_title_echo(text, chapter_keys):
        data["core_question"] = ""
        return True
    title_key = _normalise_title_key(title)
    if title_key and _normalise_title_key(text) == title_key:
        data["core_question"] = ""
        return True
    return False


def _renumber_chapters(data: dict[str, Any]) -> dict[str, Any] | None:
    """Renumber ``data['chapters']`` to a contiguous 1..N sequence when the
    LLM-supplied indices are non-contiguous, non-monotonic, or don't start
    at 1.

    Also patches ``knowledge_units[*].chapter_index`` cross-references and
    ``chapters[*].code_blocks[*].chapter_index`` / ``formula_blocks[*].
    chapter_index`` to follow the renumbering.

    Returns a telemetry dict ``{"original_indices": [...], "renumbered":
    True}`` when a change was applied, otherwise ``None``. Chapter order
    in the list is preserved — only the *labels* are rewritten.
    """
    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        return None

    original = []
    expected_contiguous = True
    for offset, ch in enumerate(chapters):
        idx = ch.get("index") if isinstance(ch, dict) else None
        try:
            idx_int = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            idx_int = None
        original.append(idx_int)
        if idx_int != offset + 1:
            expected_contiguous = False

    if expected_contiguous:
        return None

    # Build remap from any positive original index -> new 1-based label.
    remap: dict[int, int] = {}
    for offset, idx_int in enumerate(original):
        if isinstance(idx_int, int) and idx_int > 0:
            # First-write wins so duplicate originals route to the first slot
            remap.setdefault(idx_int, offset + 1)
        # Rewrite the chapter index in-place
        chapters[offset]["index"] = offset + 1

    # Fix knowledge_units cross-refs.
    for ku in data.get("knowledge_units") or []:
        if not isinstance(ku, dict):
            continue
        try:
            ku_idx = int(ku.get("chapter_index"))
        except (TypeError, ValueError):
            continue
        if ku_idx in remap:
            ku["chapter_index"] = remap[ku_idx]
        elif not (1 <= ku_idx <= len(chapters)):
            # Out of range — pin to first chapter so the renderer can still
            # locate it instead of silently dropping the unit.
            ku["chapter_index"] = 1

    # Fix code/formula block cross-refs (kept on each chapter dict).
    for ch in chapters:
        for arr_key in ("code_blocks", "formula_blocks"):
            arr = ch.get(arr_key)
            if not isinstance(arr, list):
                continue
            for blk in arr:
                if not isinstance(blk, dict):
                    continue
                try:
                    blk_idx = int(blk.get("chapter_index"))
                except (TypeError, ValueError):
                    continue
                if blk_idx in remap:
                    blk["chapter_index"] = remap[blk_idx]

    return {"original_indices": original, "remap": remap}


# --------------------------------------------------------------------------- #
# Public entry-point                                                           #
# --------------------------------------------------------------------------- #


def sanitize_ir_data(data: dict[str, Any]) -> dict[str, Any]:
    """Apply all sanity passes to ``data`` in-place and return telemetry.

    Telemetry keys (always present, defaults are falsy):

    * ``process_steps_prefix_stripped`` (int): number of ``process_steps``
      items that had a leading list-number prefix removed.
    * ``mainline_degraded`` (bool): True when ``mainline`` was cleared
      because it had collapsed into a chapter-title list.
    * ``core_question_dropped`` (bool): True when ``core_question`` was
      cleared because it echoed a chapter / lecture title.
    * ``chapter_renumbered`` (bool): True when chapter indices were
      rewritten to a contiguous 1..N range.
    * ``chapter_remap`` (dict|None): mapping from original index to new
      label when renumbering occurred.
    * ``chapter_original_indices`` (list|None): the LLM-supplied indices
      in list order, captured before renumbering.

    The dict is safe to expose via ``pipeline_stats``.
    """
    chapters = data.get("chapters") or []
    if not isinstance(chapters, list):
        chapters = []
        data["chapters"] = chapters

    title = str(data.get("title") or "")
    chapter_keys = {
        _normalise_title_key(ch.get("title", "") if isinstance(ch, dict) else "")
        for ch in chapters
    }
    chapter_keys.discard("")

    stripped = _sanitize_process_steps(chapters)
    mainline_degraded = _sanitize_mainline(data, chapter_keys)
    core_question_dropped = _sanitize_core_question(data, chapter_keys, title)
    renumber = _renumber_chapters(data)

    return {
        "process_steps_prefix_stripped": stripped,
        "mainline_degraded": mainline_degraded,
        "core_question_dropped": core_question_dropped,
        "chapter_renumbered": renumber is not None,
        "chapter_remap": renumber["remap"] if renumber else None,
        "chapter_original_indices": (
            renumber["original_indices"] if renumber else None
        ),
    }
