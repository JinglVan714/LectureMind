"""IRPatch whitelist + ``apply_patches`` for the M2 patch Reviser (P6).

Replaces the legacy "rewrite the entire LectureIR JSON" Reviser path
with a strict whitelist of small, surgical edits. Three problems
motivated the change (commit b8483b6 disabled the legacy Reviser by
default for the same reasons):

1. **Latency** — full-rewrite Reviser took 200-600s on long lectures
   even when the only fix the Critic asked for was a one-line
   ``set_quote``. The patch path keeps the LLM's output bounded to
   "issues × patches" instead of "entire IR".
2. **Silent regressions** — the legacy Reviser's system prompt forbade
   dropping protected arrays, but the model occasionally did so
   anyway. ``_restore_dropped_content`` was the bandaid; it stays
   around as a defensive layer but the *root* fix is to never let the
   model touch unrelated arrays in the first place.
3. **Auditability** — every accepted/rejected patch shows up in
   ``RevisePatchResult.rejected`` with a human-readable reason, so
   matrix reports can attribute Reviser cost to specific edits.

The whitelist is deliberately closed — anything not enumerated here is
rejected. New op / path combinations live behind a ``LECTURE_REVISER_PROMPT_VERSION``
bump so we can tell rolled-out behaviour apart in telemetry.
"""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from pydantic import ValidationError

from .ir import LectureIR

logger = logging.getLogger(__name__)


PatchOp = Literal[
    "replace_array",
    "append",
    "set_quote",
    "set_text",
    "set_explanation",
]

_KNOWN_OPS: frozenset[str] = frozenset(
    {"replace_array", "append", "set_quote", "set_text", "set_explanation"}
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IRPatch:
    """A single surgical edit proposed by the patch Reviser.

    Only one of ``value`` / ``quote`` / ``text`` is consumed depending
    on ``op`` — keeping them as separate fields (rather than a single
    polymorphic ``value``) makes the LLM's intended slot explicit and
    lets the prompt examples stay self-documenting.
    """

    op: str
    path: str
    value: Any | None = None
    quote: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class RejectedPatch:
    """Diagnostic record for a patch we refused to apply.

    ``patch=None`` is reserved for the "schema_violation" terminal
    rejection emitted when ``apply_patches`` rolls the entire batch
    back because ``LectureIR.model_validate`` failed post-apply.
    """

    patch: IRPatch | None
    reason: str


class PatchRejected(Exception):
    """Internal signal raised by validation/apply helpers; never leaks."""


@dataclass(frozen=True)
class RevisePatchResult:
    """Output of :meth:`CriticReviserAgent.revise_patch`.

    ``patches`` is the model's proposed list (whitelist filtering
    happens later inside ``apply_patches``); ``rejected`` is empty at
    this stage and populated only after application. ``unfixable_issues``
    captures the model's explicit "can't fix" list so the pipeline can
    surface them in ``pipeline_stats.reviser`` without losing them.
    """

    patches: list[IRPatch] = field(default_factory=list)
    rejected: list[RejectedPatch] = field(default_factory=list)
    unfixable_issues: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path whitelist
# ---------------------------------------------------------------------------
# Each pattern matches a fully-resolved path (``/chapters/0/...``); the
# whitelist is positive-only, so any path that fails to match is
# automatically rejected. Forbidden paths (``/profile`` / ``/taxonomy``
# / chapter ``start_sec`` / point ``ts``) therefore do not need
# explicit deny rules — they simply can't appear under any op.

_CHAPTER_ARRAY_FIELDS = (
    "code_blocks",
    "formula_blocks",
    "pitfalls",
    "key_takeaways",
    "process_steps",
)
_TOP_ARRAY_FIELDS = ("knowledge_units", "study_questions", "review_questions", "mainline")


def _re(*parts: str) -> re.Pattern[str]:
    return re.compile("^" + "|".join(parts) + "$")


# ``replace_array`` — full array replacement; ``append`` — same paths +
# ``/-`` suffix.
_PATTERN_REPLACE_ARRAY = _re(
    *(rf"/chapters/\d+/{f}" for f in _CHAPTER_ARRAY_FIELDS),
    *(rf"/{f}" for f in _TOP_ARRAY_FIELDS),
)
_PATTERN_APPEND = _re(
    *(rf"/chapters/\d+/{f}/-" for f in _CHAPTER_ARRAY_FIELDS),
    *(rf"/{f}/-" for f in _TOP_ARRAY_FIELDS),
)
_PATTERN_SET_QUOTE = _re(
    r"/chapters/\d+/points/\d+/quote",
    r"/knowledge_units/\d+/quote",
)
_PATTERN_SET_TEXT = _re(
    r"/chapters/\d+/points/\d+/text",
    r"/knowledge_units/\d+/title",
)
_PATTERN_SET_EXPLANATION = _re(
    r"/chapters/\d+/code_blocks/\d+/explanation",
    r"/chapters/\d+/formula_blocks/\d+/explanation",
    r"/knowledge_units/\d+/explanation",
)


_OP_PATTERNS: dict[str, re.Pattern[str]] = {
    "replace_array": _PATTERN_REPLACE_ARRAY,
    "append": _PATTERN_APPEND,
    "set_quote": _PATTERN_SET_QUOTE,
    "set_text": _PATTERN_SET_TEXT,
    "set_explanation": _PATTERN_SET_EXPLANATION,
}


def _validate_op_and_path(patch: IRPatch) -> None:
    """Guard the whitelist; raise :class:`PatchRejected` on any miss.

    The reason strings are short, machine-readable tags
    (``unknown_op:set_severity`` / ``path_not_in_whitelist:...``) so
    downstream telemetry can group similar failures.
    """
    if patch.op not in _KNOWN_OPS:
        raise PatchRejected(f"unknown_op:{patch.op}")
    pattern = _OP_PATTERNS[patch.op]
    if not pattern.match(patch.path or ""):
        raise PatchRejected(f"path_not_in_whitelist:{patch.path}")


# ---------------------------------------------------------------------------
# Path resolution & application
# ---------------------------------------------------------------------------


def _split_path(path: str) -> list[str]:
    """JSON-Pointer-ish splitter (no escape handling — none of our
    whitelisted paths contain ``~`` or ``/`` inside a token).
    """
    if not path or not path.startswith("/"):
        raise PatchRejected(f"malformed_path:{path}")
    return path[1:].split("/")


def _walk(payload: dict, tokens: list[str]) -> Any:
    cursor: Any = payload
    for tok in tokens:
        if isinstance(cursor, list):
            try:
                idx = int(tok)
            except (TypeError, ValueError) as exc:
                raise PatchRejected(f"path_index_not_int:{tok}") from exc
            if not (0 <= idx < len(cursor)):
                raise PatchRejected(f"path_index_oob:{idx}")
            cursor = cursor[idx]
        elif isinstance(cursor, dict):
            if tok not in cursor:
                raise PatchRejected(f"path_missing_key:{tok}")
            cursor = cursor[tok]
        else:
            raise PatchRejected(f"path_traversal_failed_at:{tok}")
    return cursor


def _set_leaf(payload: dict, tokens: list[str], value: Any) -> None:
    parent = _walk(payload, tokens[:-1])
    leaf = tokens[-1]
    if isinstance(parent, list):
        try:
            idx = int(leaf)
        except (TypeError, ValueError) as exc:
            raise PatchRejected(f"path_index_not_int:{leaf}") from exc
        if not (0 <= idx < len(parent)):
            raise PatchRejected(f"path_index_oob:{idx}")
        parent[idx] = value
        return
    if isinstance(parent, dict):
        parent[leaf] = value
        return
    raise PatchRejected(f"path_parent_not_container:{leaf}")


def _apply_one(payload: dict, patch: IRPatch) -> None:
    """Mutate ``payload`` in place per ``patch`` semantics.

    ``apply_patches`` clones the payload before calling this so a
    rejected mid-batch patch does not leave the dict half-mutated.
    """
    tokens = _split_path(patch.path)

    if patch.op == "replace_array":
        if not isinstance(patch.value, list):
            raise PatchRejected("replace_array_requires_list_value")
        _set_leaf(payload, tokens, list(patch.value))
        return

    if patch.op == "append":
        if not tokens or tokens[-1] != "-":
            raise PatchRejected("append_requires_dash_suffix")
        target = _walk(payload, tokens[:-1])
        if not isinstance(target, list):
            raise PatchRejected("append_target_not_list")
        if patch.value is None:
            raise PatchRejected("append_requires_value")
        target.append(patch.value)
        return

    if patch.op == "set_quote":
        if patch.quote is None:
            raise PatchRejected("set_quote_requires_quote")
        _set_leaf(payload, tokens, str(patch.quote))
        return

    if patch.op in ("set_text", "set_explanation"):
        if patch.text is None:
            raise PatchRejected(f"{patch.op}_requires_text")
        _set_leaf(payload, tokens, str(patch.text))
        return

    # Unreachable — _validate_op_and_path catches unknown ops earlier.
    raise PatchRejected(f"unknown_op:{patch.op}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_patches(
    ir: LectureIR, patches: Iterable[IRPatch]
) -> tuple[LectureIR, list[RejectedPatch]]:
    """Apply ``patches`` to ``ir`` and return ``(revised_ir, rejected)``.

    Semantics:

    * Each patch is validated against the whitelist; mismatches are
      collected in ``rejected`` and skipped.
    * Whitelisted patches mutate a *deep copy* of ``ir.model_dump()``
      so a runtime ``PatchRejected`` (out-of-bounds index, missing
      key) never leaves the working tree half-applied.
    * After every individual patch attempt, the resulting dict is
      validated against :class:`LectureIR`. If validation fails, the
      whole batch is rolled back and the original ``ir`` is returned
      untouched, with a synthetic ``RejectedPatch(patch=None,
      reason="schema_violation:...")`` appended so the caller can
      surface the real exception.

    The "rollback the entire batch" choice (vs. dropping just the
    offending patch) is deliberate: the LLM proposes patches as a
    coherent fix-up plan, and partial application can leave the IR
    in a worse state than the input (e.g. a freshly emptied
    ``code_blocks`` array combined with a now-pointless ``set_quote``
    on the surrounding chapter).
    """
    rejected: list[RejectedPatch] = []
    payload = copy.deepcopy(ir.model_dump())
    patches = list(patches)

    for patch in patches:
        try:
            _validate_op_and_path(patch)
            _apply_one(payload, patch)
        except PatchRejected as exc:
            rejected.append(RejectedPatch(patch=patch, reason=str(exc)))
            logger.debug("Patch rejected: %s — %s", patch, exc)

    try:
        revised = LectureIR.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Patch batch failed schema validation; rolling back: %s", exc)
        rejected.append(
            RejectedPatch(
                patch=None,
                reason=f"schema_violation:{exc.errors()[0] if exc.errors() else exc}",
            )
        )
        return ir, rejected
    return revised, rejected


__all__ = [
    "IRPatch",
    "PatchOp",
    "RejectedPatch",
    "PatchRejected",
    "RevisePatchResult",
    "apply_patches",
]
