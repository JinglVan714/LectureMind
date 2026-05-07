"""Repair JSON-eaten LaTeX backslashes.

LLMs frequently emit LaTeX inline math in JSON output without escaping the
backslash, e.g. ``"$Total = \\beta_1 A$"`` is written as the literal seven
characters ``$\beta_1$`` instead of the JSON-correct ``$\\beta_1$``.  The
strict ``json.loads`` parser then honours the implicit JSON escape:

* ``\b`` → U+0008 (backspace), so ``\beta_1`` becomes ``[BS]eta_1``.
* ``\t`` → U+0009 (tab), so ``\tau`` / ``\text`` / ``\theta`` lose their
  leading backslash AND first character.
* ``\f`` → U+000C (form feed), so ``\frac`` becomes ``[FF]rac``.
* ``\r`` → U+000D (carriage return), so ``\rho`` / ``\rightarrow`` follow
  suit.

The renderer later splices these control characters into HTML, KaTeX sees
gibberish, and the user sees ``⊘eta_1`` or ``ightarrow``.

We can losslessly repair these because in lecture-summary text, a lone
control character followed immediately by an ASCII letter is **never** a
legitimate run of plain prose — it is always a LaTeX command whose
backslash got eaten.  ``\x0a`` (LF) is excluded on purpose: paragraph
breaks legitimately produce ``[LF]<letter>`` runs, and over-correcting
would re-introduce ``\nu`` inside normal sentences.
"""
from __future__ import annotations

import re
from typing import Any

# Map: control char eaten by JSON → the LaTeX-escape we want to restore.
_CONTROL_TO_ESCAPE: dict[str, str] = {
    "\x08": "\\b",  # backspace ← \beta, \binom, \bar, \boxed, ...
    "\x09": "\\t",  # tab       ← \tau, \text, \theta, \times, \to, ...
    "\x0c": "\\f",  # form feed ← \frac, \forall, \floor, \fbox, ...
    "\x0d": "\\r",  # CR        ← \rho, \rightarrow, \rangle, ...
}

_CONTROL_RE = re.compile(r"([\x08\x09\x0c\x0d])(?=[A-Za-z])")


def repair_string(text: str) -> str:
    """Restore the leading backslash for any LaTeX command whose escape
    sequence was silently consumed by ``json.loads``.

    Idempotent: a clean string with no control-char-followed-by-letter
    pattern is returned unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    return _CONTROL_RE.sub(lambda m: _CONTROL_TO_ESCAPE[m.group(1)], text)


def repair_obj(obj: Any) -> Any:
    """Recursively apply :func:`repair_string` to every string leaf inside
    ``obj`` (dicts, lists, tuples).  Non-string scalars pass through.

    Returns the same container objects with their string leaves replaced
    in-place when possible (dicts and lists).  Tuples are immutable so a
    repaired tuple is returned.
    """
    if isinstance(obj, str):
        return repair_string(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = repair_obj(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = repair_obj(v)
        return obj
    if isinstance(obj, tuple):
        return tuple(repair_obj(v) for v in obj)
    return obj


__all__ = ["repair_string", "repair_obj"]
