from __future__ import annotations

import html
import re
from typing import Final, Literal

SECTION_TYPES: Final[tuple[str, ...]] = (
    "evidence",
    "extension",
    "background",
    "deep_dive",
    "application",
    "boundary",
    "offtopic",
)

LABELS: Final[dict[str, dict[str, str]]] = {
    "evidence": {"html": "讲义证据", "markdown": "讲义证据", "text": "讲义证据"},
    "extension": {"html": "延伸理解", "markdown": "延伸理解", "text": "延伸理解"},
    "background": {"html": "背景补全", "markdown": "背景补全", "text": "背景补全"},
    "deep_dive": {"html": "原理深挖", "markdown": "原理深挖", "text": "原理深挖"},
    "application": {"html": "应用举例", "markdown": "应用举例", "text": "应用举例"},
    "boundary": {"html": "边界说明", "markdown": "边界说明", "text": "边界说明"},
    "offtopic": {"html": "问题引导", "markdown": "问题引导", "text": "问题引导"},
}

SECTION_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\[\[\s*([a-zA-Z_]+)\s*\]\]\s*$",
    re.MULTILINE,
)


def normalise_section_type(value: str) -> str:
    return value.strip().lower()


def extract_section_types(raw: str) -> list[str]:
    return [normalise_section_type(m.group(1)) for m in SECTION_LINE_RE.finditer(raw)]


def transform(raw: str, target: Literal["html", "markdown", "text"]) -> str:
    blocks = _split_blocks(raw)
    if not blocks:
        return raw
    if target == "html":
        return "\n".join(_block_to_html(kind, body) for kind, body in blocks)
    if target == "markdown":
        return "\n\n".join(_block_to_markdown(kind, body) for kind, body in blocks)
    if target == "text":
        return "\n\n".join(_block_to_text(kind, body) for kind, body in blocks)
    raise ValueError(f"unknown transform target: {target}")


def _split_blocks(raw: str) -> list[tuple[str, str]]:
    matches = list(SECTION_LINE_RE.finditer(raw))
    blocks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        kind = normalise_section_type(match.group(1))
        if kind not in SECTION_TYPES:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        blocks.append((kind, body))
    return blocks


def _label(kind: str, target: Literal["html", "markdown", "text"]) -> str:
    return LABELS.get(kind, {}).get(target, kind)


def _block_to_html(kind: str, body: str) -> str:
    escaped = html.escape(body).replace("\n", "<br>")
    label = html.escape(_label(kind, "html"))
    return f'<section class="cp-section cp-section-{kind}" data-section-type="{kind}"><h4>{label}</h4><div>{escaped}</div></section>'


def _block_to_markdown(kind: str, body: str) -> str:
    return f"### {_label(kind, 'markdown')}\n\n{body}".strip()


def _block_to_text(kind: str, body: str) -> str:
    return f"【{_label(kind, 'text')}】\n{body}".strip()
