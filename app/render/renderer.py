"""Jinja2 renderer: LectureJSON → self-contained HTML on disk."""
from __future__ import annotations

import base64
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from ..config import get_settings
from ..understand.schema import LectureJSON

logger = logging.getLogger(__name__)
_GLOBAL_VISUAL_LIMIT = 5
_CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+\-.#]*)?[ \t]*\n(.*?)```", re.DOTALL)
_TIMELINE_TS_RE = re.compile(
    r"^\s*(?P<time>(?:\d{1,2}:)?\d{1,2}:\d{2}|\d+(?:\.\d+)?\s*(?:s|秒))\s*[:：\-—]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_KATEX_REQUIRED_ASSETS = (
    "katex.min.css",
    "katex.min.js",
    "contrib/auto-render.min.js",
    "contrib/mhchem.min.js",
)


_PENDING_DOMAIN = "待归类"


def group_summaries_by_domain(rows: list) -> list[dict]:
    """Group ``SummaryRow`` objects into a two-level tree for the index page.

    Output schema (kept as plain dicts so the Jinja template doesn't need
    to import any dataclass)::

        [
          {
            "domain": "AI 技术",
            "count": 5,
            "needs_review": False,
            "directions": [
              {"direction": "注意力机制", "count": 3, "cards": [SummaryRow, ...]},
              {"direction": "RLHF",         "count": 2, "cards": [...]},
            ],
          },
          {
            "domain": "待归类",
            "count": 2,
            "needs_review": True,
            "directions": [
              {"direction": "",  # flat bucket — no sub-grouping
               "count": 2,
               "cards": [...]},
            ],
          },
        ]

    Sort rules (kept deterministic to simplify snapshot tests):

    * ``"待归类"`` is always pinned after classified domains when present.
    * Remaining domains are sorted by ``count`` DESC then ``domain`` ASC.
    * Within a domain, directions are sorted by ``count`` DESC then name ASC.
    * Within a direction, items are sorted by ``created_at`` DESC (empty
      strings sink to the bottom).

    A row lands in ``"待归类"`` when its persisted ``domain`` is missing,
    is the ``"其他"`` fall-back, or when ``direction`` is empty. This
    mirrors :func:`app.copilot.taxonomy.needs_review` but operates on
    the flat ``summaries`` columns so we don't have to re-parse
    ``summary_json`` for every row at render time.
    """
    pending: list = []
    buckets: dict[tuple[str, str], list] = {}

    for row in rows:
        if getattr(row, "status", "done") != "done":
            # Skip in-flight / failed rows — they're surfaced by the
            # existing /api/jobs page, not by the lecture library.
            continue
        domain = (getattr(row, "domain", None) or "").strip()
        direction = (getattr(row, "direction", None) or "").strip()
        if not domain or domain == "其他" or not direction:
            pending.append(row)
            continue
        buckets.setdefault((domain, direction), []).append(row)

    # Sort items inside each bucket by created_at DESC.
    def _sort_key_item(r):
        return (getattr(r, "created_at", "") or "")

    domain_map: dict[str, list[dict]] = {}
    for (domain, direction), items in buckets.items():
        items.sort(key=_sort_key_item, reverse=True)
        domain_map.setdefault(domain, []).append(
            {"direction": direction, "count": len(items), "cards": items}
        )

    groups: list[dict] = []
    for domain, directions in domain_map.items():
        directions.sort(key=lambda d: (-d["count"], d["direction"]))
        groups.append(
            {
                "domain": domain,
                "count": sum(d["count"] for d in directions),
                "needs_review": False,
                "directions": directions,
            }
        )
    groups.sort(key=lambda g: (-g["count"], g["domain"]))

    if pending:
        pending.sort(key=_sort_key_item, reverse=True)
        groups.append(
            {
                "domain": _PENDING_DOMAIN,
                "count": len(pending),
                "needs_review": True,
                "directions": [
                    {"direction": "", "count": len(pending), "cards": pending}
                ],
            }
        )

    return groups


def _format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_duration(seconds: float | int) -> str:
    return _format_ts(seconds)


def _bili_link(bv_id: str, ts: float) -> str:
    return f"https://www.bilibili.com/video/{bv_id}/?t={int(ts)}"


def _rich_text(value: object) -> Markup:
    if value is None:
        return Markup("")
    text = str(value)
    chunks: list[str] = []
    pos = 0
    for match in _CODE_FENCE_RE.finditer(text):
        chunks.append(_rich_plain(text[pos : match.start()]))
        language = _sanitize_code_language(match.group(1) or "")
        code = escape(match.group(2).strip("\n"))
        class_attr = f' class="language-{language}"' if language else ""
        chunks.append(f'<pre class="code-block"><code{class_attr}>{code}</code></pre>')
        pos = match.end()
    chunks.append(_rich_plain(text[pos:]))
    return Markup("".join(chunks))


def _rich_plain(text: str) -> str:
    return str(escape(text)).replace("\n", "<br>")


def _sanitize_code_language(language: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+\-.#]", "", language.strip())[:32]


def _compact_inline_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        text = "".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def _frame_url(path: str, bv_id: str, data_dir: Path | None = None) -> str:
    if not path:
        return ""
    data_dir = data_dir or get_settings().data_dir
    resolved = _resolve_frame_path(path, bv_id, data_dir)
    if resolved and resolved.exists():
        try:
            encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except OSError as exc:
            logger.warning("Failed to inline frame %s: %s", resolved, exc)
    return _relative_frame_url(path, bv_id)


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _cover_url(lecture: LectureJSON, cover_path: Path | None) -> str:
    if cover_path and cover_path.exists():
        try:
            return _image_data_url(cover_path)
        except OSError as exc:
            logger.warning("Failed to inline cover %s: %s", cover_path, exc)
    frame = next((f for f in lecture.visual_evidence if f.path), None)
    if frame is None:
        for ch in lecture.chapters:
            frame = next((f for f in ch.frames if f.path), None)
            if frame is not None:
                break
    if frame is not None:
        return _frame_url(frame.path, lecture.bv_id)
    return lecture.cover_url


def _resolve_frame_path(path: str, bv_id: str, data_dir: Path) -> Path | None:
    p = Path(path)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                p,
                data_dir / p,
                data_dir / "keyframes" / bv_id / p.name,
            ]
        )
        s = str(path).replace("\\", "/")
        if s.startswith("keyframes/"):
            candidates.append(data_dir / s)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _relative_frame_url(path: str, bv_id: str) -> str:
    p = Path(path)
    s = str(path).replace("\\", "/")
    if s.startswith("keyframes/"):
        return "../" + s
    if f"keyframes/{bv_id}/" in s:
        tail = s.split("keyframes/", 1)[-1]
        return "../keyframes/" + tail
    return f"../keyframes/{bv_id}/{p.name}"


def _index_frame_url(path: str, bv_id: str) -> str:
    if not path:
        return ""
    s = str(path).replace("\\", "/")
    if s.startswith(("data:image/", "http://", "https://")):
        return s
    return f"/keyframes/{bv_id}/{Path(path).name}"


def _summary_cover_url(row: object) -> str:
    bv_id = str(getattr(row, "bv_id", "") or "")
    data = getattr(row, "summary_json", None) or {}
    if isinstance(data, dict):
        for frame in data.get("visual_evidence") or []:
            if isinstance(frame, dict) and frame.get("path"):
                return _index_frame_url(str(frame["path"]), bv_id)
        for chapter in data.get("chapters") or []:
            if not isinstance(chapter, dict):
                continue
            for frame in chapter.get("frames") or []:
                if isinstance(frame, dict) and frame.get("path"):
                    return _index_frame_url(str(frame["path"]), bv_id)
        for key in ("cover_url", "cover"):
            value = str(data.get(key) or "").strip()
            if value:
                return value
    return str(getattr(row, "cover_url", "") or "")


def _type_recap(lecture: LectureJSON) -> dict[str, object] | None:
    units = [u for u in lecture.knowledge_units if u.title]
    primary = lecture.profile.primary_type
    if primary == "technical_formula":
        selected = [u for u in units if u.type in {"formula", "code", "mechanism", "pitfall"}]
        if selected:
            return {"title": "技术复盘", "units": selected, "style": "technical-section"}
    if primary == "procedural_tutorial":
        selected = [u for u in units if u.type == "procedure"]
        if len(selected) >= 3:
            return {"title": "操作路线与检查点", "units": selected, "style": "procedure-section"}
    if primary == "conceptual_talk":
        selected = [u for u in units if u.type in {"concept", "mechanism", "example", "boundary", "pitfall"}]
        if selected:
            return {"title": "观点与边界复盘", "units": selected, "style": "conceptual-section"}
    selected = [u for u in units if u.type in {"concept", "mechanism", "example", "boundary", "pitfall"}]
    if selected:
        return {"title": "关键知识点复盘", "units": selected, "style": "generic-section"}
    return None


def _mainline_items(lecture: LectureJSON) -> list[str]:
    items = lecture.learning_path or lecture.timeline.argument_path
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        key = re.sub(r"[\s，。！？!?、；;：:,.《》“”\"'（）()\[\]【】\-—_]+", "", text).lower()
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 8:
            break
    return out


def _timeline_detail_items(items: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items:
        text = _compact_inline_text(item)
        if not text:
            continue
        time = ""
        body = text
        match = _TIMELINE_TS_RE.match(text)
        if match:
            time = re.sub(r"\s+", "", match.group("time"))
            body = match.group("body").strip() or text
        out.append({"time": time, "text": body})
    return out


def _ensure_katex_assets(reports_dir: Path) -> bool:
    vendor_dir = Path(__file__).parent.parent / "static" / "vendor" / "katex"
    missing = [asset for asset in _KATEX_REQUIRED_ASSETS if not (vendor_dir / asset).exists()]
    if missing:
        logger.warning("KaTeX assets are missing: %s", ", ".join(missing))
        return False
    target_dir = reports_dir / "assets" / "katex"
    try:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendor_dir, target_dir, dirs_exist_ok=True)
    except OSError as exc:
        logger.warning("Failed to copy KaTeX assets to %s: %s", target_dir, exc)
        return False
    missing_target = [asset for asset in _KATEX_REQUIRED_ASSETS if not (target_dir / asset).exists()]
    if missing_target:
        logger.warning("Copied KaTeX assets are incomplete: %s", ", ".join(missing_target))
        return False
    return True


def _global_visual_evidence(lecture: LectureJSON) -> list:
    frames = []
    seen: set[str] = set()
    for frame in lecture.visual_evidence:
        if not frame.path or frame.path in seen:
            continue
        seen.add(frame.path)
        frames.append(frame)
    if len(frames) > _GLOBAL_VISUAL_LIMIT:
        preferred = {"diagram", "formula", "code", "table", "slide_text"}
        frames = sorted(
            frames,
            key=lambda f: (
                f.importance_score,
                1 if f.visual_type in preferred else 0,
                1 if f.ocr_text or f.caption else 0,
            ),
            reverse=True,
        )[:_GLOBAL_VISUAL_LIMIT]
    return sorted(frames, key=lambda f: f.ts)


class Renderer:
    def __init__(self) -> None:
        self._settings = get_settings()
        templates_dir = Path(__file__).parent / "templates"
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html", "j2"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["fmt_ts"] = _format_ts
        self._env.filters["fmt_dur"] = _format_duration
        self._env.filters["rich_text"] = _rich_text

    def render_lecture(
        self, lecture: LectureJSON, css_inline: str, cover_path: Path | None = None
    ) -> Path:
        tpl = self._env.get_template("lecture.html.j2")
        katex_assets_available = _ensure_katex_assets(self._settings.reports_dir)
        # Build a ``frame.path → global frame_id`` map that matches
        # Stage-3's ``_flat_frames`` so the Copilot panel's [F7]
        # anchors resolve to the exact DOM nodes (D28).
        from app.copilot.tools import _flat_frames  # local import avoids cycles

        frame_id_map = {
            fr.path: idx
            for idx, (fr, _chapter_idx) in enumerate(_flat_frames(lecture), start=1)
            if fr.path
        }
        html = tpl.render(
            lecture=lecture,
            css_inline=css_inline,
            copilot_css_inline=self.load_copilot_css(),
            frame_id_map=frame_id_map,
            cover_url=_cover_url(lecture, cover_path),
            type_recap=_type_recap(lecture),
            mainline_items=_mainline_items(lecture),
            timeline_turning_points=_timeline_detail_items(lecture.timeline.turning_points),
            timeline_state_evolution=_timeline_detail_items(lecture.timeline.state_evolution),
            global_visual_evidence=_global_visual_evidence(lecture),
            katex_assets_available=katex_assets_available,
            bili_link=_bili_link,
            frame_url=lambda p: _frame_url(p, lecture.bv_id, self._settings.data_dir),
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        out_path = self._settings.reports_dir / f"{lecture.bv_id}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Wrote lecture HTML: %s", out_path)
        return out_path

    def render_index(self, summaries: list, css_inline: str) -> str:
        from app.copilot.taxonomy import DOMAIN_WHITELIST

        tpl = self._env.get_template("index.html.j2")
        groups = group_summaries_by_domain(summaries)
        return tpl.render(
            summaries=summaries,
            groups=groups,
            domain_whitelist=list(DOMAIN_WHITELIST),
            css_inline=css_inline,
            index_cover_url=_summary_cover_url,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def load_copilot_css(self) -> str:
        css_path = Path(__file__).parent.parent / "static" / "copilot.css"
        if not css_path.exists():
            return ""
        try:
            return css_path.read_text(encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            logger.warning("Failed to read copilot.css: %s", exc)
            return ""

    def load_inline_css(self) -> str:
        css_path = Path(__file__).parent.parent / "static" / "lecture.css"
        if not css_path.exists():
            return ""
        return css_path.read_text(encoding="utf-8")
