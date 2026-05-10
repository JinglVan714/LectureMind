"""Pure-CPU pre-classifier for keyframes feeding the VLM.

The pipeline today sends every keyframe through Qwen-VL with the full
caption + OCR + visual_type + scoring prompt. That is ~1k input tokens
+ ~200 output tokens *per frame*, paid even for visually trivial
frames (talking-head shots, repeated PowerPoint slides, black
transitions).

This module decides — without any model call — which frames warrant
the expensive HIGH-tier prompt and which can settle for a lightweight
OCR-only LOW-tier prompt. The decision uses three signals computed
with Pillow standard ops:

* **dHash similarity to previous frame** — repeats a slide we already
  described.
* **Shannon entropy** of the grayscale histogram — black / blank
  frames score near 0.
* **Edge density** via ``ImageFilter.FIND_EDGES`` — text-dense slides
  (code, formulas) light up.

The classifier returns a parallel list of :class:`Tier` values, one
per input frame. The mixed strategy (D3 of the architecture
roadmap) is:

1. Absolute junk filter: ``similarity ≥ JUNK_SIM_THRESHOLD`` *or*
   ``entropy < JUNK_ENTROPY_THRESHOLD`` → forced LOW (no protection).
2. Among survivors, keep at least ``ceil(N * HIGH_FLOOR_RATIO)`` as
   HIGH, picked by composite score
   ``0.5 (1 - sim) + 0.3 entropy + 0.2 edge_density``.

When ``enabled=False`` the ranker becomes a no-op: every frame is
HIGH (matches the legacy pipeline behaviour for the
``VLM_TIERING_ENABLED=false`` escape hatch).

Tier inherits from ``str`` so the enum members are accepted directly
by :class:`app.understand.vlm_cache.VLMCache` without any conversion.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from PIL import Image, ImageFilter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ingest.keyframe import Keyframe

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass
class RankResult:
    """Bundle returned by :meth:`KeyframeRanker.classify`.

    ``tiers``, ``signals`` and ``is_junk`` are parallel lists with one
    entry per input frame. Exposed so callers (notably
    :class:`FrameDescriber`) can produce accurate ``junk_filtered``
    telemetry without a second pass over the frames.
    """

    tiers: list["Tier"]
    signals: list["FrameSignals"]
    is_junk: list[bool]


@dataclass
class FrameSignals:
    """Per-frame CPU-only descriptors produced by :class:`KeyframeRanker`.

    Attributes
    ----------
    similarity_to_prev:
        1.0 if dHash bits are identical to the previous frame, 0.0 if
        all 64 bits differ. The first frame has 0.0 (no prev).
    entropy:
        Shannon entropy of the grayscale 8-bit histogram. Range
        [0, 8]. Solid colour ≈ 0; natural scene ≈ 7+.
    edge_density:
        Fraction of pixels above the FIND_EDGES threshold (50/255).
        Range [0, 1]. Code / formula slides are typically > 0.05.
    """

    similarity_to_prev: float
    entropy: float
    edge_density: float


# Internal weighted score used to pick HIGH tier among survivors.
def _score(sig: FrameSignals) -> float:
    return 0.5 * (1.0 - sig.similarity_to_prev) + 0.3 * sig.entropy + 0.2 * sig.edge_density


class KeyframeRanker:
    """Classify a list of keyframes into HIGH / LOW VLM-prompt tiers."""

    def __init__(
        self,
        *,
        junk_sim_threshold: float = 0.93,
        junk_entropy_threshold: float = 2.0,
        high_floor_ratio: float = 0.5,
        enabled: bool = True,
    ) -> None:
        if not 0.0 <= junk_sim_threshold <= 1.0:
            raise ValueError("junk_sim_threshold must be in [0, 1]")
        if junk_entropy_threshold < 0.0:
            raise ValueError("junk_entropy_threshold must be >= 0")
        if not 0.0 <= high_floor_ratio <= 1.0:
            raise ValueError("high_floor_ratio must be in [0, 1]")
        self._sim_thr = float(junk_sim_threshold)
        self._entropy_thr = float(junk_entropy_threshold)
        self._floor = float(high_floor_ratio)
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def signals(self, frames: "list[Keyframe]") -> list[FrameSignals]:
        """Compute :class:`FrameSignals` for each input frame.

        On a per-frame Pillow exception (corrupt JPEG, missing file)
        the signal falls back to ``(0.0, 8.0, 1.0)`` which keeps the
        frame out of the junk filter and gives it the maximum
        composite score → safe HIGH tier.
        """
        results: list[FrameSignals] = []
        prev_hash: int | None = None
        for frame in frames:
            try:
                with Image.open(frame.path) as im:
                    im.load()
                    h = _dhash(im)
                    e = _entropy(im)
                    d = _edge_density(im)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "KeyframeRanker: signal computation failed for %s: %s — "
                    "falling back to safe HIGH-tier defaults",
                    getattr(frame, "path", frame),
                    exc,
                )
                results.append(FrameSignals(0.0, 8.0, 1.0))
                prev_hash = None
                continue

            if prev_hash is None:
                sim = 0.0
            else:
                sim = 1.0 - (_hamming(h, prev_hash) / 64.0)
            results.append(FrameSignals(similarity_to_prev=sim, entropy=e, edge_density=d))
            prev_hash = h
        return results

    def rank(self, frames: "list[Keyframe]") -> list[Tier]:
        """Return one :class:`Tier` per frame, in input order."""
        return self.classify(frames).tiers

    def classify(self, frames: "list[Keyframe]") -> RankResult:
        """Return tiers + per-frame signals + per-frame junk mask.

        Computing them together avoids the double-pass that calling
        :meth:`signals` and :meth:`rank` separately would incur, and
        gives :class:`FrameDescriber` the data it needs for accurate
        ``junk_filtered`` telemetry.
        """
        if not frames:
            return RankResult(tiers=[], signals=[], is_junk=[])
        if not self._enabled:
            sigs = self.signals(frames)
            return RankResult(
                tiers=[Tier.HIGH] * len(frames),
                signals=sigs,
                is_junk=[False] * len(frames),
            )

        sigs = self.signals(frames)
        is_junk = [
            (s.similarity_to_prev >= self._sim_thr) or (s.entropy < self._entropy_thr)
            for s in sigs
        ]
        survivors = [i for i, j in enumerate(is_junk) if not j]
        if not survivors:
            # Edge case: every frame flagged junk. Still respect the
            # floor — give the highest-scoring frames HIGH so we don't
            # silently emit a LectureIR with zero captioned frames.
            scored = sorted(range(len(sigs)), key=lambda i: _score(sigs[i]), reverse=True)
            k = max(1, math.ceil(len(frames) * self._floor))
            high = set(scored[:k])
            tiers = [Tier.HIGH if i in high else Tier.LOW for i in range(len(frames))]
            return RankResult(tiers=tiers, signals=sigs, is_junk=is_junk)

        scores = {i: _score(sigs[i]) for i in survivors}
        # Floor is computed against the survivor set, per spec §2.1.
        k = max(1, math.ceil(len(survivors) * self._floor))
        high = set(sorted(survivors, key=lambda i: scores[i], reverse=True)[:k])
        tiers = [Tier.HIGH if i in high else Tier.LOW for i in range(len(frames))]
        return RankResult(tiers=tiers, signals=sigs, is_junk=is_junk)


# ---- pure-function signal helpers (also used by tests) ---------------


def _dhash(img: Image.Image) -> int:
    """Compute a 64-bit difference hash from a 9x8 grayscale resample."""
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(g.getdata())
    bits = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            left = pixels[base + col]
            right = pixels[base + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _entropy(img: Image.Image) -> float:
    g = img.convert("L")
    hist = g.histogram()
    total = sum(hist) or 1
    h = 0.0
    for c in hist:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def _edge_density(img: Image.Image, threshold: int = 50) -> float:
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    if not pixels:
        return 0.0
    return sum(1 for v in pixels if v > threshold) / len(pixels)
