"""Unit tests for :mod:`app.understand.frame_ranker`.

Covers the six contract points from M3 Phase 2:

1. black screen → LOW (filtered as junk by entropy threshold)
2. repeated identical slide → second copy LOW (filtered by similarity)
3. high-entropy text-dense slide → HIGH
4. floor ratio: at least ceil(N * 0.5) HIGH among non-junk frames
5. first frame: similarity_to_prev == 0.0 (no false junk classification)
6. ``enabled=False`` → all HIGH

Plus a couple of robustness tests (corrupt path → safe fallback,
empty input).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from app.understand.frame_ranker import (
    FrameSignals,
    KeyframeRanker,
    Tier,
    _dhash,
    _entropy,
    _hamming,
)


@dataclass
class _Frame:
    """Minimal Keyframe stand-in (avoids importing yt-dlp transitively)."""

    timestamp: float
    path: Path


def _save(img: Image.Image, path: Path) -> _Frame:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=85)
    return _Frame(timestamp=0.0, path=path)


def _black(path: Path, size: tuple[int, int] = (320, 180)) -> _Frame:
    return _save(Image.new("RGB", size, color=(0, 0, 0)), path)


def _solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (320, 180)) -> _Frame:
    return _save(Image.new("RGB", size, color=color), path)


def _noise(path: Path, seed: int = 0, size: tuple[int, int] = (320, 180)) -> _Frame:
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(size[0] * size[1])])
    return _save(img, path)


def _code_slide(path: Path, size: tuple[int, int] = (640, 360)) -> _Frame:
    """A high-entropy, edge-dense synthetic 'code' slide.

    Uses a noisy paper-like background (180-255 grayscale) so the
    histogram has many populated bins and entropy clears the default
    2.0 threshold. Layered black strokes provide the edge density
    needed to discriminate this from low-edge background frames.
    """
    rng = random.Random(42)
    img = Image.new("RGB", size)
    pixels = []
    for _ in range(size[0] * size[1]):
        v = 180 + rng.randint(0, 75)
        pixels.append((v, v, v))
    img.putdata(pixels)
    draw = ImageDraw.Draw(img)
    # Many short horizontal strokes simulate text lines without
    # depending on a system font.
    for row in range(20, size[1] - 20, 18):
        x = 20
        while x < size[0] - 20:
            length = 8 + (row * 7 + x) % 40
            shade = rng.randint(0, 60)
            draw.line(
                [(x, row), (x + length, row)],
                fill=(shade, shade, shade),
                width=2,
            )
            x += length + 6
        # An indented sub-line to bump edge density vertically.
        draw.line([(40, row + 6), (40, row + 12)], fill=(0, 0, 0), width=2)
    return _save(img, path)


# ---- signal sanity --------------------------------------------------


def test_first_frame_no_prev(tmp_path: Path) -> None:
    """The first frame must have similarity_to_prev == 0.0."""
    f = _noise(tmp_path / "f0.jpg", seed=1)
    ranker = KeyframeRanker()
    sigs = ranker.signals([f])
    assert sigs[0].similarity_to_prev == 0.0


def test_dhash_identical_is_zero_distance(tmp_path: Path) -> None:
    a = _code_slide(tmp_path / "a.jpg")
    b = _code_slide(tmp_path / "b.jpg")  # same content, byte-identical pixels
    with Image.open(a.path) as im_a, Image.open(b.path) as im_b:
        assert _hamming(_dhash(im_a), _dhash(im_b)) == 0


def test_entropy_black_is_zero(tmp_path: Path) -> None:
    f = _black(tmp_path / "k.jpg")
    with Image.open(f.path) as im:
        assert _entropy(im) < 0.5  # JPEG noise can lift it slightly above 0


# ---- tier classification --------------------------------------------


def test_black_screen_is_low_tier(tmp_path: Path) -> None:
    """Junk filter via entropy threshold must drop a black frame to LOW."""
    frames = [
        _noise(tmp_path / "rich.jpg", seed=1),  # informative
        _black(tmp_path / "blank.jpg"),  # junk by entropy
    ]
    tiers = KeyframeRanker().rank(frames)
    assert tiers[0] == Tier.HIGH
    assert tiers[1] == Tier.LOW


def test_repeated_slide_is_low_tier(tmp_path: Path) -> None:
    """A second / third identical slide must be filtered by similarity."""
    frames = [
        _code_slide(tmp_path / "s0.jpg"),
        _code_slide(tmp_path / "s1.jpg"),
        _code_slide(tmp_path / "s2.jpg"),
    ]
    tiers = KeyframeRanker().rank(frames)
    # The first slide must always be high — it's the first occurrence.
    assert tiers[0] == Tier.HIGH
    # Subsequent identical slides hit the similarity threshold and
    # become LOW (the floor still kicks in but there is exactly one
    # survivor, so floor=ceil(1*0.5)=1 keeps just the first slide).
    assert tiers[1] == Tier.LOW
    assert tiers[2] == Tier.LOW


def test_text_dense_slide_is_high_tier(tmp_path: Path) -> None:
    """A code-like slide with many edges must survive the junk filter.

    With ``high_floor_ratio=1.0`` every non-junk frame becomes HIGH,
    which directly verifies that the code slide is *not* dropped to
    LOW by the junk filter (entropy + similarity).
    """
    frames = [
        _noise(tmp_path / "n.jpg", seed=2),
        _code_slide(tmp_path / "c.jpg"),
    ]
    tiers = KeyframeRanker(high_floor_ratio=1.0).rank(frames)
    assert tiers == [Tier.HIGH, Tier.HIGH]

    # And independently, its raw signals must clear the junk thresholds.
    sigs = KeyframeRanker().signals([frames[1]])
    assert sigs[0].entropy >= 2.0
    assert sigs[0].edge_density > 0.02


def test_floor_ratio_guarantee(tmp_path: Path) -> None:
    """At least ceil(N * 0.5) survivors must be HIGH."""
    frames = [_noise(tmp_path / f"n{i}.jpg", seed=10 + i) for i in range(5)]
    tiers = KeyframeRanker().rank(frames)
    high_count = sum(1 for t in tiers if t == Tier.HIGH)
    assert high_count >= 3  # ceil(5 * 0.5) == 3


def test_disabled_returns_all_high(tmp_path: Path) -> None:
    frames = [
        _black(tmp_path / "k0.jpg"),
        _solid(tmp_path / "k1.jpg", (10, 10, 10)),
        _solid(tmp_path / "k2.jpg", (250, 250, 250)),
    ]
    tiers = KeyframeRanker(enabled=False).rank(frames)
    assert tiers == [Tier.HIGH, Tier.HIGH, Tier.HIGH]


def test_empty_input(tmp_path: Path) -> None:
    assert KeyframeRanker().rank([]) == []
    assert KeyframeRanker().signals([]) == []


def test_corrupt_path_falls_back_to_high(tmp_path: Path) -> None:
    """A frame whose file is unreadable must not crash the pipeline."""
    good = _noise(tmp_path / "good.jpg", seed=3)
    bad = _Frame(timestamp=1.0, path=tmp_path / "missing.jpg")
    tiers = KeyframeRanker().rank([good, bad])
    # Bad frame should still be ranked (HIGH on fallback, never crash).
    assert len(tiers) == 2
    assert tiers[1] == Tier.HIGH


def test_all_junk_still_keeps_floor(tmp_path: Path) -> None:
    """If every frame is junk we still emit ceil(N*floor) HIGH tiers.

    Otherwise the IR would render with zero captioned frames, which
    would be a correctness regression vs the legacy pipeline.
    """
    frames = [_black(tmp_path / f"b{i}.jpg") for i in range(4)]
    tiers = KeyframeRanker().rank(frames)
    high_count = sum(1 for t in tiers if t == Tier.HIGH)
    assert high_count == 2  # ceil(4 * 0.5)


def test_validation(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        KeyframeRanker(junk_sim_threshold=1.5)
    with pytest.raises(ValueError):
        KeyframeRanker(high_floor_ratio=-0.1)
    with pytest.raises(ValueError):
        KeyframeRanker(junk_entropy_threshold=-1.0)


def test_signals_dataclass() -> None:
    s = FrameSignals(similarity_to_prev=0.1, entropy=4.5, edge_density=0.2)
    assert s.similarity_to_prev == 0.1
    assert s.entropy == 4.5
    assert s.edge_density == 0.2


def test_perf_50_frames_under_2s(tmp_path: Path) -> None:
    """50-frame ranking must finish well under 2 seconds on CPU."""
    import time

    frames = [_noise(tmp_path / f"p{i}.jpg", seed=100 + i) for i in range(50)]
    t0 = time.perf_counter()
    tiers = KeyframeRanker().rank(frames)
    elapsed = time.perf_counter() - t0
    assert len(tiers) == 50
    assert elapsed < 2.0, f"ranker too slow: {elapsed:.2f}s"
