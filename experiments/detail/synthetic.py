"""Synthetic defect injection for training the learned fusion combiner.

This is deliberately simple and fully self-contained. It creates small,
plausibly-realistic defect-like regions (scratches, pits, blobs) in native
resolution with a known mask, and blends them without leaving an obvious
paste boundary.

Synthetic defects are used *only* to teach the fusion combiner which channel
carries defect evidence. They are never used to fit the reference detector,
the validation thresholds, or the evaluation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DEFECT_KINDS = ("scratch", "pit", "inclusion", "patch")


def foreground_mask(paths, stride: int = 16, mean_below: float = 250.0, var_above: float = 25.0) -> np.ndarray:
    """Static object silhouette learned from good training captures.

    The can is fixed in place across the training split while the backdrop is a
    flat white paper, so a per-pixel "ever interesting" statistic separates
    object from background without any masks. Returns a boolean HxW mask.
    """
    stack = []
    for path in list(paths)[::stride]:
        with Image.open(path) as opened:
            stack.append(np.asarray(opened.convert("L"), dtype=np.float32))
    if not stack:
        raise ValueError("foreground_mask needs at least one source image")
    stack = np.stack(stack)
    return (stack.mean(axis=0) < mean_below) | (stack.var(axis=0) > var_above)


def _feather(mask: np.ndarray, feather: float) -> np.ndarray:
    """Turn a hard mask into a soft alpha in [0, 1]."""
    if feather <= 0:
        return mask.astype(np.float32)
    soft = gaussian_filter(mask.astype(np.float32), sigma=feather, mode="reflect")
    return np.clip(soft, 0.0, 1.0)


def _scratch(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    angle = rng.uniform(0, np.pi)
    length = rng.uniform(0.05, 0.35) * max(h, w)
    half = rng.uniform(1.2, 4.0)
    n = int(length / 2) + 2
    ts = np.linspace(-length / 2, length / 2, n)
    xs = cx + ts * np.cos(angle)
    ys = cy + ts * np.sin(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), np.inf, dtype=np.float32)
    for x, y in zip(xs, ys):
        best = np.minimum(best, np.hypot(xx - x, yy - y))
    return (best <= half).astype(np.float32)


def _ellipse(mask: np.ndarray, rng: np.random.Generator, radius_lo: float, radius_hi: float,
            aspect: tuple[float, float] = (0.4, 1.0)) -> np.ndarray:
    h, w = mask.shape
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    r = rng.uniform(radius_lo, radius_hi)
    rx = r * rng.uniform(*aspect)
    ry = r
    angle = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    return ((u / rx) ** 2 + (v / ry) ** 2 <= 1.0).astype(np.float32)


def _polygon(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    n = int(rng.integers(5, 10))
    radii = rng.uniform(0.02, 0.09, size=n) * max(h, w)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n))
    points = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=-1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), np.inf, dtype=np.float32)
    for i in range(n):
        p0, p1 = points[i], points[(i + 1) % n]
        seg = p1 - p0
        length2 = float(seg @ seg) + 1e-8
        t = np.clip(((xx - p0[0]) * seg[0] + (yy - p0[1]) * seg[1]) / length2, 0.0, 1.0)
        dist = np.hypot(xx - (p0[0] + t * seg[0]), yy - (p0[1] + t * seg[1]))
        best = np.minimum(best, dist)
    return (best <= 2.0).astype(np.float32)


def make_mask(shape: tuple[int, int], rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
    """Create a binary defect-like mask of roughly the target size."""
    kind = kind or rng.choice(DEFECT_KINDS)
    temp = np.zeros(shape, dtype=np.float32)
    if kind == "scratch":
        return _scratch(temp, rng)
    if kind == "pit":
        return _ellipse(temp, rng, radius_lo=3.0, radius_hi=9.0, aspect=(0.5, 1.0))
    if kind == "inclusion":
        return _ellipse(temp, rng, radius_lo=2.0, radius_hi=6.0, aspect=(0.6, 1.0))
    if kind == "patch":
        return _polygon(temp, rng)
    raise ValueError(f"Unknown synthetic defect kind: {kind}")


def _texture_from_image(image: np.ndarray, rng: np.random.Generator, polarity: str) -> np.ndarray:
    """Build a plausible defect appearance: dark or bright, mildly textured."""
    lum = (np.asarray(image, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)
    base = (np.roll(np.roll(lum, 137, axis=0), 173, axis=1) + np.roll(np.roll(lum, 71, axis=0), 983, axis=1)) * 0.5
    local = gaussian_filter(base, sigma=3.0, mode="reflect")
    noise = gaussian_filter(rng.normal(0.0, 1.0, size=lum.shape).astype(np.float32), sigma=1.5, mode="reflect")
    if polarity == "dark":
        strength = rng.uniform(45, 110)
        return np.clip(local - strength * (0.75 + 0.25 * noise), 0, 255)
    strength = rng.uniform(45, 110)
    return np.clip(local + strength * (0.75 + 0.25 * noise), 0, 255)


def inject(image: np.ndarray, rng: np.random.Generator, kind: str | None = None,
           polarity: str | None = None, max_attempts: int = 16,
           foreground: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (image with an injected defect, binary defect mask).

    If ``foreground`` is given, the defect mass is required to land on the
    object so training pixels are never backdrop pixels.
    """
    lum = np.asarray(image, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if polarity is None:
        polarity = "dark" if rng.random() < 0.7 else "bright"  # match the observed real bias
    for _ in range(max_attempts):
        mask = make_mask(lum.shape, rng, kind)
        mass = int(mask.sum())
        if mass < 8 or mass > 2000:
            continue
        if foreground is not None and foreground.shape == mask.shape:
            ys, xs = np.nonzero(mask)
            if not foreground[ys, xs].any():
                continue
        alpha = _feather(mask, feather=float(rng.uniform(1.0, 2.5)))[..., None]
        alpha = alpha * rng.uniform(0.75, 1.0)
        appearance = _texture_from_image(image, rng, polarity)[..., None].repeat(3, axis=2)
        blended = np.clip((np.asarray(image, dtype=np.float32) * (1 - alpha) + appearance * alpha), 0, 255)
        return blended.astype(np.float32), mask > 0.5
    raise RuntimeError("Could not place a synthetic defect")


def generate(image: np.ndarray, count: int, rng: np.random.Generator, kinds: tuple[str, ...] = DEFECT_KINDS):
    """Yield (defective_image, mask) pairs for one good source image."""
    for _ in range(count):
        yield inject(image, rng, kind=rng.choice(kinds) if kinds else None)
