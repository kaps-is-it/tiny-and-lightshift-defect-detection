"""Native-resolution "unexpected local contrast" detail branch.

Why this exists
---------------
A tiled CNN detector ranks defects at only about the 94th percentile of their
own surrounding background. Two things it cannot do well:

1.  Polarity. In the public ``can`` split, 14 of 16 defect regions are *darker*
    than their surroundings in all six lighting conditions. Nearest-neighbour
    distances on CNN features do not know the sign of a deviation, so they fire
    on dark print and on bright specular bands alike.
2.  Per-pixel resolution. A 25-pixel defect spans roughly three stride-8 feature
    cells, so its signal is diluted by neighbouring cells and by the tile pad.

This module adds a second, native-resolution branch. It measures how much each
pixel deviates from a wide local surround, keeps one sign at a time, and is
standardised against good captures. See ``fusion.py`` for the learned combiner
that removes the need to assume a polarity.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import uniform_filter

LUMINANCE = np.array((0.299, 0.587, 0.114), dtype=np.float32)
DEFAULT_RADIUS = 201
MIN_SPREAD = 1e-6
POLARITIES = ("dark", "bright", "both")


def luminance(image: np.ndarray) -> np.ndarray:
    """Return a float32 HxW luminance image."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Expected an HxWx3 image")
    return (values[..., :3] * LUMINANCE[None, None, :]).sum(axis=-1)


def local_residual(lum: np.ndarray, radius: int = DEFAULT_RADIUS) -> np.ndarray:
    """Local surround minus pixel.

    Positive values mean the pixel is darker than its surround; negative values
    mean it is brighter. Keeping the sign is what lets this branch supply the
    polarity a CNN feature distance cannot.
    """
    if radius < 3 or radius % 2 == 0:
        raise ValueError("radius must be an odd integer of at least 3")
    return uniform_filter(lum, size=radius, mode="reflect") - lum


def one_sided(residual: np.ndarray, polarity: str) -> np.ndarray:
    """Turn a local residual into a non-negative detail map for one polarity."""
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}, got {polarity}")
    residual = _checked_2d(residual)
    if polarity == "dark":
        return np.maximum(residual, 0)
    if polarity == "bright":
        return np.maximum(-residual, 0)
    return np.abs(residual)


def _checked_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError("Expected a nonempty two-dimensional map")
    if not np.isfinite(array).all():
        raise ValueError("Map contains a nonfinite score")
    return array


def _summary(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    spread = float(np.percentile(values, 99) - np.percentile(values, 50))
    return center, max(spread, MIN_SPREAD)


class BranchFusion:
    """Standardise the deep map and both one-sided branches against good images.

    Fit on good validation captures only. The returned feature dict drives both
    fusion rules in this module and every rule in ``fusion.py``.
    """

    def __init__(self, radius: int = DEFAULT_RADIUS) -> None:
        self.radius = radius
        self._fitted = False
        self._deep_stats: tuple[float, float] | None = None
        self._branch_stats: dict[str, tuple[float, float]] | None = None

    def fit(self, deep_maps: Iterable[np.ndarray], good_images: Iterable[np.ndarray]) -> "BranchFusion":
        """Learn per-branch medians and upper spreads on good captures."""
        deep_maps = [np.asarray(m) for m in deep_maps]
        good_images = [image for image in good_images]
        if len(deep_maps) != len(good_images):
            raise ValueError("Pass one good image per deep map")
        if not deep_maps:
            raise ValueError("Fit the fusion with at least one good capture")
        flat_deep = np.concatenate([_checked_2d(m).ravel() for m in deep_maps])
        residuals = [local_residual(luminance(im), self.radius) for im in good_images]
        stats = {
            "deep": _summary(flat_deep),
            "dark": _summary(np.concatenate([np.maximum(r, 0).ravel() for r in residuals])),
            "bright": _summary(np.concatenate([np.maximum(-r, 0).ravel() for r in residuals])),
        }
        self._deep_stats = stats["deep"]
        self._branch_stats = {name: stats[name] for name in ("dark", "bright")}
        self._fitted = True
        return self

    def components(self, deep_map: np.ndarray, image: np.ndarray) -> dict[str, np.ndarray]:
        """Return the standardised deep map and one-sided branch maps."""
        self._require_fitted()
        assert self._deep_stats is not None and self._branch_stats is not None
        lum = luminance(image)
        residual = local_residual(lum, self.radius)
        dmu, dsd = self._deep_stats
        kmu, ksd = self._branch_stats["dark"]
        bmu, bsd = self._branch_stats["bright"]
        return {
            "deep": (_checked_2d(deep_map) - dmu) / dsd,
            "dark": (np.maximum(residual, 0) - kmu) / ksd,
            "bright": (np.maximum(-residual, 0) - bmu) / bsd,
        }

    def fuse(self, deep_map: np.ndarray, image: np.ndarray, weight: float, polarity: str = "dark") -> np.ndarray:
        """Hand-tuned fusion of the deep map with one detail branch."""
        parts = self.components(deep_map, image)
        if polarity == "dark":
            branch = parts["dark"]
        elif polarity == "bright":
            branch = parts["bright"]
        elif polarity == "both":
            branch = parts["dark"] + parts["bright"]
        else:
            raise ValueError(f"Unknown polarity: {polarity}")
        return parts["deep"] + weight * branch

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the fusion on good captures before transforming")
