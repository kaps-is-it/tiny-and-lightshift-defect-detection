"""Make anomaly scores comparable across locations in a fixed camera view."""

from collections.abc import Iterable

import numpy as np
from PIL import Image


def _checked_map(values: np.ndarray) -> np.ndarray:
    anomaly_map = np.asarray(values, dtype=np.float32)
    if anomaly_map.ndim != 2 or not anomaly_map.size:
        raise ValueError("Expected a nonempty two-dimensional anomaly map")
    if not np.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map contains a nonfinite score")
    return anomaly_map


def _resize(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape == shape:
        return values
    height, width = shape
    return np.asarray(Image.fromarray(values).resize((width, height), Image.Resampling.BILINEAR))


class SpatialScorer:
    """Compare each pixel with the normal anomaly scores at the same location.

    Fit on good captures after fitting the detector. Map shapes and camera
    alignment must agree across the fit and test images. Use a separate group
    of good captures to set the part rejection threshold.
    """

    def __init__(self) -> None:
        self._reference: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, maps: Iterable[np.ndarray]) -> "SpatialScorer":
        """Fit a median and robust spread at each location on a coarse grid."""
        coarse_maps = []
        native_shape = grid_shape = None
        for values in maps:
            anomaly_map = _checked_map(values)
            if native_shape is None:
                native_shape = anomaly_map.shape
                ratio = min(1.0, 128 / min(native_shape), 512 / max(native_shape))
                grid_shape = tuple(max(1, round(length * ratio)) for length in native_shape)
            if anomaly_map.shape != native_shape:
                raise ValueError("Spatial reference maps must have the same shape")
            coarse_maps.append(_resize(anomaly_map, grid_shape).copy())

        if len(coarse_maps) < 2:
            raise ValueError("Fit the spatial reference with at least two good maps")

        low, center, high = np.quantile(np.stack(coarse_maps), [0.25, 0.5, 0.75], axis=0)
        spread = (high - low) / 1.349
        typical_spread = np.median(spread)
        typical_score = np.percentile(np.abs(center), 75)
        # Keep background locations with little variation from dominating the score.
        floor = max(0.25 * typical_spread, 0.01 * typical_score, np.finfo(np.float32).eps)
        self._reference = (
            _resize(center.astype(np.float32), native_shape),
            _resize(np.maximum(spread, floor).astype(np.float32), native_shape),
        )
        return self

    def transform(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Return positive standardized residuals in the original image coordinates."""
        if self._reference is None:
            raise RuntimeError("Fit a spatial reference before transforming anomaly maps")
        center, scale = self._reference
        values = _checked_map(anomaly_map)
        if values.shape != center.shape:
            raise ValueError(f"Expected anomaly map shape {center.shape}, got {values.shape}")
        residual = values - center
        np.maximum(residual, 0, out=residual)
        return np.divide(residual, scale, out=residual)

    @staticmethod
    def score(normalized_map: np.ndarray) -> float:
        """Average the strongest 0.01% of the pixel scores, rounding the count up."""
        scores = _checked_map(normalized_map).ravel()
        count = (scores.size + 9_999) // 10_000
        cutoff = scores.size - count
        return float(np.partition(scores, cutoff)[cutoff:].mean(dtype=np.float64))
