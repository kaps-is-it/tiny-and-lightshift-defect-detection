"""Score sharp local peaks in a full-resolution anomaly map."""

import numpy as np
from scipy.ndimage import uniform_filter

from local_contrast.spatial_scoring import SpatialScorer


BACKGROUND_WINDOW = 65  # Slightly wider than eight feature strides in a native tile.


def local_contrast(anomaly_map: np.ndarray) -> np.ndarray:
    """Subtract the local mean and retain positive residuals at native resolution."""
    background = uniform_filter(anomaly_map, size=BACKGROUND_WINDOW, mode="reflect")
    return np.maximum(anomaly_map - background, 0)


def local_contrast_score(anomaly_map: np.ndarray) -> float:
    """Average the top 0.01% of the contrast map without changing the input."""
    return SpatialScorer.score(local_contrast(anomaly_map))
