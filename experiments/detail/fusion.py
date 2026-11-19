"""Learned pixel-wise combiner for the deep map and the detail branches.

The hand-tuned fusion (``detail_branch.BranchFusion``) requires choosing a
polarity and a weight, which transfers poorly: a scratch on sheet metal is
*brighter* than its surround while a can defect is *darker*. A small linear
model trained on synthetic defects learns the polarity and the weighting by
itself, using no defect labels at all.

Features per pixel
------------------
* ``deep``   the standardised deep map
* ``dark``   the standardised "darker than surround" branch
* ``bright`` the standardised "brighter than surround" branch
* ``deep_mean``  the local mean of the deep map (suppresses isolated noise)
* ``grad``   the local gradient magnitude (kept small so edges stay cheap)

Training uses synthetic defects injected into good training images. The
reference detector, the validation thresholds and the evaluation are all
fitted on good captures only, so the whole pipeline is protocol-compliant
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import uniform_filter

from local_contrast.detail_branch import BranchFusion, DEFAULT_RADIUS, MIN_SPREAD, luminance

FEATURE_NAMES = ("deep", "dark", "bright", "deep_mean", "grad")


def _local_mean(map_: np.ndarray, window: int) -> np.ndarray:
    return uniform_filter(map_, size=window, mode="reflect")


def _gradient(lum: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    mag = np.hypot(gx, gy)
    center = float(np.median(mag))
    spread = float(np.percentile(mag, 99) - np.percentile(mag, 50))
    return (mag - center) / max(spread, MIN_SPREAD)


def feature_matrix(fusion: BranchFusion, deep_map: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Compute the per-pixel feature matrix (rows = pixels) for one image."""
    parts = fusion.components(deep_map, image)
    lum = luminance(image)
    grad = _gradient(lum)
    feats = [parts["deep"], parts["dark"], parts["bright"], _local_mean(parts["deep"], 9), grad]
    return np.stack([f.ravel() for f in feats], axis=1)


class LearnedFusion:
    """Logistic-regression combiner trained on synthetic defects."""

    def __init__(self, radius: int = DEFAULT_RADIUS, model_kwargs: dict | None = None) -> None:
        self.radius = radius
        self._fusion = BranchFusion(radius=radius)
        self._model_kwargs = model_kwargs or {"C": 1.0, "max_iter": 400, "solver": "lbfgs"}
        self._model = None

    def fit_fusion(self, deep_maps: Iterable[np.ndarray], good_images: Iterable[np.ndarray]) -> "LearnedFusion":
        self._fusion.fit(deep_maps, good_images)
        return self

    def fit(self, deep_maps: list[np.ndarray], images: list[np.ndarray], masks: list[np.ndarray],
            background_ratio: float = 4.0, seed: int = 42,
            foregrounds: list[np.ndarray] | None = None) -> "LearnedFusion":
        """Fit the combiner on synthetic (deep map, image, mask) triples.

        Defect pixels are all used; background pixels are sampled ``background_ratio``
        times as many as there are defect pixels, so neither class dominates. If
        ``foregrounds`` is supplied, both classes are drawn from the object only,
        which keeps the combiner from learning "every bright backdrop pixel is
        boring" as a shortcut.
        """
        if not (len(deep_maps) == len(images) == len(masks)) or not deep_maps:
            raise ValueError("Pass matching lists of deep maps, images and masks")
        if foregrounds is not None and len(foregrounds) != len(masks):
            raise ValueError("Pass one foreground mask per image")
        rng = np.random.default_rng(seed)
        defect_features, background_features = [], []
        for index, (deep, image, mask) in enumerate(zip(deep_maps, images, masks)):
            mask = np.asarray(mask, dtype=bool)
            feats = feature_matrix(self._fusion, deep, image)
            flat_mask = mask.ravel()
            if foregrounds is not None:
                foreground = np.asarray(foregrounds[index], dtype=bool).ravel()
                flat_mask &= foreground
                background = foreground & ~flat_mask
            else:
                background = ~flat_mask
            if flat_mask.sum():
                defect_features.append(feats[flat_mask])
            if not int(background.sum()):
                background = ~flat_mask
            take = min(int(background.sum()), max(1, int(flat_mask.sum() * background_ratio)))
            pick = rng.choice(int(background.sum()), size=take, replace=False)
            background_features.append(feats[background][pick])
        if not defect_features:
            raise ValueError("No defect pixels found in the synthetic set")
        X = np.concatenate(defect_features + background_features)
        y = np.concatenate([np.ones(len(f), dtype=np.int32) for f in defect_features] +
                           [np.zeros(len(f), dtype=np.int32) for f in background_features])
        if len(np.unique(y)) < 2:
            raise ValueError("The synthetic set needs both defect and background pixels")
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(class_weight=None, **self._model_kwargs)
        self._model.fit(X, y)
        return self

    def scores(self, deep_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Return a continuous anomaly map (larger = more anomalous)."""
        if self._model is None:
            raise RuntimeError("Fit the combiner before scoring")
        feats = feature_matrix(self._fusion, deep_map, image)
        raw = self._model.decision_function(feats)
        return raw.reshape(np.asarray(deep_map).shape).astype(np.float32)

    def report(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Fit the combiner before reporting")
        coefs = self._model.coef_.ravel()
        return {name: float(value) for name, value in zip(FEATURE_NAMES, coefs)} | {"intercept": float(self._model.intercept_[0])}
