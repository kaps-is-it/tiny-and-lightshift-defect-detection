"""Evaluation harness for the MVTec AD 2 experiments.

Adds the measurements the earlier reports were missing:

* a trivial, learning-free reference row (Sobel gradient, darkness) so every
  AU-PRO number has a floor to be compared against;
* a worst-lighting summary in addition to the mean, because the mean hides the
  fact that one lighting condition is consistently the hardest;
* cross-light agreement of the false-positive hotspot set, which is what a
  lighting change actually breaks;
* a per-defect-region percentile ranking inside its own image.

AU-PRO, tiny-PRO and the decision metrics come from ``evaluation.py`` and are
reported per lighting condition.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import sobel

from evaluation import ImageResult, summarize


def trivial_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    """Learning-free maps used to set a reference floor."""
    lum = np.asarray(image.convert("L"), dtype=np.float32)
    return {
        "darkness": 255.0 - lum,
        "sobel": np.hypot(sobel(lum, axis=0), sobel(lum, axis=1)).astype(np.float32),
    }


def defect_percentile(results: list[ImageResult]) -> float:
    """Mean percentile of the defect inside its own image's background pixels."""
    values = []
    for result in results:
        if result.mask is None:
            continue
        background = result.anomaly_map[~result.mask]
        values.append(float((background < result.anomaly_map[result.mask].mean()).mean() * 100))
    return float(np.mean(values)) if values else float("nan")


def fp_agreement(top_sets: dict[tuple[str, str], set[tuple[int, int]]]) -> float:
    """Mean IoU of the top-0.01% false-positive pixel sets, regular vs shifted."""
    ious = []
    for (scene, light), pixels in top_sets.items():
        if light == "regular":
            continue
        reference = top_sets.get((scene, "regular"))
        if reference is None:
            continue
        union = len(reference | pixels)
        if union:
            ious.append(len(reference & pixels) / union)
    return float(np.mean(ious)) if ious else float("nan")


def top_fp_sets(results: list[ImageResult], light: str, fraction: float = 0.0001) -> dict[tuple[str, str], set[tuple[int, int]]]:
    """Collect the top-scoring pixel set of each good image (a false positive)."""
    sets: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for result in results:
        if result.is_defective:
            continue
        count = max(1, int(result.anomaly_map.size * fraction))
        flat = np.argpartition(result.anomaly_map.ravel(), -count)[-count:]
        yy, xx = np.unravel_index(flat, result.anomaly_map.shape)
        sets[(result.scene, light)] = set(zip(yy.tolist(), xx.tolist()))
    return sets


def summarise_results(results: list[ImageResult], threshold: float = 0.0) -> dict:
    """Decision-free metrics for one lighting group."""
    summary = summarize(results, threshold)
    summary["defect_percentile"] = defect_percentile(results)
    return summary


def summarise_by_light(by_light: dict[str, list[ImageResult]], threshold: float = 0.0) -> dict:
    """Per-light metrics plus the mean and worst-lighting numbers."""
    lights = sorted(by_light, key=lambda name: (name != "regular", name))
    per_light = {light: summarise_results(by_light[light], threshold) for light in lights}
    au = [per_light[l]["au_pro_05"] for l in lights]
    tiny = [per_light[l]["tiny_pro_at_05"] for l in lights]
    pct = [per_light[l]["defect_percentile"] for l in lights]
    by_light_ids = {light: [r for r in by_light[light]] for light in lights}
    fp_sets = {}
    for light in lights:
        fp_sets.update(top_fp_sets(by_light_ids[light], light))
    by_light_ids_fp = fp_agreement(fp_sets)
    return {
        "per_light": per_light,
        "au_pro_mean": float(np.mean(au)),
        "au_pro_worst": float(np.min(au)),
        "tiny_pro_mean": float(np.mean(tiny)),
        "defect_percentile": float(np.mean(pct)),
        "fp_agreement": by_light_ids_fp,
        "worst_light": lights[int(np.argmin(au))],
    }


def compare_region_level(left: list[ImageResult], right: list[ImageResult]) -> dict:
    """Paired comparison of two maps on the same defect regions.

    For each defective 8-connected region, report how much higher it ranks in
    its own image background. Positive deltas mean the right-hand map
    localises the defect better. A paired sign test reports whether the
    improvement is systematic rather than luck.
    """
    if len(left) != len(right):
        raise ValueError("Compare maps produced for the same scenes")
    from scipy import ndimage
    from scipy.stats import binomtest

    deltas = []
    for a, b in zip(left, right):
        if a.mask is None:
            continue
        if a.anomaly_map.shape != b.anomaly_map.shape or a.anomaly_map.shape != a.mask.shape:
            raise ValueError("Map and mask shapes disagree")
        regions, count = ndimage.label(a.mask, structure=np.ones((3, 3), dtype=bool))
        for index in range(1, count + 1):
            region = regions == index
            pa = float((a.anomaly_map[~region] < a.anomaly_map[region].mean()).mean())
            pb = float((b.anomaly_map[~region] < b.anomaly_map[region].mean()).mean())
            deltas.append(pb - pa)
    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        test = None
    else:
        wins = sum(1 for d in nonzero if d > 0)
        test = float(binomtest(wins, len(nonzero), 0.5).pvalue)
    return {
        "regions": len(deltas),
        "mean_delta_percentile": float(np.mean(deltas)) if deltas else None,
        "median_delta_percentile": float(np.median(deltas)) if deltas else None,
        "regions_improved": int(sum(1 for d in deltas if d > 0)),
        "regions_worsened": int(sum(1 for d in deltas if d < 0)),
        "sign_test_p": test,
    }
