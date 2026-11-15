"""Metrics and previews for the public MVTec experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.stats import rankdata


PIXEL_FPR_LIMIT = 0.05
PRO_BINS = 4096


@dataclass
class ImageResult:
    image: str
    scene: str
    lighting: str
    is_defective: bool
    score: float
    anomaly_map: np.ndarray
    mask: np.ndarray | None


def _localization_metrics(results: list[ImageResult], tiny_area_ratio: float) -> dict:
    """Approximate PRO with a shared score histogram across original-size maps.

    Each defect region contributes the same total weight. The curve is integrated
    to 5% pixel false positive rate and divided by that width. Scores in a common
    bin are treated as ties, so small differences from exact sweeps are expected.
    """
    low = np.inf
    high = -np.inf
    for result in results:
        scores = result.anomaly_map
        if scores.ndim != 2 or scores.size == 0:
            raise ValueError(f"Expected a nonempty 2D anomaly map for {result.image}")
        image_low, image_high = float(scores.min()), float(scores.max())
        if not (np.isfinite(image_low) and np.isfinite(image_high)):
            raise ValueError(f"Non-finite anomaly map for {result.image}")
        low = min(low, image_low)
        high = max(high, image_high)

    background = np.zeros(PRO_BINS, dtype=np.int64)
    overlap = np.zeros(PRO_BINS, dtype=np.float64)
    tiny_overlap = np.zeros_like(overlap)
    region_count = tiny_count = 0
    scale = (PRO_BINS - 1) / (high - low) if high > low else 0.0
    connectivity = np.ones((3, 3), dtype=bool)

    for result in results:
        scores = result.anomaly_map
        bins = np.clip((scores - low) * scale, 0, PRO_BINS - 1).astype(np.int32)
        background += np.bincount(bins.ravel(), minlength=PRO_BINS)
        if result.mask is None:
            if result.is_defective:
                raise ValueError(f"Missing defect mask for {result.image}")
            continue

        foreground = result.mask.astype(bool, copy=False)
        if foreground.shape != scores.shape:
            raise ValueError(f"Mask shape differs from the anomaly map for {result.image}")
        if bool(foreground.any()) != result.is_defective:
            raise ValueError(f"Mask content disagrees with the label for {result.image}")
        if not result.is_defective:
            continue

        regions, count = ndimage.label(foreground, structure=connectivity)
        regions = regions[foreground]
        positive_bins = bins[foreground]
        region_sizes = np.bincount(regions, minlength=count + 1)[1:]
        pixel_weights = 1.0 / region_sizes[regions - 1]
        tiny_regions = region_sizes <= tiny_area_ratio * scores.size
        tiny_pixels = tiny_regions[regions - 1]

        background -= np.bincount(positive_bins, minlength=PRO_BINS)
        overlap += np.bincount(positive_bins, pixel_weights, minlength=PRO_BINS)
        tiny_overlap += np.bincount(
            positive_bins[tiny_pixels], pixel_weights[tiny_pixels], minlength=PRO_BINS
        )
        region_count += count
        tiny_count += int(tiny_regions.sum())

    metrics = {
        "au_pro_05": None,
        "tiny_pro_at_05": None,
        "defect_regions": region_count,
        "tiny_regions": tiny_count,
    }
    negative_pixels = int(background.sum())
    if not negative_pixels or not region_count:
        return metrics

    # Sweep from the highest scores down, including the empty prediction.
    fpr = np.r_[0.0, np.cumsum(background[::-1]) / negative_pixels]
    pro = np.r_[0.0, np.cumsum(overlap[::-1]) / region_count]
    last = np.searchsorted(fpr, PIXEL_FPR_LIMIT, side="right")
    clipped_fpr = np.r_[fpr[:last], PIXEL_FPR_LIMIT]
    clipped_pro = np.r_[pro[:last], np.interp(PIXEL_FPR_LIMIT, fpr, pro)]
    area = np.sum(np.diff(clipped_fpr) * (clipped_pro[:-1] + clipped_pro[1:]) / 2)
    metrics["au_pro_05"] = float(np.clip(area / PIXEL_FPR_LIMIT, 0, 1))

    if tiny_count:
        tiny_pro = np.r_[0.0, np.cumsum(tiny_overlap[::-1]) / tiny_count]
        metrics["tiny_pro_at_05"] = float(
            np.clip(np.interp(PIXEL_FPR_LIMIT, fpr, tiny_pro), 0, 1)
        )
    return metrics


def summarize_decisions(results: list[ImageResult], threshold: float) -> dict:
    """Measure part-level decisions and ranking at a fixed score threshold."""
    if not results:
        raise ValueError("No images to evaluate")
    scores = np.array([result.score for result in results], dtype=np.float64)
    if not (np.isfinite(scores).all() and np.isfinite(threshold)):
        raise ValueError("Image scores and threshold must be finite")

    defective = np.array([result.is_defective for result in results], dtype=bool)
    rejected = scores >= threshold
    tp = int(np.count_nonzero(defective & rejected))
    tn = int(np.count_nonzero(~defective & ~rejected))
    fp = int(np.count_nonzero(~defective & rejected))
    fn = int(np.count_nonzero(defective & ~rejected))
    positives = tp + fn
    negatives = tn + fp
    f1_denominator = 2 * tp + fp + fn
    image_auroc = None
    if positives and negatives:
        positive_rank_sum = rankdata(scores)[defective].sum()
        image_auroc = float(
            (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
        )

    return {
        "images": len(results),
        "defective": positives,
        "good": negatives,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": tp / positives if positives else None,
        "false_reject_rate": fp / negatives if negatives else None,
        "image_f1": 2 * tp / f1_denominator if f1_denominator else None,
        "image_auroc": image_auroc,
    }


def summarize(
    results: list[ImageResult], threshold: float, tiny_area_ratio: float = 0.0001
) -> dict:
    """Summarize decisions and pixel localization for one lighting group."""
    if not 0 < tiny_area_ratio <= 1:
        raise ValueError("Tiny region area ratio must be in (0, 1]")
    return {**summarize_decisions(results, threshold), **_localization_metrics(results, tiny_area_ratio)}


def save_overlay(
    image_path: Path, result: ImageResult, out: Path, scale: tuple[float, float]
) -> None:
    """Render the input, anomaly overlay, and pixel labels on a shared layout."""
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    if result.anomaly_map.shape != (original.height, original.width):
        raise ValueError(f"Anomaly map shape does not match the source for {result.image}")

    preview = original.copy()
    preview.thumbnail((360, 320), Image.Resampling.LANCZOS)
    width, height = preview.size
    score_preview = Image.fromarray(result.anomaly_map.astype(np.float32, copy=False))
    scores = np.asarray(score_preview.resize(preview.size, Image.Resampling.BILINEAR))
    low, high = scale
    strength = np.clip((scores - low) / max(high - low, 1e-8), 0.0, 1.0)
    levels = np.linspace(0.0, 1.0, 5)
    palette = np.array([[20, 24, 58], [44, 53, 112], [139, 56, 112], [232, 95, 68], [255, 222, 124]])
    colors = np.stack([np.interp(strength, levels, channel) for channel in palette.T], axis=-1)
    alpha = (0.7 * strength)[..., None]
    heatmap = Image.fromarray((np.asarray(preview) * (1 - alpha) + colors * alpha).astype(np.uint8))

    mask = np.zeros((height, width), dtype=np.uint8)
    if result.mask is not None:
        if result.mask.shape != result.anomaly_map.shape:
            raise ValueError(f"Mask shape differs from the anomaly map for {result.image}")
        # Keep subpixel defects visible when downsampling the label preview.
        ys, xs = np.nonzero(result.mask)
        mask[ys * height // original.height, xs * width // original.width] = 255

    gap = 12
    top = 62
    canvas = Image.new("RGB", (3 * width + 4 * gap, top + height + gap), (19, 23, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), f"{result.scene} | {result.lighting} | score {result.score:.3f}", fill="white")
    panels = (("Input", preview), ("Anomaly overlay", heatmap), ("Ground-truth mask", Image.fromarray(mask)))
    for index, (label, panel) in enumerate(panels):
        x = gap + index * (width + gap)
        draw.text((x, 37), label, fill=(193, 204, 217))
        canvas.paste(panel, (x, top))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
