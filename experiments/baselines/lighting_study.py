"""Study lighting alerts on good scenes and defect visibility before resizing."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy import ndimage

from local_contrast.build_scene_manifest import build_manifest


FEATURES = ("median_luma", "luma_range_05_95", "near_black", "near_white")
SCALE_FLOORS = np.array([1 / 255, 1 / 255, 0.001, 0.001])
BATCH_SIZE = 4
SEED = 42
ALERT_QUANTILE = 0.90
INPUT_LONG_EDGE = 256
TINY_AREA_FRACTION = 0.0001
CONNECTED = np.ones((3, 3), dtype=bool)


def quality_features(gray: Image.Image) -> np.ndarray:
    """Measure whole-frame luma and near-clipping fractions in the decoded image."""
    histogram = np.bincount(np.asarray(gray).ravel(), minlength=256)
    total = histogram.sum()
    cumulative = histogram.cumsum()
    low, mid, high = np.searchsorted(cumulative, np.ceil(np.array([0.05, 0.50, 0.95]) * total))
    return np.array([mid / 255, (high - low) / 255, histogram[:6].sum() / total, histogram[250:].sum() / total])


def batch_indices(size: int) -> np.ndarray:
    if size < BATCH_SIZE:
        raise ValueError(f"At least {BATCH_SIZE} good scenes are required for batch replay")
    return np.asarray(list(combinations(range(size), BATCH_SIZE)))


def calibration(dataset: Path, category: str) -> tuple[np.ndarray, np.ndarray, dict]:
    validation = sorted((dataset / category / "validation" / "good").glob("*.png"))
    if len(validation) < 2 * BATCH_SIZE or any(not path.stem.endswith("_regular") for path in validation):
        raise ValueError(f"Expected at least {2 * BATCH_SIZE} good regular-light validation images for {category}")

    features = []
    for path in validation:
        with Image.open(path) as image:
            features.append(quality_features(image.convert("L")))
    values = np.stack(features)
    reference = sorted(random.Random(SEED).sample(range(len(validation)), len(validation) // 2))
    held_out = sorted(set(range(len(validation))) - set(reference))
    center = np.median(values[reference], axis=0)
    mad = np.median(np.abs(values[reference] - center), axis=0)
    scale = np.maximum(1.4826 * mad, SCALE_FLOORS)
    scores = np.max(np.abs(values[held_out] - center) / scale, axis=1)
    batch_scores = np.median(scores[batch_indices(len(held_out))], axis=1)
    report = {
        "seed": SEED,
        "reference_images": [validation[i].relative_to(dataset).as_posix() for i in reference],
        "threshold_images": [validation[i].relative_to(dataset).as_posix() for i in held_out],
        "feature_centers": dict(zip(FEATURES, center.tolist(), strict=True)),
        "feature_scales": dict(zip(FEATURES, scale.tolist(), strict=True)),
        "quantile": ALERT_QUANTILE,
        "scene_threshold": float(np.quantile(scores, ALERT_QUANTILE)),
        "batch_size": BATCH_SIZE,
        "threshold_batch_combinations": len(batch_scores),
        "batch_threshold": float(np.quantile(batch_scores, ALERT_QUANTILE)),
    }
    return center, scale, report


def visibility(gray: Image.Image, mask_image: Image.Image) -> dict:
    """Measure mask support and nearby luma contrast at two image scales."""
    native = np.asarray(gray)
    mask = np.asarray(mask_image) != 0
    if mask.shape != native.shape or not mask.any():
        raise ValueError("A defective capture must have a nonempty mask at the original image size")
    ys, xs = mask.nonzero()
    y0, y1 = max(0, ys.min() - 3), min(gray.height, ys.max() + 4)
    x0, x1 = max(0, xs.min() - 3), min(gray.width, xs.max() + 4)
    crop = mask[y0:y1, x0:x1]
    ring = np.zeros_like(mask)
    ring[y0:y1, x0:x1] = ndimage.binary_dilation(crop, structure=CONNECTED, iterations=3) & ~crop
    if not ring.any():
        raise ValueError("The defect mask leaves no neighboring pixels for contrast")
    components, _ = ndimage.label(crop, structure=CONNECTED)
    largest_region = int(np.bincount(components.ravel())[1:].max())

    scale = INPUT_LONG_EDGE / max(gray.size)
    size = tuple(max(1, round(length * scale)) for length in gray.size)
    resized = np.asarray(gray.resize(size, Image.Resampling.BILINEAR))
    mask_weights = np.asarray(Image.fromarray(mask.astype(np.float32)).resize(size, Image.Resampling.BOX))
    ring_weights = np.asarray(Image.fromarray(ring.astype(np.float32)).resize(size, Image.Resampling.BOX))
    foreground = native[mask]
    return {
        "native_mask_pixels": len(ys),
        "largest_native_region_pixels": largest_region,
        "tiny_only_in_capture": largest_region <= TINY_AREA_FRACTION * native.size,
        "resized_mask_area_pixels": float(mask_weights.sum()),
        "resized_majority_mask_pixels": int(np.count_nonzero(mask_weights >= 0.5)),
        "native_luma_contrast": float(abs(foreground.mean() - native[ring].mean()) / 255),
        "resized_luma_contrast": float(abs(np.average(resized, weights=mask_weights) - np.average(resized, weights=ring_weights)) / 255),
        "native_foreground_near_clipped": float(np.mean((foreground <= 5) | (foreground >= 250))),
    }


def alert_summary(scores: np.ndarray, regular: np.ndarray, threshold: float) -> dict:
    alerted = scores > threshold
    alerted_regular = regular > threshold
    return {
        "count": len(scores),
        "alerts": int(alerted.sum()),
        "median_score": float(np.median(scores)),
        "median_paired_score_change": float(np.median(scores - regular)),
        "new_alerts_vs_regular": int((alerted & ~alerted_regular).sum()),
        "cleared_alerts_vs_regular": int((~alerted & alerted_regular).sum()),
    }


def visibility_summary(rows: list[dict]) -> dict:
    summary = {"scenes": len(rows)}
    if rows:
        summary.update({
            "scenes_with_no_majority_mask_pixel_at_256": sum(row["resized_majority_mask_pixels"] == 0 for row in rows),
            "median_native_mask_pixels": float(np.median([row["native_mask_pixels"] for row in rows])),
            "median_resized_mask_area_pixels": float(np.median([row["resized_mask_area_pixels"] for row in rows])),
            "median_native_luma_contrast": float(np.median([row["native_luma_contrast"] for row in rows])),
            "median_resized_luma_contrast": float(np.median([row["resized_luma_contrast"] for row in rows])),
            "median_native_contrast_change_vs_regular": float(np.median([row["native_contrast_change_vs_regular"] for row in rows])),
            "median_native_foreground_near_clipped": float(np.median([row["native_foreground_near_clipped"] for row in rows])),
        })
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(dataset: Path, category: str, output: Path) -> None:
    captures = build_manifest(dataset, category)
    center, scale, monitor = calibration(dataset, category)
    good = {}
    defective = {}
    for capture in captures:
        scene, light, path = (capture[key] for key in ("scene", "lighting", "image"))
        with Image.open(dataset / path) as image:
            gray = image.convert("L")
        if capture["label"] == "good":
            features = quality_features(gray)
            score = float(np.max(np.abs(features - center) / scale))
            good[scene, light] = {
                "scene": scene, "lighting": light, "image": path,
                **dict(zip(FEATURES, features.tolist(), strict=True)),
                "score": score, "alert": score > monitor["scene_threshold"],
            }
        else:
            with Image.open(dataset / capture["mask"]) as mask:
                defective[scene, light] = {"scene": scene, "lighting": light, "image": path, **visibility(gray, mask)}

    for (scene, _), row in good.items():
        row["score_change_vs_regular"] = row["score"] - good[scene, "regular"]["score"]
    for (scene, _), row in defective.items():
        regular = defective[scene, "regular"]
        row["tiny_only_from_regular"] = regular["tiny_only_in_capture"]
        row["native_contrast_change_vs_regular"] = row["native_luma_contrast"] - regular["native_luma_contrast"]

    good_scenes = sorted({scene for scene, _ in good})
    batches = batch_indices(len(good_scenes))
    regular_scores = np.array([good[scene, "regular"]["score"] for scene in good_scenes])
    regular_batches = np.median(regular_scores[batches], axis=1)
    lights = sorted({light for _, light in good}, key=lambda light: (light != "regular", light))
    by_lighting = {}
    for light in lights:
        scores = np.array([good[scene, light]["score"] for scene in good_scenes])
        rows = [row for (_, condition), row in defective.items() if condition == light]
        by_lighting[light] = {
            "scene_alerts": alert_summary(scores, regular_scores, monitor["scene_threshold"]),
            "batch_alerts": alert_summary(np.median(scores[batches], axis=1), regular_batches, monitor["batch_threshold"]),
            "visibility": visibility_summary(rows),
            "tiny_only_visibility": visibility_summary([row for row in rows if row["tiny_only_from_regular"]]),
        }

    run_dir = output / category
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "good_scene_alerts.csv", list(good.values()))
    write_csv(run_dir / "defect_visibility.csv", list(defective.values()))
    report = {
        "category": category, "monitor": monitor, "by_lighting": by_lighting,
        "limitations": [
            "Regular-light alerts on known-good public scenes are counted as false alerts for the acquisition monitor.",
            "Shifted-light alerts indicate changes in image statistics. They do not estimate defect accuracy or set factory alarm rates.",
            "Batch results enumerate overlapping four-scene combinations. Those batches are not independent samples.",
            "Contrast is the absolute difference in mean decoded luma between the mask and its three-pixel outer ring, on a 0–1 scale.",
            "The resized audit uses the long edge of 256 pixels and projects mask and ring coverage by box resampling. It is a visibility proxy, not a model or human detection score.",
            "Each light uses its own annotation. Mask changes, texture, color, and glare can affect paired contrast independently of defect detection.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for light, results in by_lighting.items():
        scenes, batch = results["scene_alerts"], results["batch_alerts"]
        print(f"{category} {light}: good-scene alerts {scenes['alerts']}/{scenes['count']}, replay-batch alerts {batch['alerts']}/{batch['count']}")
    print(f"Saved {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="+", help="MVTec AD 2 categories to evaluate")
    parser.add_argument("--dataset", type=Path, default=Path("data/mvtec_ad_2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lighting_study"))
    args = parser.parse_args()
    try:
        for category in args.categories:
            run(args.dataset, category, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
