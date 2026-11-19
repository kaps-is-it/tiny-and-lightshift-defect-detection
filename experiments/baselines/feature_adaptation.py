"""Replay feature calibration from four verified-good paired Can scenes."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from local_contrast.evaluation import ImageResult, summarize
from experiments.baselines.multiscale_detector import MultiScaleDetector, _distance_map
from experiments.baselines.multiscale_experiment import add_fusion, predict_branches
from local_contrast.run_experiment import INPUT_SIZE, load_splits, read_image, training_batches


SUPPORT_SEED = 42
SUPPORT_SIZE = 4
SHRINKAGE = 0.5
GAIN_BOUNDS = (0.5, 2.0)


@torch.inference_mode()
def measure_moments(detector, paths, batch_size):
    sums = [0.0, 0.0]
    squares = [0.0, 0.0]
    counts = [0, 0]
    images = (read_image(path) for path in paths)
    for batch in training_batches(images, INPUT_SIZE, batch_size):
        for index, grid in enumerate(detector._feature_grids(batch)):
            sums[index] = sums[index] + grid.sum(dim=(0, 1, 2)).double()
            squares[index] = squares[index] + grid.square().sum(dim=(0, 1, 2)).double()
            counts[index] += grid.numel() // grid.shape[-1]
    moments = []
    for total, squared, count in zip(sums, squares, counts, strict=True):
        if not count:
            raise ValueError("Expected at least one support crop")
        center = total / count
        spread = (squared / count - center.square()).clamp_min(1e-6).sqrt()
        moments.append((center.float(), spread.float()))
    return moments


def correction(source, shifted):
    source_center, source_spread = source
    shifted_center, shifted_spread = shifted
    gain = (source_spread / shifted_spread).clamp(*GAIN_BOUNDS)
    return shifted_center, 1 + SHRINKAGE * (gain - 1), SHRINKAGE * (source_center - shifted_center)


class AdjustedDetector:
    """Keep the fitted normal banks and transport shifted features before matching."""

    def __init__(self, reference, adjustments):
        self.reference = reference
        self.adjustments = adjustments

    @torch.inference_mode()
    def predict(self, batch):
        banks = self.reference._references
        if banks is None:
            raise RuntimeError("Fit the source model before applying feature calibration")
        grids = self.reference._feature_grids(batch)
        maps = []
        for grid, bank, adjustment in zip(grids, banks, self.adjustments, strict=True):
            center, gain, offset = adjustment
            adjusted = (grid - center) * gain + center + offset
            maps.append(_distance_map(adjusted, bank))
        return tuple(maps)


def collect_support(captures):
    known_good = defaultdict(dict)
    for capture in captures:
        if capture.mask is None:
            known_good[capture.scene][capture.lighting] = capture.image
    if len(known_good) <= SUPPORT_SIZE:
        raise ValueError("At least one known-good scene must remain held out")
    selected = sorted(random.Random(SUPPORT_SEED).sample(sorted(known_good), SUPPORT_SIZE))
    lights = sorted({capture.lighting for capture in captures})
    for scene in selected:
        if set(known_good[scene]) != set(lights):
            raise ValueError(f"Missing paired reference view for {scene}")
    return selected, {light: [known_good[scene][light] for scene in selected] for light in lights}


def evaluate_light(reference, adjusted, captures, scales, threshold, original, root, batch_size):
    results = ([], [])
    max_baseline_gap = 0.0
    for capture in captures:
        image = read_image(capture.image)
        mask = None
        if capture.mask:
            with Image.open(capture.mask) as label:
                mask = np.asarray(label) != 0
        relative = capture.image.relative_to(root).as_posix()
        for index, detector in enumerate((reference, adjusted)):
            maps = predict_branches(detector, image, batch_size)
            anomaly = add_fusion(maps, scales)["fusion"]
            score = float(anomaly.max())
            if index == 0:
                max_baseline_gap = max(max_baseline_gap, abs(score - original[relative]))
            results[index].append(ImageResult(relative, capture.scene, capture.lighting,
                                              mask is not None, score, anomaly, mask))
    return {
        "baseline": summarize(results[0], threshold),
        "feature_calibrated": summarize(results[1], threshold),
        "baseline_score_reproduction_max_abs_error": max_baseline_gap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/mvtec_ad_2"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/multiscale/native/can"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_adaptation_can.json"))
    args = parser.parse_args()
    metadata = json.loads((args.reference / "metrics.json").read_text(encoding="utf-8"))
    device = "cpu" if metadata["device"] == "CPU" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("Recreate the native Can reference with --device cpu to replay it on this machine")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if metadata["category"] != "can" or metadata["treatment"] != "native" or metadata.get("backbone", "resnet") != "resnet":
        parser.error("Expected the run report for native Can ResNet")
    _, _, captures = load_splits(args.dataset, "can", metadata["seed"])
    support_scenes, support = collect_support(captures)
    source = [args.dataset / path for path in metadata["training_images"]]
    model = MultiScaleDetector(device, metadata["context_bank"], metadata["fine_bank"], metadata["seed"])
    batch_size = metadata["batch_size"]
    model.fit(training_batches((read_image(path) for path in source), INPUT_SIZE, batch_size))
    regular = measure_moments(model, support["regular"], batch_size)
    with (args.reference / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        original = {row["image"]: float(row["fusion"]) for row in csv.DictReader(stream)}

    by_light = {}
    scales = metadata["fusion_scales"]
    threshold = metadata["score_thresholds"]["fusion"]
    for light, support_paths in support.items():
        if light == "regular":
            continue
        shifted = measure_moments(model, support_paths, batch_size)
        adjusted = AdjustedDetector(model, [correction(ref, target) for ref, target in zip(regular, shifted, strict=True)])
        held_out = [capture for capture in captures if capture.lighting == light and capture.scene not in support_scenes]
        row = evaluate_light(model, adjusted, held_out, scales, threshold, original, args.dataset, batch_size)
        if row["baseline_score_reproduction_max_abs_error"] > 1e-4:
            raise ValueError("The refitted baseline did not reproduce its recorded Can scores")
        by_light[light] = row
        print(light, " | ".join(
            f"{name} {row[name]['au_pro_05']:.1%} AU-PRO, {row[name]['fn']} FN, {row[name]['fp']} FP"
            for name in ("baseline", "feature_calibrated")
        ), flush=True)

    report = {
        "category": "can", "source_training_images": len(source), "device": device,
        "support_seed": SUPPORT_SEED, "support_scenes": support_scenes,
        "gain_bounds_before_shrinkage": GAIN_BOUNDS, "shrinkage": SHRINKAGE,
        "fixed_part_threshold": threshold, "fusion_scales": scales,
        "setting": "Simulated commissioning with verified same-light normal support and matched regular captures",
        "by_lighting": by_light,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
