"""Compare four part scores on the same tiled PatchCore heatmaps."""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from local_contrast.detector import PatchDetector
from local_contrast.evaluation import ImageResult, summarize, summarize_decisions
from local_contrast.part_scoring import BACKGROUND_WINDOW, local_contrast
from local_contrast.run_experiment import (
    load_splits,
    predict_map,
    read_image,
    training_batches,
    training_views,
    write_predictions,
)
from local_contrast.spatial_scoring import SpatialScorer


TILE_SIZE = 256


def score_maps(anomaly_map: np.ndarray, spatial: SpatialScorer) -> dict[str, tuple[float, np.ndarray]]:
    standardized = spatial.transform(anomaly_map)
    contrast = local_contrast(anomaly_map)
    return {
        "raw_max": (float(anomaly_map.max()), anomaly_map),
        "raw_tail": (SpatialScorer.score(anomaly_map), anomaly_map),
        "spatial_tail": (SpatialScorer.score(standardized), standardized),
        "local_tail": (SpatialScorer.score(contrast), contrast),
    }


def run(args: argparse.Namespace) -> None:
    training, validation, captures = load_splits(args.dataset, args.category, args.seed)
    if len(validation) < 6:
        raise ValueError("Need at least six good validation images to split spatial fitting and calibration")
    rng = random.Random(args.seed)
    reference = set(rng.sample(validation, len(validation) // 2))
    calibration = [path for path in validation if path not in reference]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    detector = PatchDetector(device, bank_size=args.bank_size, seed=args.seed)

    def heatmap(path: Path) -> np.ndarray:
        return predict_map(detector, read_image(path), TILE_SIZE, args.batch_size)

    print(f"Fitting on {len(training)} good {args.category} images ({device})", flush=True)
    detector.fit(training_batches(training_views(training, False, args.seed), TILE_SIZE, args.batch_size))
    print(f"Fitting spatial statistics on {len(reference)} good maps", flush=True)
    spatial = SpatialScorer().fit(heatmap(path) for path in sorted(reference))

    # Calibrate every score on the same good images, separate from the spatial fit.
    normal_scores = defaultdict(list)
    for path in calibration:
        for method, (score, _) in score_maps(heatmap(path), spatial).items():
            normal_scores[method].append(score)
    thresholds = {method: float(np.quantile(scores, 0.90)) for method, scores in normal_scores.items()}

    groups = defaultdict(list)
    for capture in captures:
        groups[capture.lighting].append(capture)

    summaries = {}
    records = []
    for light, group in groups.items():
        results = defaultdict(list)
        for capture in group:
            mapping = score_maps(heatmap(capture.image), spatial)
            mask = None
            if capture.mask:
                with Image.open(capture.mask) as truth:
                    mask = np.asarray(truth) != 0
            image = capture.image.relative_to(args.dataset).as_posix()
            raw_score, raw_map = mapping["raw_max"]
            raw_result = ImageResult(image, capture.scene, light, mask is not None, raw_score, raw_map, mask)
            for method, (score, anomaly_map) in mapping.items():
                results[method].append(replace(raw_result, score=score, anomaly_map=anomaly_map))
                records.append({
                    "image": image, "scene": capture.scene, "lighting": light,
                    "is_defective": mask is not None, "method": method,
                    "score": score, "rejected": score >= thresholds[method],
                })
        summary = {
            method: summarize(results[method], thresholds[method])
            for method in ("raw_max", "spatial_tail", "local_tail")
        }
        # Raw max and raw tail use identical heatmaps and localization metrics.
        summary["raw_tail"] = {**summary["raw_max"], **summarize_decisions(results["raw_tail"], thresholds["raw_tail"])}
        summaries[light] = summary
        counts = ", ".join(f"{method}: {values['fn']} misses / {values['fp']} false rejects" for method, values in summary.items())
        print(f"{light}: {counts}", flush=True)

    output = args.output / args.category
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(output / "predictions.csv", records)
    report = {
        "category": args.category, "seed": args.seed, "device": device,
        "bank_size": args.bank_size, "tile_size": TILE_SIZE, "batch_size": args.batch_size,
        "background_window": BACKGROUND_WINDOW, "train_count": len(training), "thresholds": thresholds,
        "reference_images": [path.relative_to(args.dataset).as_posix() for path in sorted(reference)],
        "calibration_images": [path.relative_to(args.dataset).as_posix() for path in calibration],
        "by_lighting": summaries,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved score comparison to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=("can", "sheet_metal"))
    parser.add_argument("--dataset", type=Path, default=Path("data/mvtec_ad_2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/score-comparison"))
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.bank_size, args.batch_size) <= 0:
        parser.error("Bank size and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
