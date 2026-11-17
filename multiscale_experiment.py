"""Compare fine and contextual patch features on native-resolution MVTec AD 2 scenes."""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from evaluation import ImageResult, save_overlay, summarize, summarize_decisions
from multiscale_detector import MultiScaleDetector
from part_scoring import local_contrast_score
from run_experiment import (INPUT_SIZE, fit_frame, load_splits, normalize, read_image, restore_map,
                            tile_origins, training_batches, write_predictions)


CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
MAP_VARIANTS = ("context", "fine", "fusion")
DECISION_VARIANTS = ("context_local", "fusion_local")
VARIANTS = MAP_VARIANTS + DECISION_VARIANTS


def native_tiles(image):
    for y in tile_origins(image.height, INPUT_SIZE):
        for x in tile_origins(image.width, INPUT_SIZE):
            end_x, end_y = min(x + INPUT_SIZE, image.width), min(y + INPUT_SIZE, image.height)
            crop = image.crop((x, y, end_x, end_y))
            if crop.size == (INPUT_SIZE, INPUT_SIZE):
                pixels, frame = np.asarray(crop), (0, 0, INPUT_SIZE, INPUT_SIZE)
            else:
                pixels, frame = fit_frame(crop)
            yield pixels, frame, (x, y, end_x, end_y)


@lru_cache(maxsize=2)
def stitch_weights(width, height):
    taper = np.maximum(np.hanning(INPUT_SIZE), 0.1).astype(np.float32)
    window = taper[:, None] * taper[None, :]
    weights = np.zeros((height, width), dtype=np.float32)
    for y in tile_origins(height, INPUT_SIZE):
        for x in tile_origins(width, INPUT_SIZE):
            region = weights[y:y + INPUT_SIZE, x:x + INPUT_SIZE]
            region += window[:region.shape[0], :region.shape[1]]
    return window, weights


def predict_branches(detector, image, batch_size, *, full_frame=False):
    """Return native-coordinate context and fine maps using a shared forward pass."""
    if full_frame:
        pixels, frame = fit_frame(image)
        fine, context = detector.predict(normalize([pixels]))
        return {
            name: gaussian_filter(restore_map(grid[0].numpy(), frame, image.size), sigma=2.0)
            for name, grid in (("context", context), ("fine", fine))
        }

    window, weights = stitch_weights(*image.size)
    merged = {name: np.zeros_like(weights) for name in ("context", "fine")}
    crops = native_tiles(image)
    while batch := list(islice(crops, batch_size)):
        fine, context = detector.predict(normalize([pixels for pixels, _, _ in batch]))
        grids = {"context": context.numpy(), "fine": fine.numpy()}
        for index, (_, frame, (x, y, end_x, end_y)) in enumerate(batch):
            width, height = end_x - x, end_y - y
            tile_weights = window[:height, :width]
            for name, scores in merged.items():
                scores[y:end_y, x:end_x] += tile_weights * restore_map(grids[name][index], frame, (width, height))
    for name, scores in merged.items():
        scores /= weights
        gaussian_filter(scores, sigma=2.0, output=scores)
    return merged


def source_views(paths, treatment, seed):
    for index, path in enumerate(paths):
        image = read_image(path)
        if treatment in ("luma", "retinex"):
            from photometric_views import transform_image
            image = transform_image(image, treatment)
        yield image
        if treatment == "augmented":
            from photometric_views import domain_randomized
            yield domain_randomized(image, seed + index)


def infer(detector, image, args):
    if args.treatment in ("luma", "retinex"):
        from photometric_views import transform_image
        image = transform_image(image, args.treatment)
    return predict_branches(detector, image, args.batch_size, full_frame=args.treatment == "full")


def add_fusion(maps, scales):
    maps["fusion"] = 0.5 * (maps["context"] / scales["context"] + maps["fine"] / scales["fine"])
    return maps


def score_variants(maps):
    scores = {name: float(maps[name].max()) for name in MAP_VARIANTS}
    scores.update({name: local_contrast_score(maps[name.removesuffix("_local")]) for name in DECISION_VARIANTS})
    return scores


def build_reference(training, args, device):
    if args.train_limit and len(training) > args.train_limit:
        indices = np.linspace(0, len(training) - 1, args.train_limit, dtype=int)
        training = [training[index] for index in indices]
    options = dict(device=device, bank_size=args.bank_size, fine_bank_size=args.fine_bank_size, seed=args.seed)
    if args.backbone == "convnext":
        from convnext_detector import ConvNeXtDetector
        detector = ConvNeXtDetector(**options, weights=args.weights)
    elif args.backbone == "gradient":
        from gradient_detector import GradientDetector
        detector = GradientDetector(**options)
    elif args.backbone == "synthetic":
        from synthetic_detector import SyntheticDetector
        detector = SyntheticDetector(**options)
    elif args.backbone == "distillation":
        from distillation_detector import DistillationDetector
        detector = DistillationDetector(**options)
    else:
        detector = MultiScaleDetector(**options)
    detector.coreset = args.treatment == "coreset"
    started = time.perf_counter()
    views = source_views(training, args.treatment, args.seed)
    detector.fit(training_batches(views, None if args.treatment == "full" else INPUT_SIZE, args.batch_size))
    return detector, training, time.perf_counter() - started


def calibrate(detector, validation, args):
    # Fit fusion scales and fixed thresholds on the same normal-only validation split.
    if args.validation_limit and len(validation) > args.validation_limit:
        indices = np.linspace(0, len(validation) - 1, args.validation_limit, dtype=int)
        validation = [validation[index] for index in indices]
    maps_by_file = []
    for path in validation:
        maps = infer(detector, read_image(path), args)
        maps_by_file.append((path, maps))
    scales = {
        name: max(1e-8, float(np.median([maps[name].max() for _, maps in maps_by_file])))
        for name in ("context", "fine")
    }
    records = []
    while maps_by_file:
        path, maps = maps_by_file.pop()
        records.append({"image": path.relative_to(args.dataset).as_posix(), **score_variants(add_fusion(maps, scales))})
    records.reverse()
    thresholds = {name: float(np.quantile([row[name] for row in records], 0.90)) for name in VARIANTS}
    return scales, thresholds, records


def pixel_operating_point(results, threshold):
    predicted = positives = overlap = 0
    for result in results:
        selected = result.anomaly_map >= threshold
        predicted += int(np.count_nonzero(selected))
        if result.mask is not None:
            positives += int(np.count_nonzero(result.mask))
            overlap += int(np.count_nonzero(selected[result.mask]))
    return {
        "pixel_tp": overlap, "pixel_predicted": predicted, "pixel_positive": positives,
        "pixel_f1": 2 * overlap / (positives + predicted) if positives + predicted else None,
    }


def evaluate_group(detector, captures, scales, thresholds, args, output):
    results = defaultdict(list)
    records = []
    times = []
    for capture in captures:
        image = read_image(capture.image)
        started = time.perf_counter()
        maps = add_fusion(infer(detector, image, args), scales)
        times.append((time.perf_counter() - started) * 1000)
        scores = score_variants(maps)
        mask = None
        if capture.mask:
            with Image.open(capture.mask) as stream:
                mask = np.asarray(stream) != 0
        if mask is not None and mask.shape != (image.height, image.width):
            raise ValueError(f"Mask dimensions differ from the capture: {capture.image}")
        path = capture.image.relative_to(args.dataset).as_posix()
        record = {"image": path, "scene": capture.scene, "lighting": capture.lighting,
                  "is_defective": mask is not None, "paired_map_ms": times[-1]}
        for name in MAP_VARIANTS:
            results[name].append(ImageResult(path, capture.scene, capture.lighting, mask is not None,
                                             scores[name], maps[name], mask))
        for name in VARIANTS:
            record[name] = scores[name]
            record[f"{name}_rejected"] = scores[name] >= thresholds[name]
        records.append(record)

    metrics = {
        name: {**summarize(results[name], thresholds[name]),
               **pixel_operating_point(results[name], thresholds[name])}
        for name in MAP_VARIANTS
    }
    for name in DECISION_VARIANTS:
        raw_maps = results[name.removesuffix("_local")]
        local_scores = [replace(result, score=record[name]) for result, record in zip(raw_maps, records, strict=True)]
        metrics[name] = summarize_decisions(local_scores, thresholds[name])
    if args.previews:
        lighting = captures[0].lighting
        for name in MAP_VARIANTS:
            example = next(result for result in results[name] if result.is_defective)
            score_scale = (0.0, thresholds[name])
            save_overlay(args.dataset / example.image, example, output / f"{lighting}-{name}.png", score_scale)
    return metrics, records, {"median": float(np.median(times)), "p95": float(np.quantile(times, 0.95))}


def run_category(category, args, device):
    training, validation, captures = load_splits(args.dataset, category, args.seed)
    run_name = args.treatment if args.backbone == "resnet" else f"{args.backbone}-{args.treatment}"
    output = args.output / run_name / category
    output.mkdir(parents=True, exist_ok=True)
    detector, training, fit_seconds = build_reference(training, args, device)
    print(f"{category} {run_name} fitted {len(training)} source images in {fit_seconds:.1f}s", flush=True)
    scales, thresholds, validation_records = calibrate(detector, validation, args)
    by_light = defaultdict(list)
    for capture in captures:
        by_light[capture.lighting].append(capture)
    all_records = []
    all_metrics = {}
    for light in sorted(by_light, key=lambda name: (name != "regular", name)):
        metrics, rows, timing = evaluate_group(detector, by_light[light], scales, thresholds, args, output)
        all_records.extend(rows)
        all_metrics[light] = {"variants": metrics, "paired_map_ms": timing}
        pro = " ".join(f"{name} {100 * metrics[name]['au_pro_05']:.1f}%" for name in MAP_VARIANTS)
        print(f"{category} {run_name} {light} | {pro} | {timing['median']:.0f} ms", flush=True)

    write_predictions(output / "predictions.csv", all_records)
    write_predictions(output / "validation_scores.csv", validation_records)
    report = {
        "category": category, "treatment": args.treatment, "backbone": args.backbone, "seed": args.seed,
        "device": torch.cuda.get_device_name(device) if device == "cuda" else "CPU",
        "torch": torch.__version__, "input_size": INPUT_SIZE, "batch_size": args.batch_size,
        "context_bank": args.bank_size, "fine_bank": args.fine_bank_size, "fit_seconds": fit_seconds,
        "training_images": [path.relative_to(args.dataset).as_posix() for path in training],
        "validation_count": len(validation_records), "fusion_scales": scales,
        "score_thresholds": thresholds, "by_lighting": all_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def add_model_args(parser):
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--treatment", choices=("native", "full", "augmented", "luma", "retinex", "coreset"), default="native")
    parser.add_argument("--backbone", choices=("resnet", "convnext", "gradient", "synthetic", "distillation"), default="resnet")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/backbones/convnext_tiny_dinov3.safetensors"))
    parser.add_argument("--train-limit", type=int, default=96, help="Evenly spaced source images, or 0 for all")
    parser.add_argument("--validation-limit", type=int, default=0, help="Evenly spaced validation images, or 0 for all")
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--fine-bank-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)


def configure(parser, args):
    if min(args.bank_size, args.fine_bank_size, args.batch_size) < 1 or min(args.train_limit, args.validation_limit) < 0:
        parser.error("Use positive bank and batch sizes and nonnegative split limits")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", choices=CATEGORIES, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "multiscale")
    parser.add_argument("--previews", action="store_true")
    add_model_args(parser)
    args = parser.parse_args()
    device = configure(parser, args)
    for category in dict.fromkeys(args.categories):
        run_category(category, args, device)


if __name__ == "__main__":
    main()
