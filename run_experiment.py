"""Fit a normal-image reference and evaluate defect localization."""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

from build_scene_manifest import build_manifest
from detector import PatchDetector
from evaluation import ImageResult, save_overlay, summarize, summarize_decisions
from part_scoring import BACKGROUND_WINDOW, local_contrast_score


INPUT_SIZE = 256
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class Capture:
    image: Path
    scene: str
    lighting: str = "regular"
    mask: Path | None = None


def load_splits(root: Path, category: str, seed: int):
    directory = root / category
    training = sorted((directory / "train" / "good").glob("*.png"))

    if category == "bottle":
        if len(training) < 5:
            raise ValueError(f"Missing good training images under {directory}")
        held_out = set(random.Random(seed).sample(training, round(0.2 * len(training))))
        validation = sorted(held_out)
        training = [path for path in training if path not in held_out]
        captures = []
        for image in sorted((directory / "test").glob("*/*.png")):
            label = image.parent.name
            mask = directory / "ground_truth" / label / f"{image.stem}_mask.png"
            if label != "good" and not mask.is_file():
                raise ValueError(f"Missing ground-truth mask: {mask}")
            captures.append(Capture(image, f"bottle/{label}/{image.stem}", mask=mask if label != "good" else None))
    else:
        validation = sorted((directory / "validation" / "good").glob("*.png"))
        captures = [
            Capture(root / row["image"], row["scene"], row["lighting"], root / row["mask"] if row["mask"] else None)
            for row in build_manifest(root, category)
        ]

    train_and_validation = training + validation
    if not training or not validation or not any(item.mask for item in captures) or not any(item.mask is None for item in captures):
        raise ValueError(f"Incomplete training, validation, or evaluation split under {directory}")
    if category != "bottle" and any(not path.stem.endswith("_regular") for path in train_and_validation):
        raise ValueError("Expected regular-light training and validation captures")
    return training, validation, captures


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def fit_frame(image: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ratio = INPUT_SIZE / max(image.size)
    width, height = (max(1, round(size * ratio)) for size in image.size)
    left = (INPUT_SIZE - width) // 2
    top = (INPUT_SIZE - height) // 2
    right = left + width
    bottom = top + height
    pixels = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    padding = ((top, INPUT_SIZE - bottom), (left, INPUT_SIZE - right), (0, 0))
    # Extend existing pixels around the frame to avoid an artificial black edge.
    return np.pad(pixels, padding, mode="reflect"), (left, top, right, bottom)


def tile_origins(length: int, tile_size: int) -> list[int]:
    last_start = max(0, length - tile_size)
    step = max(1, round(tile_size * 0.75))
    if last_start == 0:
        return [0]
    # Space the windows evenly so the last two tiles do not nearly coincide.
    steps = (last_start + step - 1) // step
    return [round(index * last_start / steps) for index in range(steps + 1)]


def tiles(image: Image.Image, tile_size: int):
    for y in tile_origins(image.height, tile_size):
        for x in tile_origins(image.width, tile_size):
            end_x, end_y = min(x + tile_size, image.width), min(y + tile_size, image.height)
            pixels, frame = fit_frame(image.crop((x, y, end_x, end_y)))
            yield pixels, frame, (x, y, end_x, end_y)


def normalize(pixels: list[np.ndarray]) -> torch.Tensor:
    batch = np.stack(pixels).astype(np.float32) / 255.0
    batch = (batch - MEAN) / STD
    return torch.from_numpy(batch).permute(0, 3, 1, 2)


def training_views(paths: list[Path], augment_lighting: bool, seed: int):
    rng = random.Random(seed)
    for path in paths:
        image = read_image(path)
        yield image
        if augment_lighting:
            brighter = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
            yield ImageEnhance.Contrast(brighter).enhance(rng.uniform(0.85, 1.15))


def training_batches(images, tile_size: int | None, batch_size: int):
    batch = []
    for image in images:
        crops = (pixels for pixels, _, _ in tiles(image, tile_size)) if tile_size else (fit_frame(image)[0],)
        for pixels in crops:
            batch.append(pixels)
            if len(batch) == batch_size:
                yield normalize(batch)
                batch.clear()
    if batch:
        yield normalize(batch)


def restore_map(scores: np.ndarray, frame: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    # Crop the padding after upsampling to model-input coordinates.
    grid = Image.fromarray(scores).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(grid.crop(frame).resize(size, Image.Resampling.BILINEAR))


def predict_map(detector: PatchDetector, image: Image.Image, tile_size: int | None, batch_size: int) -> np.ndarray:
    if tile_size is None:
        pixels, frame = fit_frame(image)
        scores = detector.predict(normalize([pixels]))[0].numpy()
        anomaly_map = restore_map(scores, frame, image.size)
    else:
        crops = list(tiles(image, tile_size))
        weighted_scores = np.zeros((image.height, image.width), dtype=np.float32)
        weights = np.zeros_like(weighted_scores)
        taper = np.maximum(np.hanning(tile_size), 0.1).astype(np.float32)
        window = taper[:, None] * taper[None, :]

        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            maps = detector.predict(normalize([pixels for pixels, _, _ in batch])).numpy()
            for scores, (_, frame, (x, y, end_x, end_y)) in zip(maps, batch, strict=True):
                width, height = end_x - x, end_y - y
                tile_weights = window[:height, :width]
                weighted_scores[y:end_y, x:end_x] += tile_weights * restore_map(scores, frame, (width, height))
                weights[y:end_y, x:end_x] += tile_weights
        anomaly_map = weighted_scores / weights

    # Use the same small smoothing radius in native coordinates for both modes.
    return gaussian_filter(anomaly_map, sigma=2.0)


def write_predictions(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    training, validation, captures = load_splits(args.dataset, args.category, args.seed)
    detector = PatchDetector(device=device, bank_size=args.bank_size, seed=args.seed)
    print(f"Fitting {args.category} {args.mode} on {len(training)} good images ({device})", flush=True)
    started = time.perf_counter()
    tile_size = args.tile_size if args.mode == "tiled" else None
    views = training_views(training, args.augment_lighting, args.seed)
    detector.fit(training_batches(views, tile_size, args.batch_size))
    fit_time = time.perf_counter() - started
    print(f"Reference bank ready in {fit_time:.1f}s; scoring {len(validation)} good validation images", flush=True)
    validation_records = []
    for path in validation:
        anomaly_map = predict_map(detector, read_image(path), tile_size, args.batch_size)
        record = {"image": path.relative_to(args.dataset).as_posix(), "score": float(anomaly_map.max())}
        if args.compare_part_scores:
            record["local_contrast_score"] = local_contrast_score(anomaly_map)
        validation_records.append(record)
    threshold = float(np.quantile([record["score"] for record in validation_records], 0.90))
    local_threshold = None
    if args.compare_part_scores:
        normal_scores = [record["local_contrast_score"] for record in validation_records]
        local_threshold = float(np.quantile(normal_scores, 0.90))

    variant = f"{args.mode}-augmented" if args.augment_lighting else args.mode
    if args.compare_part_scores:
        variant += "-paired"
    output = args.output / args.category / variant
    output.mkdir(parents=True, exist_ok=True)
    by_light = defaultdict(list)
    for capture in captures:
        by_light[capture.lighting].append(capture)
    records = []
    metrics = {}
    local_metrics = {}
    for lighting in sorted(by_light, key=lambda name: (name != "regular", name)):
        results = []
        local_results = []
        times = []
        local_times = []
        for capture in by_light[lighting]:
            image = read_image(capture.image)
            started = time.perf_counter()
            anomaly_map = predict_map(detector, image, tile_size, args.batch_size)
            elapsed = (time.perf_counter() - started) * 1000
            times.append(elapsed)
            mask = None
            if capture.mask:
                with Image.open(capture.mask) as truth:
                    mask = np.asarray(truth) != 0
                if mask.shape != anomaly_map.shape:
                    raise ValueError(f"Mask image size mismatch: {capture.mask}")
            result = ImageResult(
                capture.image.relative_to(args.dataset).as_posix(), capture.scene, lighting,
                capture.mask is not None, float(anomaly_map.max()), anomaly_map, mask,
            )
            results.append(result)
            record = {
                "image": result.image, "scene": result.scene, "lighting": result.lighting,
                "is_defective": result.is_defective, "score": result.score,
                "rejected": result.score >= threshold, "inference_ms": elapsed,
            }
            if local_threshold is not None:
                scoring_start = time.perf_counter()
                local_score = local_contrast_score(anomaly_map)
                local_elapsed = (time.perf_counter() - scoring_start) * 1000
                local_times.append(local_elapsed)
                local_results.append(replace(result, score=local_score))
                record.update({
                    "local_contrast_score": local_score,
                    "local_contrast_rejected": local_score >= local_threshold,
                    "local_contrast_ms": local_elapsed,
                })
            records.append(record)

        summary = summarize(results, threshold)
        summary["inference_ms_median"] = float(np.median(times))
        summary["inference_ms_p95"] = float(np.quantile(times, 0.95))
        metrics[lighting] = summary
        local_summary = None
        if local_threshold is not None:
            local_summary = summarize_decisions(local_results, local_threshold)
            local_summary["scoring_ms_median"] = float(np.median(local_times))
            local_metrics[lighting] = local_summary
        for label, defective in (("good", False), ("defect", True)):
            examples = local_results if local_threshold is not None else results
            preview = next((result for result in examples if result.is_defective == defective), None)
            if preview:
                decision = ""
                if local_threshold is not None:
                    verdict = "reject" if preview.score >= local_threshold else "pass"
                    decision = f"local contrast {verdict}"
                save_overlay(args.dataset / preview.image, preview, output / f"{lighting}_{label}.png", (0.0, threshold), decision=decision)
        message = f"{lighting}: {summary['fn']} missed, {summary['fp']} false rejects, {summary['inference_ms_median']:.0f} ms/image"
        if local_summary is not None:
            message += f" | local contrast: {local_summary['fn']} missed, {local_summary['fp']} false rejects"
        print(message, flush=True)

    write_predictions(output / "predictions.csv", records)
    write_predictions(output / "validation_scores.csv", validation_records)
    report = {
        "category": args.category, "mode": args.mode, "seed": args.seed,
        "device": torch.cuda.get_device_name(device) if device == "cuda" else "CPU",
        "input_size": INPUT_SIZE, "tile_size": tile_size, "bank_size": args.bank_size,
        "batch_size": args.batch_size, "train_count": len(training), "validation_count": len(validation),
        "lighting_augmentation": args.augment_lighting,
        "fit_seconds": fit_time, "score_threshold": threshold, "by_lighting": metrics,
        "training_images": [path.relative_to(args.dataset).as_posix() for path in training],
    }
    if local_threshold is not None:
        report.update({
            "local_contrast_threshold": local_threshold,
            "local_contrast_window": BACKGROUND_WINDOW,
            "local_contrast_top_fraction": 0.0001,
            "local_contrast_by_lighting": local_metrics,
        })
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=("bottle", "can", "fabric", "sheet_metal", "wallplugs"))
    parser.add_argument("--dataset", type=Path, help="Override the extracted dataset location")
    parser.add_argument("--mode", choices=("full", "tiled"), default="full")
    parser.add_argument("--augment-lighting", action="store_true", help="Add one brightness/contrast variant per good training image")
    parser.add_argument("--compare-part-scores", action="store_true", help="Compare raw maximum and local contrast on the same tiled heatmap")
    parser.add_argument("--tile-size", type=int, default=256, help="Native-pixel crop size for tiled mode")
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if min(args.tile_size, args.bank_size, args.batch_size) <= 0:
        parser.error("Tile size, bank size, and batch size must be positive")
    if args.compare_part_scores and args.mode != "tiled":
        parser.error("The local-contrast score comparison requires tiled mode")
    if args.dataset is None:
        args.dataset = Path("MVTECH_ANAMOLY_DATASET" if args.category == "bottle" else "mvtec_ad_2_dataset")
    run(args)


if __name__ == "__main__":
    main()
