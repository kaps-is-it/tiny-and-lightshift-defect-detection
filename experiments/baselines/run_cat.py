"""Evaluate the released VAND 2026 CAT checkpoint on the public Can scenes."""

import argparse
from collections import defaultdict
from functools import partial
from hashlib import file_digest
import json
import math
from pathlib import Path
from runpy import run_path
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F

from local_contrast.evaluation import ImageResult, summarize
from local_contrast.run_experiment import load_splits, read_image, write_predictions


REVISION = "9038ee3dddf947f30c61251c6fce898f2334ba6c"
UPSTREAM_ROOT = Path("artifacts/upstream-cat")
DEFAULT_SOURCE = UPSTREAM_ROOT / f"VAND2026_CAT-{REVISION}"
DEFAULT_WEIGHTS = UPSTREAM_ROOT / "checkpoints/can-model.pth"
GRID_SIZE = 4
INPUT_SIZE = 448
CHECKPOINT_SHA256 = "f2a9b45d4dc65779cc4dc15e2972de17fdb268982f6e389e7bc942df8894493e"
RELEASED_MASK_THRESHOLD = 42
RELEASED_MIN_AREA_PIXELS = 8
RELEASED_MIN_AREA_RATIO = 0.00002


def load_cat(source: Path, weights: Path, device: torch.device):
    """Use the upstream modules and every encoder/decoder tensor from the author's checkpoint."""
    if not (source / "models/uad.py").is_file() or not (source / "dinov3/dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"Expected the pinned CAT source tree under {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Expected a released Can checkpoint at {weights}")
    sys.path[:0] = [str(source.resolve()), str((source / "dinov3").resolve())]
    from dinov3.hub.backbones import dinov3_vitl16
    from mvtec2_dataset import get_data_transforms

    # Load these independent modules without initializing unrelated upstream backbones.
    INP_Former = run_path(str(source / "models/uad.py"))["INP_Former"]
    transformer = run_path(str(source / "models/vision_transformer.py"))
    Aggregation_Block, Mlp, Prototype_Block = (
        transformer[name] for name in ("Aggregation_Block", "Mlp", "Prototype_Block")
    )

    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not any(name.startswith("encoder.") for name in state):
        raise ValueError("The CAT checkpoint does not contain the DINOv3 encoder")
    # The checked-in CAT inference configuration uses ViT-L and six prototypes.
    dimension, heads = 1024, 16
    norm_layer = partial(nn.LayerNorm, eps=1e-8)
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    with torch.device("meta"):
        encoder = dinov3_vitl16(pretrained=False)
        encoder.is_dinov3 = True
        encoder.num_register_tokens = encoder.n_storage_tokens
        model = INP_Former(
            encoder=encoder,
            bottleneck=nn.ModuleList([Mlp(dimension, dimension * 4, dimension, drop=0.0)]),
            aggregation=nn.ModuleList([
                Aggregation_Block(dim=dimension, num_heads=heads, mlp_ratio=4.0,
                                  qkv_bias=True, norm_layer=norm_layer),
            ]),
            decoder=nn.ModuleList([
                Prototype_Block(dim=dimension, num_heads=heads, mlp_ratio=4.0,
                                qkv_bias=True, norm_layer=norm_layer)
                for _ in range(8)
            ]),
            target_layers=[4, 6, 8, 10, 12, 14, 16, 18],
            fuse_layer_encoder=groups,
            fuse_layer_decoder=groups,
            remove_class_token=True,
            prototype_token=nn.ParameterList([nn.Parameter(torch.empty(6, dimension))]),
        )
    model.load_state_dict(state, strict=True, assign=True)
    transform, _ = get_data_transforms(INPUT_SIZE, 392)
    return model.eval().requires_grad_(False).to(device), transform


def tile_boxes(width: int, height: int):
    """Match the upstream four-by-four split, including edge remainders."""
    tile_width, tile_height = width // GRID_SIZE, height // GRID_SIZE
    for row in range(GRID_SIZE):
        top = row * tile_height
        bottom = height if row == GRID_SIZE - 1 else top + tile_height
        for col in range(GRID_SIZE):
            left = col * tile_width
            right = width if col == GRID_SIZE - 1 else left + tile_width
            yield left, top, right, bottom


def gaussian_kernel(device: torch.device) -> torch.Tensor:
    xy = torch.arange(5, dtype=torch.float32) - 2
    variance = 4.0**2
    kernel = (1.0 / (2 * math.pi * variance)) * torch.exp(
        -(xy.square()[:, None] + xy.square()[None, :]) / (2 * variance)
    )
    return (kernel / kernel.sum()).reshape(1, 1, 5, 5).to(device)


def tile_anomaly_map(encoder, decoder, size: tuple[int, int], kernel: torch.Tensor) -> torch.Tensor:
    # Same cosine residual, bilinear alignment and blur as the upstream inference.
    maps = [
        F.interpolate(
            (1 - F.cosine_similarity(reference, reconstructed, dim=1))[:, None],
            size=size, mode="bilinear", align_corners=True,
        )
        for reference, reconstructed in zip(encoder, decoder, strict=True)
    ]
    residual = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
    return F.conv2d(residual, kernel, padding=2).clamp(0, 1)


@torch.inference_mode()
def predict_map(model, transform, kernel: torch.Tensor, image: Image.Image,
                device: torch.device, batch_size: int) -> np.ndarray:
    boxes = list(tile_boxes(*image.size))
    reconstructed = np.zeros((image.height, image.width), dtype=np.uint8)
    for start in range(0, len(boxes), batch_size):
        group = boxes[start:start + batch_size]
        tiles = torch.stack([transform(image.crop(box)) for box in group]).to(device)
        encoded, decoded, *_ = model(tiles)
        for index, (left, top, right, bottom) in enumerate(group):
            score = tile_anomaly_map(
                [level[index:index + 1] for level in encoded],
                [level[index:index + 1] for level in decoded],
                (bottom - top, right - left), kernel,
            )
            # CAT writes each tile as uint8 TIFF before assembling the full image.
            reconstructed[top:bottom, left:right] = (score[0, 0].cpu().numpy() * 255).astype(np.uint8)
    return reconstructed


def released_mask(scores: np.ndarray) -> np.ndarray:
    """Apply the author's Can private-split mask rule to an assembled uint8 map."""
    minimum_area = max(RELEASED_MIN_AREA_PIXELS, int(scores.size * RELEASED_MIN_AREA_RATIO))
    labels, count = ndimage.label(
        scores > RELEASED_MASK_THRESHOLD, structure=np.ones((3, 3), dtype=bool)
    )
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= minimum_area
    keep[0] = False
    return keep[labels]


def binary_metrics(tp: int, positives: int, predicted_positives: int, total: int) -> dict:
    fp = predicted_positives - tp
    fn = positives - tp
    negatives = total - positives
    f1_denominator = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": negatives - fp,
        "f1": 2 * tp / f1_denominator if f1_denominator else None,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
    }


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    _, validation, captures = load_splits(args.dataset, "can", 42)
    if args.smoke:
        regular = [capture for capture in captures if capture.lighting == "regular"]
        captures = [next(capture for capture in regular if bool(capture.mask) == label)
                    for label in (False, True)]
        validation = validation[:2]

    print(f"Verifying and loading released Can checkpoint on {device}", flush=True)
    started = time.perf_counter()
    with args.weights.open("rb") as stream:
        checkpoint_sha256 = file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected SHA-256 for the released Can checkpoint: {checkpoint_sha256}")
    model, transform = load_cat(args.source, args.weights, device)
    kernel = gaussian_kernel(device)
    print(
        f"Scoring {len(validation)} clean validation captures and {len(captures)} public captures",
        flush=True,
    )

    def score(path: Path) -> np.ndarray:
        return predict_map(model, transform, kernel, read_image(path), device, args.batch_size)

    # Image-level counts use this project's normal-only part-score calibration.
    # AU-PRO uses the stitched author outputs and requires no decision threshold.
    validation_scores = [float(score(path).max()) / 255 for path in validation]
    threshold = float(np.quantile(validation_scores, 0.90))
    groups = defaultdict(list)
    for capture in captures:
        groups[capture.lighting].append(capture)

    metrics, records = {}, []
    args.output.mkdir(parents=True, exist_ok=True)
    for light in sorted(groups, key=lambda name: (name != "regular", name)):
        results, times, released_rejections = [], [], []
        pixel_tp = pixel_positives = pixel_predicted_positives = pixels = 0
        for capture in groups[light]:
            capture_started = time.perf_counter()
            anomaly = score(capture.image)
            elapsed = (time.perf_counter() - capture_started) * 1000
            times.append(elapsed)
            mask = None
            if capture.mask is not None:
                with Image.open(capture.mask) as opened:
                    mask = np.asarray(opened).copy()
            result = ImageResult(
                capture.image.relative_to(args.dataset).as_posix(),
                capture.scene, light, mask is not None, float(anomaly.max()) / 255, anomaly, mask,
            )
            results.append(result)
            binary = released_mask(anomaly)
            flagged = int(np.count_nonzero(binary))
            released_rejections.append(bool(flagged))
            pixels += binary.size
            pixel_predicted_positives += flagged
            if mask is not None:
                pixel_positives += int(np.count_nonzero(mask))
                pixel_tp += int(np.count_nonzero(binary & (mask != 0)))
            records.append({
                "image": result.image, "scene": result.scene, "lighting": light,
                "is_defective": result.is_defective, "score": result.score,
                "rejected": result.score >= threshold, "released_mask_rejected": bool(flagged),
                "inference_ms": elapsed,
            })
        part_labels = [result.is_defective for result in results]
        released_part_tp = sum(label and rejected for label, rejected in zip(
            part_labels, released_rejections, strict=True,
        ))
        metrics[light] = {
            **summarize(results, threshold),
            "inference_ms_median": float(np.median(times)),
            "inference_ms_p95": float(np.quantile(times, 0.95)),
            "released_binary_rule": {
                "pixels": binary_metrics(pixel_tp, pixel_positives, pixel_predicted_positives, pixels),
                "any_mask_pixel_part_rule": binary_metrics(
                    released_part_tp, sum(part_labels), sum(released_rejections), len(results),
                ),
            },
        }
        print(
            f"{light}: {len(results)} captures, AU-PRO={metrics[light]['au_pro_05']:.4f}",
            flush=True,
        )

    write_predictions(args.output / "predictions.csv", records)
    report = {
        "category": "can",
        "source_revision": REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": "public_smoke" if args.smoke else "public_can",
        "tile_grid": [GRID_SIZE, GRID_SIZE],
        "inference_input_size": INPUT_SIZE,
        "center_crop": False,
        "localization_scores": "author uint8 scores stitched at native resolution",
        "part_threshold_rule": "90th percentile of validation-normal maximum scores divided by 255",
        "released_binary_rule": {
            "source": "author settings for both Can private splits, applied to the public split",
            "strict_uint8_threshold": RELEASED_MASK_THRESHOLD,
            "component_connectivity": 8,
            "minimum_component_pixels": RELEASED_MIN_AREA_PIXELS,
            "minimum_component_area_ratio": RELEASED_MIN_AREA_RATIO,
            "adaptive_threshold": False,
        },
        "validation_images": len(validation),
        "score_threshold": threshold,
        "device": str(device),
        "batch_size": args.batch_size,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "by_lighting": metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved CAT public results to {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", type=Path, default=Path("data/mvtec_ad_2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1, help="Tiles per forward pass")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Run two validation and two public images")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= GRID_SIZE**2:
        parser.error("batch size must be between 1 and 16")
    args.output = args.output or Path("artifacts/cat/can-smoke" if args.smoke else "artifacts/cat/can")
    run(args)


if __name__ == "__main__":
    main()
