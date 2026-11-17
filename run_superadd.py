"""Evaluate upstream SuperADD on the public MVTec AD 2 can captures."""

import argparse
import csv
import importlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from build_scene_manifest import build_manifest
from evaluation import ImageResult, summarize


_DEPS = Path(__file__).resolve().parent / "artifacts" / "superadd-deps"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))

TIMM_MODELS = {
    "dinov3_vith16plus": "vit_huge_plus_patch16_dinov3_qkvb.lvd1689m",
    "dinov3_vits16": "vit_small_patch16_dinov3_qkvb.lvd1689m",
}
SMALL_TRAIN_STRIDE = 16


class TimmBackbone:
    """Expose the Meta DINOv3 intermediate-token interface used by SuperADD."""

    model_patch_size = 16

    def __init__(self, model_name: str, layers: list[int], device: str, weights: Path):
        import timm

        self.model = timm.create_model(TIMM_MODELS[model_name], checkpoint_path=str(weights)).to(device).eval()
        # Match the positional-embedding frequency precision of the Meta model.
        self.model.rope.periods = self.model.rope.periods.to(torch.bfloat16).to(torch.float32)
        self.layers = layers
        self.window_batch = 1 if model_name == "dinov3_vith16plus" else 8

    @torch.inference_mode()
    def __call__(self, windows: torch.Tensor) -> list[torch.Tensor]:
        # Keep attention memory bounded, especially for the released ViT-H+ model.
        features = [
            self.model.forward_intermediates(
                window,
                indices=self.layers,
                norm=False,
                output_fmt="NLC",
                intermediates_only=True,
            )
            for window in windows.split(self.window_batch)
        ]
        return [torch.cat(layer) for layer in zip(*features, strict=True)]


def load_upstream(checkout: Path, weights: Path):
    sys.path[:0] = [str(checkout / "utils"), str(checkout / "tracks/industrial/src")]
    upstream = importlib.import_module("industrial.model")
    nearest_neighbors = upstream.nearest_neighbors

    def bounded_neighbors(queries, keys, knn_neighbors, normalize=False):
        # The upstream operation is unchanged, but split its query rows so the
        # intermediate distance matrix does not exhaust laptop GPU memory.
        max_rows = max(1, (64 * 1024**2) // (keys.element_size() * len(keys)))
        matches = [
            nearest_neighbors(chunk, keys, knn_neighbors, normalize=normalize)
            for chunk in queries.split(max_rows)
        ]
        return tuple(torch.cat(column) for column in zip(*matches, strict=True))

    upstream.DinoV3Backbone = lambda name, layers, device: TimmBackbone(name, layers, device, weights)
    upstream.nearest_neighbors = bounded_neighbors
    return upstream.SuperADD


def read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"))
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def train_small(superadd, dataset: Path, resize_factor: float):
    """Fit a deliberately smaller reference using a deterministic subset of good images."""
    training = sorted((dataset / "can" / "train" / "good").glob("*.png"))[::SMALL_TRAIN_STRIDE]
    if len(training) < 8:
        raise ValueError("Expected the complete MVTec AD 2 can training split")
    np.random.seed(42)
    model = superadd(
        backbone="dinov3_vits16", layers=[2, 5, 8, 11],
        threshold_fraction=8, resize_factor=resize_factor, patch_size=384, patch_overlap=64,
        max_database_size=12000, subsampling_iterations=100, threshold_percentile=95,
        threshold_factor=1.421, evaluation_downscale=4, closing_radius=26,
        closing_angles=16, closing_lower_threshold=0.8, binary_erosion=1,
        brightness_augmentation=(0.8, 1.2), device="cuda",
    )
    model.train([read_image(path) for path in training])
    return model, len(training)


def predict_capture(model, dataset: Path, capture: dict[str, str]):
    tensor = read_image(dataset / capture["image"])
    height, width = tensor.shape[-2:]
    start = time.perf_counter()
    anomaly_map, binary_map = model.predict(tensor)
    scores = np.asarray(
        Image.fromarray(anomaly_map).resize((width, height), Image.Resampling.BILINEAR)
    )
    binary = np.asarray(
        Image.fromarray(binary_map.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
    ) > 0
    elapsed = (time.perf_counter() - start) * 1000
    mask = None
    if capture["mask"]:
        with Image.open(dataset / capture["mask"]) as label:
            mask = np.asarray(label.convert("L")) > 0
    return ImageResult(
        capture["image"], capture["scene"], capture["lighting"],
        capture["label"] == "bad", float(anomaly_map.max()), scores, mask,
    ), binary, elapsed


def evaluate(model, dataset: Path, output: Path, metadata: dict) -> None:
    captures = defaultdict(list)
    for capture in build_manifest(dataset, "can"):
        captures[capture["lighting"]].append(capture)
    predictions = []
    summaries = {}
    for lighting, group in sorted(captures.items()):
        results = []
        latencies = []
        pixel_true = pixel_predicted = pixel_hit = 0
        postprocess_misses = postprocess_false_rejects = 0
        for capture in group:
            result, binary, elapsed = predict_capture(model, dataset, capture)
            results.append(result)
            latencies.append(elapsed)
            rejected = bool(binary.any())
            predictions.append({
                "image": result.image,
                "scene": result.scene,
                "lighting": result.lighting,
                "is_defective": result.is_defective,
                "score": result.score,
                "rejected": result.score >= model.threshold,
                "postprocessed_rejected": rejected,
                "inference_ms": elapsed,
            })
            postprocess_misses += result.is_defective and not rejected
            postprocess_false_rejects += not result.is_defective and rejected
            pixel_predicted += int(np.count_nonzero(binary))
            if result.mask is not None:
                pixel_true += int(np.count_nonzero(result.mask))
                pixel_hit += int(np.count_nonzero(binary & result.mask))

        summary = summarize(results, model.threshold)
        summary.update({
            "segmentation_f1": 2 * pixel_hit / (pixel_true + pixel_predicted),
            "postprocessed_misses": postprocess_misses,
            "postprocessed_false_rejects": postprocess_false_rejects,
            "inference_ms_median": float(np.median(latencies)),
        })
        summaries[lighting] = summary
        print(
            f"{lighting}: AU-PRO {summary['au_pro_05']:.3f}, "
            f"pixel F1 {summary['segmentation_f1']:.3f}, "
            f"{summary['fn']} missed, {summary['fp']} false rejects", flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "category": "can",
        **metadata,
        "backbone": TIMM_MODELS[model.backbone_name],
        "layers": model.layers,
        "patch_size": model.patch_exec.patch_size,
        "patch_overlap": model.patch_exec.patch_overlap,
        "resize_factor": model.preprocessing.resize_factor,
        "bank_vectors_per_layer": {
            str(layer): len(vectors) for layer, vectors in model.prototype_embeddings.items()
        },
        "score_threshold": float(model.threshold),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "by_lighting": summaries,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("artifacts/upstream-superadd"))
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--bank", type=Path, default=Path("artifacts/superadd-assets/can"),
                        help="Path without extension to the authors' matching .json and .npz files")
    parser.add_argument("--weights", type=Path, help="Local timm DINOv3 safetensors checkpoint")
    parser.add_argument("--output", type=Path, help="Output directory for the local comparison")
    parser.add_argument("--train-small", action="store_true",
                        help="Run a reduced ViT-S/16 variant fitted on a subset of good training images")
    parser.add_argument("--native", action="store_true",
                        help="Extract features at native input scale in the reduced ViT-S/16 variant")
    parser.add_argument("--smoke", action="store_true", help="Run a single regular-light capture and report peak CUDA memory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("The authors' released can configuration requires CUDA")
    if args.native and not args.train_small:
        parser.error("--native requires --train-small")
    if args.weights is None:
        model_size = "vits16" if args.train_small else "vith16plus"
        args.weights = Path(f"artifacts/superadd-assets/dinov3-{model_size}-timm.safetensors")
    required = [args.weights]
    if not args.train_small:
        required.extend((args.bank.with_suffix(".json"), args.bank.with_suffix(".npz")))
    for path in required:
        if not path.is_file():
            parser.error(f"Required SuperADD asset not found: {path}")
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    superadd = load_upstream(args.upstream.resolve(), args.weights)
    metadata = {"upstream_revision": revision}
    if args.train_small:
        model, train_count = train_small(superadd, args.dataset, 1.0 if args.native else 0.625)
        metadata.update({
            "method": "Reduced SuperADD",
            "training_stride": SMALL_TRAIN_STRIDE,
            "training_image_count": train_count,
            "threshold_training_image_count": (train_count + model.threshold_fraction - 1) // model.threshold_fraction,
        })
    else:
        model = superadd.from_disk(args.bank)
        metadata["method"] = "SuperADD with the authors' released reference bank"
    if args.smoke:
        capture = next(capture for capture in build_manifest(args.dataset, "can") if capture["lighting"] == "regular")
        result, _, elapsed = predict_capture(model, args.dataset, capture)
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{result.image}: {elapsed:.0f} ms, score {result.score:.6f}, {peak_mib:.0f} MiB peak allocated")
        return
    default_run = "can-small" if args.train_small else "can"
    if args.native:
        default_run += "-native"
    output = args.output or (Path("artifacts/superadd") / default_run)
    evaluate(model, args.dataset, output, metadata)


if __name__ == "__main__":
    main()
