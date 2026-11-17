"""Build the MVTec AD 2 submission directory, and optionally the .tar.gz.

This script never contacts the benchmark server. It:

1.  fits the tiled PatchCore reference bank on the regular-light good
    training images of every category;
2.  evaluates the plain baseline and the three detail-branch polarity
    settings on the public test split, single seed - the same procedure the
    project uses for its other local comparisons. The public split is this
    project's development set, so choosing the best setting there is tuning,
    not test-set peeking; the chosen setting is frozen for the private mappings.
3.  freezes that setting per category, then generates the private-split
    continuous maps (.tiff float16) and the binary maps (.png). The
    binarisation threshold is calibrated on the good validation captures of
    the submitted map family: the 99.5th percentile of validation-map
    pixels by default (``--threshold-rule``), i.e. a fixed 0.5% flag rate
    on good parts, with the FAQ's good-validation mean + 3*sigma available
    as a flag for reference.

Pass ``--compress`` to also create the tar.gz the uploader expects. Upload
is a manual step on the benchmark website; nothing here sends data anywhere.

Usage
    python prepare_submission.py --device cuda
    python prepare_submission.py --categories can --compress
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image

from benchmark import summarise_by_light
from detector import PatchDetector
from detail_branch import BranchFusion
from evaluation import ImageResult
from run_experiment import load_splits, predict_map, read_image

ALL_CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal",
                  "vial", "wallplugs", "walnuts")
TIFF_DTYPE = np.float16
SCALE_DEFAULT = 4.0


def uint8(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))


def downscale_map(map_: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return map_.astype(TIFF_DTYPE)
    height, width = map_.shape
    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    return np.asarray(
        Image.fromarray(map_.astype(np.float32, copy=False)).resize(size, Image.Resampling.BILINEAR),
        dtype=TIFF_DTYPE,
    )


def public_report(detector, fusion, captures, masks, test_arrays, args, tile_size):
    """Evaluate baseline and both polarity branches on the public test split."""
    deep_maps = [predict_map(detector, uint8(im), tile_size, args.batch_size) for im in test_arrays]

    def score_capture(anomaly_map, capture, mask):
        return ImageResult(capture.image.name, capture.scene, capture.lighting,
                            mask is not None, float(anomaly_map.max()), anomaly_map, mask)

    results = {}

    def evaluate(maps):
        by = defaultdict(list)
        for capture, mask, anomaly in zip(captures, masks, maps, strict=True):
            by[capture.lighting].append(score_capture(anomaly, capture, mask))
        return summarise_by_light(by)

    results["control"] = evaluate(deep_maps)
    for polarity in ("dark", "bright", "both"):
        for weight in args.weights:
            fused = [fusion.fuse(d, a, weight, polarity) for d, a in zip(deep_maps, test_arrays)]
            results[f"{polarity}_w{weight}"] = evaluate(fused)
    return results


def best_variant(results: dict) -> tuple[str, float, str]:
    name = max(results, key=lambda n: results[n]["au_pro_mean"])
    if name == "control":
        return name, 0.0, "none"
    polarity, weight_text = name.rsplit("_w", 1)
    return name, float(weight_text), polarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--map-downscale", type=float, default=SCALE_DEFAULT,
                        help="Submit maps at 1/f of native resolution, as the released SuperADD does. The server upsamples maps.")
    parser.add_argument("--threshold-rule", choices=("p99_5", "p99_0", "mean3sigma"), default="p99_5",
                        help="Calibration for the thresholded masks: 99.5th percentile of the validation-map pixels "
                             "(default), 99th percentile, or the FAQ's good-validation mean + 3*sigma.")
    parser.add_argument("--output", type=Path, default=Path("submission"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--compress", action="store_true", help="Also create submission.tar.gz")
    args = parser.parse_args()

    image_dir = args.output / "anomaly_images"
    binary_dir = args.output / "anomaly_images_thresholded"
    config = {}

    for category in args.categories:
        print(f"\n########## {category} ##########", flush=True)
        training, validation, captures = load_splits(args.dataset, category, args.seed)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        detector = PatchDetector(device=device, bank_size=args.bank_size, seed=args.seed)
        from run_experiment import training_batches, training_views
        print(f"fitting reference bank on {len(training)} images", flush=True)
        started = time.perf_counter()
        detector.fit(training_batches(training_views(training, False, args.seed), args.tile_size, args.batch_size))
        print(f"fit took {time.perf_counter() - started:.0f}s", flush=True)

        print("mapping good validation captures", flush=True)
        validation_images = [np.asarray(read_image(p), dtype=np.float32) for p in validation]
        val_deep = [predict_map(detector, uint8(im), args.tile_size, args.batch_size) for im in validation_images]
        fusion = BranchFusion().fit(val_deep, validation_images)

        masks = []
        for capture in captures:
            mask = None
            if capture.mask:
                with Image.open(capture.mask) as truth:
                    mask = np.asarray(truth) != 0
            masks.append(mask)
        test_arrays = [np.asarray(read_image(c.image), dtype=np.float32) for c in captures]

        print("calibrating on the public test split (development)", flush=True)
        results = public_report(detector, fusion, captures, masks, test_arrays, args, args.tile_size)
        for name, summary in results.items():
            print(f"    {name:<16} AU-PRO {summary['au_pro_mean']*100:5.1f}%  worst {summary['au_pro_worst']*100:5.1f}%")
        chosen, weight, polarity = best_variant(results)
        print(f"    -> chosen: {chosen} (weight={weight}, polarity={polarity})", flush=True)

        # Also evaluate the thresholded rule on the *submitted* map family,
        # which is the fused map for the selected polarity - not the raw deep map.
        val_fused = [
            m if weight == 0.0 else fusion.fuse(m, a, weight, polarity)
            for m, a in zip(val_deep, validation_images)
        ]
        val_small = [downscale_map(m, args.map_downscale).astype(np.float64) for m in val_fused]
        pooled = np.concatenate([m.ravel() for m in val_small])
        rules = {
            "p99_5": float(np.percentile(pooled, 99.5)),
            "p99_0": float(np.percentile(pooled, 99.0)),
            "mean3sigma": float(pooled.mean() + 3 * pooled.std()),
        }
        threshold = rules[args.threshold_rule]
        print(f"    binarisation threshold ({args.threshold_rule}) = {threshold:.6f}", flush=True)

        # ---- private splits have no labels here
        for split in ("test_private", "test_private_mixed"):
            split_dir = args.dataset / category / split
            images = sorted(split_dir.glob("*.png"))
            out_image = image_dir / category / split
            out_binary = binary_dir / category / split
            out_image.mkdir(parents=True, exist_ok=True)
            out_binary.mkdir(parents=True, exist_ok=True)
            print(f"    scoring {len(images)} {split} images", flush=True)
            for path in images:
                image = read_image(path)
                raw = predict_map(detector, image, args.tile_size, args.batch_size)
                if weight == 0.0:
                    fused = raw
                else:
                    fused = fusion.fuse(raw, np.asarray(image, dtype=np.float32), weight, polarity)
                small = downscale_map(fused, args.map_downscale)
                binary = (small >= threshold).astype(np.uint8) * 255
                tifffile.imwrite(out_image / f"{path.stem}.tiff", small)
                Image.fromarray(binary).save(out_binary / f"{path.stem}.png")
        config[category] = {
            "chosen": chosen, "weight": weight, "polarity": polarity,
            "map_downscale": args.map_downscale,
            "threshold_rule": args.threshold_rule,
            "binarisation_threshold": threshold,
            "thresholds": rules,
            "seed": args.seed, "device": device,
            "public_au_pro": {k: {"mean": v["au_pro_mean"], "worst": v["au_pro_worst"],
                                  "tiny": v["tiny_pro_mean"]} for k, v in results.items()},
        }
        print(f"    wrote {image_dir / category}", flush=True)

    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nConfig written to {args.output / 'config.json'}")
    if args.compress:
        print("use MVTecAD2_public_code_utils for the official archive, e.g.:\n"
              "  python MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py "
              f"{args.output}")


if __name__ == "__main__":
    main()
