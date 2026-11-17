"""Fit the tiled detector and evaluate the detail branch + learned combiner.

Examples
    python run_detail.py can
    python run_detail.py can sheet_metal fabric --synthetic 64
    python run_detail.py can --mask-background

The reference detector is fitted on regular-light good training images only, the
fusion statistics and the combiner are fitted on good validation captures, and
every reported number is computed on the public test captures. The combiner is
trained on synthetic defects only, never on real defect masks.

``--mask-background`` additionally reports a "product-only" evaluation where
pixels that the training set identifies as static backdrop are ranked lowest.
That is a dataset-specific property (in this set no defect pixel ever lands on
the holder paper) and is reported separately so the special case is visible.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmark import (
    compare_region_level,
    summarise_by_light,
    trivial_maps,
)
from detail_branch import BranchFusion
from detector import PatchDetector
from evaluation import ImageResult
from fusion import LearnedFusion
from run_experiment import load_splits, predict_map, read_image
from synthetic import foreground_mask, inject


def build_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))


def deep_maps_for(detector, images, tile_size, batch_size):
    return [predict_map(detector, build_image(im), tile_size, batch_size) for im in images]


def pack(captures, masks, maps):
    return [
        ImageResult(capture.image.name, capture.scene, capture.lighting,
                    mask is not None, float(anomaly.max()), anomaly, mask)
        for capture, mask, anomaly in zip(captures, masks, maps, strict=True)
    ]


def by_lighting(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[result.lighting].append(result)
    return grouped


def apply_foreground(maps, foreground):
    """Rank static-background pixels at the bottom of each map."""
    out = []
    for map_ in maps:
        masked = np.array(map_, copy=True)
        masked[~foreground] = masked.min()
        out.append(masked)
    return out


def run(args: argparse.Namespace) -> None:
    for category in args.categories:
        print(f"\n########## {category} ##########", flush=True)
        training, validation, captures = load_splits(args.dataset, category, args.seed)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        detector = PatchDetector(device=device, bank_size=args.bank_size, seed=args.seed)
        started = time.perf_counter()
        from run_experiment import training_batches, training_views
        detector.fit(training_batches(training_views(training, False, args.seed), args.tile_size, args.batch_size))
        print(f"reference bank fitted in {time.perf_counter() - started:.0f}s ({len(training)} images)", flush=True)

        foreground = foreground_mask(training)
        print(f"foreground mask keeps {foreground.mean() * 100:.1f}% of pixels", flush=True)

        print("mapping good validation captures", flush=True)
        validation_images = [np.asarray(read_image(p), dtype=np.float32) for p in validation]
        val_deep = deep_maps_for(detector, validation_images, args.tile_size, args.batch_size)
        fusion = BranchFusion().fit(val_deep, validation_images)

        print("mapping public test captures", flush=True)
        test_arrays = [np.asarray(read_image(c.image), dtype=np.float32) for c in captures]
        test_deep = deep_maps_for(detector, test_arrays, args.tile_size, args.batch_size)
        test_masks = []
        for capture in captures:
            mask = None
            if capture.mask:
                with Image.open(capture.mask) as truth:
                    mask = np.asarray(truth) != 0
            test_masks.append(mask)

        print("building synthetic defects", flush=True)
        rng = np.random.default_rng(args.seed)
        synthetic = []
        for path in rng.choice(training, size=min(len(training), args.synthetic), replace=False):
            source = np.asarray(read_image(path), dtype=np.float32)
            defective, mask = inject(source, rng, foreground=foreground)
            synthetic.append((defective, mask))
        syn_deep = deep_maps_for(detector, [im for im, _ in synthetic], args.tile_size, args.batch_size)

        combiner = LearnedFusion().fit_fusion(val_deep, validation_images)
        syn_foreground = [foreground] * len(synthetic)
        combiner.fit(syn_deep, [im for im, _ in synthetic], [m for _, m in synthetic],
                     seed=args.seed, foregrounds=syn_foreground)
        print("combiner coefficients:", combiner.report(), flush=True)

        # ---------------- evaluate variants ----------------
        trivial = [trivial_maps(build_image(a)) for a in test_arrays]
        variants = {
            "control_deep_only": pack(captures, test_masks, test_deep),
            "trivial_sobel": pack(captures, test_masks, [t["sobel"] for t in trivial]),
            "trivial_darkness": pack(captures, test_masks, [t["darkness"] for t in trivial]),
            "learned_combiner": pack(captures, test_masks, [combiner.scores(d, a) for d, a in zip(test_deep, test_arrays)]),
        }
        for weight in args.weights:
            for polarity in ("dark", "bright", "both"):
                fused = [fusion.fuse(d, a, weight, polarity) for d, a in zip(test_deep, test_arrays)]
                variants[f"hand_{polarity}_w{weight}"] = pack(captures, test_masks, fused)

        if args.mask_background:
            fg_masks = {}
            masked_variants = {}
            for name, rows in variants.items():
                fg_masks[name] = apply_foreground([r.anomaly_map for r in rows], foreground)
                masked_variants[name + "_fgmask"] = pack(captures, test_masks, fg_masks[name])
            variants.update(masked_variants)

        summaries = {name: summarise_by_light(by_lighting(rows)) for name, rows in variants.items()}
        control = variants["control_deep_only"]
        comparisons = {name: compare_region_level(control, rows) for name, rows in variants.items() if name != "control_deep_only"}

        report = {
            "category": category,
            "seed": args.seed,
            "device": device,
            "tile_size": args.tile_size,
            "bank_size": args.bank_size,
            "synthetic_defects": len(synthetic),
            "foreground_fraction": float(foreground.mean()),
            "fusion_weights": args.weights,
            "combiner": combiner.report(),
            "variants": summaries,
            "region_level_vs_control": comparisons,
        }
        out_dir = args.output / category
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        print("\nAU-PRO @ 5% FPR (mean | worst lighting)")
        for name, summary in summaries.items():
            per = summary["per_light"]
            detail = "  ".join(f"{l[:7]}={per[l]['au_pro_05'] * 100:.1f}" for l in per)
            print(f"  {name:<24} mean {summary['au_pro_mean']*100:5.1f}% | worst {summary['au_pro_worst']*100:5.1f}% | "
                  f"tiny {summary['tiny_pro_mean']*100:5.1f}% | {detail}")
        for name, comp in comparisons.items():
            direction = "localises better" if (comp["mean_delta_percentile"] or 0) >= 0 else "localises worse"
            print(f"  region-level {name:<24} {direction}: "
                  f"{comp['regions_improved']}/{comp['regions']} improved, "
                  f"median delta {comp['median_delta_percentile']:+.2f} percentile, "
                  f"sign p={comp['sign_test_p']}")
        print(f"Saved {out_dir / 'benchmark.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="+")
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic", type=int, default=64, help="synthetic defects used to fit the combiner")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.75])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--mask-background", action="store_true",
                        help="Report an extra product-only evaluation where static backdrop pixels are ranked lowest")
    parser.add_argument("--output", type=Path, default=Path("artifacts/detail"))
    args = parser.parse_args()
    if args.synthetic < 1:
        parser.error("--synthetic must be positive")
    run(args)


if __name__ == "__main__":
    main()
