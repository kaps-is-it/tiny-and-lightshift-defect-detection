"""Run the position-conditioned experiment against the deep/tiled control.

    python run_position.py can
    python run_position.py sheet_metal fabric
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmark import summarise_by_light
from detail_branch import BranchFusion
from detector import PatchDetector
from evaluation import ImageResult
from position_map import PositionConditionedBank
from run_experiment import load_splits, predict_map, read_image, tile_origins


def pack(captures, masks, maps):
    return [ImageResult(c.image.name, c.scene, c.lighting, m is not None,
                        float(a.max()), a, m)
            for c, m, a in zip(captures, masks, maps, strict=True)]


def by_lighting(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[r.lighting].append(r)
    return grouped


def run(category: str, args: argparse.Namespace) -> None:
    training, validation, captures = load_splits(args.dataset, category, args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    mean = np.array((0.485, 0.456, 0.406), np.float32)
    std = np.array((0.229, 0.224, 0.225), np.float32)

    print(f"fitting position-conditioned bank on {len(training)} good images", flush=True)
    bank = PositionConditionedBank(device=device).fit(training)

    print("fitting deep reference bank (control)", flush=True)
    from run_experiment import training_batches, training_views
    detector = PatchDetector(device, bank_size=4096, seed=args.seed)
    detector.fit(training_batches(training_views(training, False, args.seed), 256, 32))

    print("scoring deep control", flush=True)
    test_arrays = [np.asarray(read_image(c.image), dtype=np.float32) for c in captures]
    val_arrays = [np.asarray(read_image(p), dtype=np.float32) for p in validation]
    test_deep = [predict_map(detector, Image.fromarray(a.clip(0, 255).astype(np.uint8)), 256, 32) for a in test_arrays]
    val_deep = [predict_map(detector, Image.fromarray(a.clip(0, 255).astype(np.uint8)), 256, 32) for a in val_arrays]
    if args.skip_deep:
        test_deep = None

    print("scoring position-conditioned bank", flush=True)
    test_pos = [bank.score(a) for a in test_arrays]
    val_pos = [bank.score(a) for a in val_arrays]

    fusion = BranchFusion().fit(val_deep if test_deep is not None else val_pos, val_arrays)

    masks = []
    for capture in captures:
        mask = None
        if capture.mask:
            with Image.open(capture.mask) as truth:
                mask = np.asarray(truth) != 0
        masks.append(mask)

    variants = {}
    if test_deep is not None:
        variants["control_deep"] = pack(captures, masks, test_deep)
        variants["deep_dark_branch"] = pack(captures, masks,
                                            [fusion.fuse(d, a, 0.75, "dark") for d, a in zip(test_deep, test_arrays)])
    variants["position_map"] = pack(captures, masks, test_pos)
    # Branch fused onto the position map (standardised there)
    variants["position_branch_w1.0"] = pack(captures, masks,
                                            [fusion.fuse(d, a, 1.0, "dark") for d, a in zip(test_pos, test_arrays)])
    # Deep + position, each standardised against good validation, summed
    if test_deep is not None:
        def spread(x):
            return max(np.percentile(x, 99) - np.percentile(x, 50), 1e-6)
        dma = np.concatenate([d.ravel() for d in val_deep])
        pmu = np.concatenate([d.ravel() for d in val_pos])
        dmed, dspr = float(np.median(dma)), float(spread(dma))
        pmed, pspr = float(np.median(pmu)), float(spread(pmu))
        combined = [((d - dmed) / dspr + (p - pmed) / pspr).astype(np.float32)
                    for d, p in zip(test_deep, test_pos)]
        variants["position_plus_deep"] = pack(captures, masks, combined)

    print("\nAU-PRO @ 5% FPR (mean across lights | worst light)")
    for name, rows in variants.items():
        summary = summarise_by_light(by_lighting(rows))
        detail = "  ".join(f"{l[:7]}={summary['per_light'][l]['au_pro_05']*100:.1f}" for l in sorted(summary["per_light"]))
        print(f"  {name:<22} mean {summary['au_pro_mean']*100:5.1f}% | worst {summary['au_pro_worst']*100:5.1f}% | tiny {summary['tiny_pro_mean']*100:4.1f}% | {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("categories", nargs="+")
    parser.add_argument("--dataset", default=Path("mvtec_ad_2_dataset"), type=Path)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-deep", action="store_true")
    args = parser.parse_args()
    for category in args.categories:
        print(f"\n########## {category} ##########")
        run(category, args)


if __name__ == "__main__":
    main()
