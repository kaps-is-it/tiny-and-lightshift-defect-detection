"""Two cheap, principled variants on the champion configuration:

1. Seed ensemble  — average the anomaly maps of 3 independently-fitted
   detectors. No extra data, no threshold change.
2. Smoothing radius — sweep the final Gaussian filter sigma from 2.0 down.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from benchmark import summarise_by_light, trivial_maps
from detail_branch import BranchFusion
from detector import PatchDetector
from evaluation import ImageResult
from run_experiment import load_splits, predict_map, read_image


def pack(captures, masks, maps):
    return [ImageResult(c.image.name, c.scene, c.lighting, m is not None,
                        float(a.max()), a, m)
            for c, m, a in zip(captures, masks, maps, strict=True)]


def by_lighting(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[r.lighting].append(r)
    return grouped


def run(category: str) -> None:
    training, validation, captures = load_splits(Path("mvtec_ad_2_dataset"), category, 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from run_experiment import training_batches, training_views
    seeds = (42, 7, 1234)
    detectors = []
    for seed in seeds:
        det = PatchDetector(device, bank_size=4096, seed=seed)
        det.fit(training_batches(training_views(training, False, seed), 256, 32))
        detectors.append(det)

    test_arrays = [np.asarray(read_image(c.image), dtype=np.float32) for c in captures]
    val_arrays = [np.asarray(read_image(p), dtype=np.float32) for p in validation]
    masks = []
    for c in captures:
        mask = None
        if c.mask:
            with Image.open(c.mask) as t:
                mask = np.asarray(t) != 0
        masks.append(mask)

    val_deep = [predict_map(d, Image.fromarray(a.clip(0, 255).astype(np.uint8)), 256, 32)
                for a in val_arrays for d in [detectors[0]]]
    # build ensemble maps
    print("mapping images (single + ensemble)", flush=True)
    single_maps = []
    ensemble_maps = []
    for i, a in enumerate(test_arrays):
        am42 = predict_map(detectors[0], Image.fromarray(a.clip(0, 255).astype(np.uint8)), 256, 32)
        am07 = predict_map(detectors[1], Image.fromarray(a.clip(0, 255).astype(np.uint8)), 256, 32)
        am12 = predict_map(detectors[2], Image.fromarray(a.clip(0, 255).astype(np.uint8)), 256, 32)
        single_maps.append(am42)
        ensemble_maps.append((am42 + am07 + am12) / 3.0)
        if i % 30 == 0:
            print(f"  {i}/{len(test_arrays)}", flush=True)

    fusion = BranchFusion().fit(val_deep, val_arrays)
    detail_maps = [fusion.fuse(m, a, 0.5, "dark") for m, a in zip(ensemble_maps, test_arrays)]
    detail_single = [fusion.fuse(m, a, 0.5, "dark") for m, a in zip(single_maps, test_arrays)]

    # smoothing sweep on the champion config (detail fused on ensemble)
    def swept(maps, sigma):
        return [gaussian_filter(m, sigma=sigma) for m in maps]

    variants = {
        "control_seed42": pack(captures, masks, single_maps),
        "ensemble_3seed": pack(captures, masks, ensemble_maps),
        "control_seed42_s0.5": pack(captures, masks, swept(single_maps, 0.5)),
        "ensemble_s0.5": pack(captures, masks, swept(ensemble_maps, 0.5)),
        "ensemble_detail_w0.5": pack(captures, masks, detail_maps),
        "ensemble_detail_w0.5_s0.5": pack(captures, masks, swept(detail_maps, 0.5)),
        "single_detail_w0.5": pack(captures, masks, detail_single),
    }
    for name, rows in variants.items():
        summary = summarise_by_light(by_lighting(rows))
        print(f"  {name:<26} mean {summary['au_pro_mean']*100:5.1f}% | worst {summary['au_pro_worst']*100:5.1f}% | tiny {summary['tiny_pro_mean']*100:4.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("categories", nargs="+")
    args = parser.parse_args()
    for category in args.categories:
        print(f"\n########## {category} ##########")
        run(category)


if __name__ == "__main__":
    main()
