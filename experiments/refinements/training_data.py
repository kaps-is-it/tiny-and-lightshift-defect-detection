#!/usr/bin/env python
"""Round 9: more-data champion variants + champion maps on private splits.

S) champion ensemble {42e,7e,99e} with extended training:
   S1: train + validation
   S2: train + public good test captures
T) champion ensemble with lighting augmentation in training
U) champion maps on test_private / test_private_mixed (seed 42, sample)
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import numpy as np

from experiments.refinements.hybrid_maps import ensemble_eval
from PIL import Image
from local_contrast.detail_branch import BranchFusion
from local_contrast.detector import PatchDetector
from local_contrast.ensemble_maps import BATCH, DATASET, DEVICE, TILE, champion_scores, eval_champion, report, test_maps
from local_contrast.run_experiment import load_splits, predict_map, read_image, training_batches, training_views


def fit_champion_ex(training, validation, seed, augment=False):
    det = PatchDetector(DEVICE, 4096, seed)
    det.fit(training_batches(training_views(training, augment, seed), TILE, BATCH))
    val_images = [np.asarray(read_image(p)) for p in validation]
    deep_val = [predict_map(det, Image.fromarray(im), TILE, BATCH) for im in val_images]
    fusion = BranchFusion(radius=25, eps=1.0).fit(deep_val, val_images)
    return det, fusion


def main():
    print(f"device: {DEVICE}", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    variants = [
        ("train+validation", training + validation),
        ("train+public_good", training + [c.image for c in captures if c.mask is None]),
    ]
    saved = []
    for tag, ext_training in variants:
        print("\n" + "=" * 115, flush=True)
        print(f"S) champion ensemble {{42e,7e,99e}} with training = {tag}", flush=True)
        by_lists = []
        dets = []
        for seed in (42, 7, 99):
            t0 = time.perf_counter()
            det, fusion = fit_champion_ex(ext_training, validation, seed)
            print(f"[fit seed {seed} ({tag})] {time.perf_counter() - t0:.0f}s", flush=True)
            deep_test = test_maps(det, captures)
            s, base = eval_champion(fusion, deep_test, captures)
            report(f"seed {seed} ({tag})", s)
            by_lists.append(base)
            dets.append((det, fusion))
            del deep_test
            gc.collect()
        ensemble_eval(by_lists, f"champion ensemble {{42e,7e,99e}} ({tag})")
        if tag == "train+validation":
            saved = dets

    print("\n" + "=" * 115, flush=True)
    print("T) champion ensemble {42e,7e,99e} with lighting augmentation in training", flush=True)
    by_lists = []
    for seed in (42, 7, 99):
        t0 = time.perf_counter()
        det, fusion = fit_champion_ex(training, validation, seed, augment=True)
        print(f"[fit seed {seed} augmented] {time.perf_counter() - t0:.0f}s", flush=True)
        deep_test = test_maps(det, captures)
        s, base = eval_champion(fusion, deep_test, captures)
        report(f"seed {seed} augmented", s)
        by_lists.append(base)
        del det, fusion, deep_test
        gc.collect()
    ensemble_eval(by_lists, "champion ensemble {42e,7e,99e} (augmented)")

    print("\n" + "=" * 115, flush=True)
    print("U) champion maps on private splits (seed 42 champion, sample) — sanity", flush=True)
    det, fusion = saved[0]
    for split in ("test_private", "test_private_mixed"):
        images = sorted((DATASET / "can" / split).glob("*.png"))[::6]
        by_light = defaultdict(list)
        for p in images:
            deep = predict_map(det, read_image(p), TILE, BATCH)
            m = champion_scores(fusion, deep, read_image(p))
            by_light[p.stem.split("_", 1)[-1] if "_" in p.stem else "none"].append(float(m.max()))
        parts = [f"{light}: n={len(vals)} mean={np.mean(vals):.3f} med={np.median(vals):.3f}"
                 for light, vals in sorted(by_light.items())]
        print(f"  {split}: " + "; ".join(parts), flush=True)
    print("\nDONE9", flush=True)


if __name__ == "__main__":
    main()
