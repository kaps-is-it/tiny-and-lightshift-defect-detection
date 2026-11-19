#!/usr/bin/env python
"""Round 3: fixed SPARC-lite on champion maps (Can) + region rule on champion maps."""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import numpy as np

import research_extra2 as r2
from research_extra2 import (fit_champion, champion_scores, test_maps, eval_champion, report,
                             summarize_maps, champion_val_maps, hotspots, NormalRegions,
                             region_rule_counts, DATASET)
from run_experiment import load_splits, tile_origins

read_image = r2.read_image


def sparc_lite_maps_fixed(test_maps_list, val_maps_list, val_images, tile_size=256):
    acc = defaultdict(lambda: [0.0, 0])
    for m, im in zip(val_maps_list, val_images):
        H, W = np.asarray(im).shape[:2]
        for y in tile_origins(H, tile_size):
            for x in tile_origins(W, tile_size):
                ex, ey = min(x + tile_size, W), min(y + tile_size, H)
                acc[(y, x)][0] += float(m[y:ey, x:ex].mean())
                acc[(y, x)][1] += 1
    means = {k: v[0] / v[1] for k, v in acc.items()}
    out = []
    for m in test_maps_list:
        H, W = m.shape[:2]
        corr = np.zeros_like(m)
        for y in tile_origins(H, tile_size):
            for x in tile_origins(W, tile_size):
                ex, ey = min(x + tile_size, W), min(y + tile_size, H)
                corr[y:ey, x:ex] = means.get((y, x), 0.0)
        out.append(m - corr)
    return out


def part_d():
    print("=" * 115, flush=True)
    print("D) SPARC-lite: per-cell mean correction on Can champion maps (fixed)", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    deep_test = test_maps(det, captures)
    s42, base42 = eval_champion(fusion, deep_test, captures)
    report("champion default (re-measured)", s42)
    val_maps = champion_val_maps(fusion, deep_val, val_images)
    test_maps_list = [champion_scores(fusion, deep_test[i], read_image(c.image)) for i, c in enumerate(captures)]
    corrected = sparc_lite_maps_fixed(test_maps_list, val_maps, val_images)
    print(f"  corrected maps: n={len(corrected)} min={min(m.min() for m in corrected):.3f} "
          f"max={max(m.max() for m in corrected):.3f}", flush=True)
    summary, _ = summarize_maps(captures, corrected)
    report("Can champion + SPARC-lite", summary)
    del det, fusion, deep_test, deep_val, val_maps, test_maps_list, corrected
    gc.collect()


def part_e():
    print("\n" + "=" * 115, flush=True)
    print("E) Region scorer on champion maps (Can)", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    val_maps = champion_val_maps(fusion, deep_val, val_images)
    examples = [region for m in val_maps for region in hotspots(m)]
    scorer = NormalRegions(examples)
    val_rows = [max((scorer.score(r) for r in hotspots(m)), default=0.0) for m in val_maps]
    thr = float(np.quantile(val_rows, 0.9))
    print(f"  region threshold (p90 on validation) = {thr:.4f}", flush=True)
    deep_test = test_maps(det, captures)
    _, base = eval_champion(fusion, deep_test, captures)
    caught_c, fp_c, n_def_c, n_all_c = region_rule_counts(base, scorer, thr)
    print(f"  champion region rule: changed: caught {caught_c}/{n_def_c} defective, false rejects {fp_c}/{n_all_c - n_def_c} good", flush=True)
    del det, fusion, deep_test, deep_val, val_maps
    gc.collect()


def main():
    part_d()
    part_e()
    print("\nDONE3", flush=True)


if __name__ == "__main__":
    main()
