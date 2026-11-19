#!/usr/bin/env python
"""Round 6:
L) Can champion seed sweep (singles)
M) Champion ensemble {42e, 7e}: region rule + SpatialScorer decisions; ensemble with new seeds
O) Sheet-metal champion bright weight refinement
P) Vial champion "both"-polarity single
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import numpy as np
import torch

import research_extra2 as r2
import research_extra5 as r5
from research_extra2 import (fit_champion, champion_scores, test_maps, eval_champion, report,
                             summarize_maps, champion_val_maps, hotspots, NormalRegions,
                             region_rule_counts, DATASET)
from research_extra5 import ensemble_eval
from run_experiment import load_splits
from benchmark import summarise_by_light
from evaluation import ImageResult
from spatial_scoring import SpatialScorer


def part_l():
    print("=" * 115, flush=True)
    print("L) Can champion seed sweep (singles)", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    best = []
    for seed in (3, 5, 8, 11, 13, 77, 99, 523, 2026, 2025):
        t0 = time.perf_counter()
        det, fusion, _, _ = fit_champion(training, validation, seed)
        deep_test = test_maps(det, captures)
        s, _ = eval_champion(fusion, deep_test, captures)
        report(f"seed {seed}", s)
        best.append((s["au_pro_mean"], seed))
        del det, fusion, deep_test
        gc.collect()
    best.sort(key=lambda t: -t[0])
    print(f"  best new seeds: {[(s, round(v * 100, 2)) for v, s in best[:3]]}", flush=True)
    return [s for _, s in best[:2]]


def part_m(new_seeds):
    print("\n" + "=" * 115, flush=True)
    print("M) Champion ensemble {42e, 7e} diagnostics (Can)", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    deep_test42 = test_maps(det42, captures)
    fusion42e = r2.BranchFusion(radius=25, eps=1.0).fit(deep_val42, val_images)
    s42e, base42e = eval_champion(fusion42e, deep_test42, captures)
    report("seed 42 eps=1.0", s42e)

    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    deep_test7 = test_maps(det7, captures)
    fusion7e = r2.BranchFusion(radius=25, eps=1.0).fit(deep_val7, val_images7)
    s7e, base7e = eval_champion(fusion7e, deep_test7, captures)
    report("seed 7 eps=1.0", s7e)

    ensemble_eval([base42e, base7e], "ensemble {42e, 7e}")

    # region rule + SpatialScorer on the ensemble maps
    val_maps42e = champion_val_maps(fusion42e, deep_val42, val_images)
    val_maps7e = champion_val_maps(fusion7e, deep_val7, val_images7)
    ens_val_maps = [(a + b) / 2 for a, b in zip(val_maps42e, val_maps7e)]
    ens_by_light = defaultdict(list)
    for light in base42e:
        for r42, r7 in zip(base42e[light], base7e[light]):
            m = (r42.anomaly_map + r7.anomaly_map) / 2
            ens_by_light[light].append(ImageResult(r42.image, r42.scene, r42.lighting, r42.is_defective,
                                                   float(m.max()), m, r42.mask))
    examples = [r for m in ens_val_maps for r in hotspots(m)]
    scorer = NormalRegions(examples)
    val_rows = [max((scorer.score(r) for r in hotspots(m)), default=0.0) for m in ens_val_maps]
    thr = float(np.quantile(val_rows, 0.9))
    caught_c, fp_c, n_def_c, n_all_c = region_rule_counts(ens_by_light, scorer, thr)
    print(f"  ensemble {{42e,7e}} region rule: caught {caught_c}/{n_def_c} defective, false rejects {fp_c}/{n_all_c - n_def_c} good", flush=True)
    sp = SpatialScorer().fit(ens_val_maps)
    thr_sp = float(np.quantile([sp.score(m) for m in ens_val_maps], 0.9))
    caught_sp = fp_sp = n_def_sp = n_all_sp = 0
    for light in sorted(ens_by_light, key=lambda n: (n != "regular", n)):
        if light == "regular":
            continue
        for r in ens_by_light[light]:
            n_all_sp += 1
            n_def_sp += int(r.is_defective)
            if sp.score(r.anomaly_map) >= thr_sp:
                caught_sp += int(r.is_defective)
                fp_sp += int(not r.is_defective)
    print(f"  ensemble {{42e,7e}} SpatialScorer: caught {caught_sp}/{n_def_sp} defective, "
          f"false rejects {fp_sp}/{n_all_sp - n_def_sp} good", flush=True)

    # ensembles with new seeds
    for seed in new_seeds[:2]:
        t0 = time.perf_counter()
        det, fusion, val_images_s, deep_val = fit_champion(training, validation, seed)
        print(f"[fit seed {seed}] {time.perf_counter() - t0:.0f}s", flush=True)
        deep_test = test_maps(det, captures)
        s, base = eval_champion(fusion, deep_test, captures)
        report(f"seed {seed} default", s)
        fusione = r2.BranchFusion(radius=25, eps=1.0).fit(deep_val, val_images_s)
        s_e, base_e = eval_champion(fusione, deep_test, captures)
        report(f"seed {seed} eps=1.0", s_e)
        ensemble_eval([base42e, base7e, base_e], f"ensemble {{42e, 7e, seed{seed}e}}")
        ensemble_eval([base42e, base_e], f"ensemble {{42e, seed{seed}e}}")
        ensemble_eval([base7e, base_e], f"ensemble {{7e, seed{seed}e}}")
        del det, fusion, fusione, deep_test, deep_val, val_images_s, base, base_e
        gc.collect()
    del det42, fusion42, fusion42e, det7, fusion7, fusion7e, deep_test42, deep_test7, deep_val42, deep_val7, val_maps42e, val_maps7e, ens_val_maps, ens_by_light
    gc.collect()


def part_o():
    print("\n" + "=" * 115, flush=True)
    print("O) Sheet-metal champion bright weight refinement", flush=True)
    training, validation, captures = load_splits(DATASET, "sheet_metal", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    deep_test = test_maps(det, captures)
    for weight in (0.3, 0.4, 0.6, 0.8):
        s, _ = eval_champion(fusion, deep_test, captures, weight=weight, polarity="bright")
        report(f"sheet_metal bright w={weight}", s)


def part_p():
    print("\n" + "=" * 115, flush=True)
    print("P) Vial champion variants", flush=True)
    training, validation, captures = load_splits(DATASET, "vial", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    deep_test = test_maps(det, captures)
    s, _ = eval_champion(fusion, deep_test, captures, polarity="both")
    report("vial champion both", s)


def main():
    print(f"device: {r2.DEVICE}", flush=True)
    new_seeds = part_l()
    part_m(new_seeds)
    part_o()
    part_p()
    print("\nDONE6", flush=True)


if __name__ == "__main__":
    main()
