#!/usr/bin/env python
"""Round 4: fixed region rule on champion maps (Can, E), champion-ensemble
decisions via SpatialScorer top-tail (Can, F), champion ensemble on fabric (G)."""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import numpy as np

from local_contrast.spatial_scoring import SpatialScorer
from experiments.refinements.detail_sweeps import NormalRegions, champion_val_maps, hotspots, region_rule_counts
from local_contrast.benchmark import summarise_by_light
from local_contrast.ensemble_maps import DATASET, eval_champion, fit_champion, report, test_maps
from local_contrast.run_experiment import load_splits


def part_e():
    print("=" * 115, flush=True)
    print("E) Region scorer on champion maps (Can)", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    val_maps = champion_val_maps(fusion, deep_val, val_images)
    examples = [r for m in val_maps for r in hotspots(m)]
    print(f"  {len(examples)} validation hotspots for the normal-region scorer", flush=True)
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


def part_f():
    print("\n" + "=" * 115, flush=True)
    print("F) Champion ensemble (Can): peak vs SpatialScorer top-tail decisions", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    deep_test42 = test_maps(det42, captures)
    s42, base42 = eval_champion(fusion42, deep_test42, captures)
    report("seed 42 default", s42)

    t0 = time.perf_counter()
    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    print(f"[fit seed 7] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test7 = test_maps(det7, captures)
    s7, base7 = eval_champion(fusion7, deep_test7, captures)
    report("seed 7 default", s7)

    val_maps42 = champion_val_maps(fusion42, deep_val42, val_images)
    val_maps7 = champion_val_maps(fusion7, deep_val7, val_images7)
    ens_val = [(a + b) / 2 for a, b in zip(val_maps42, val_maps7)]
    sp = SpatialScorer().fit(ens_val)
    thr = float(np.quantile([sp.score(m) for m in ens_val], 0.9))
    print(f"  SpatialScorer ensemble p90 thr={thr:.4f}", flush=True)
    caught_c = fp_c = 0
    n_def_c = n_all_c = 0
    for light in sorted(base42, key=lambda n: (n != "regular", n)):
        c = f = 0
        for r42, r7 in zip(base42[light], base7[light]):
            m = (r42.anomaly_map + r7.anomaly_map) / 2
            if sp.score(m) >= thr:
                if r42.is_defective:
                    c += 1
                else:
                    f += 1
        if light != "regular":
            caught_c += c
            fp_c += f
            n_def_c += sum(1 for r in base42[light] if r.is_defective)
            n_all_c += len(base42[light])
    print(f"  ensemble champion: SpatialScorer caught {caught_c}/{n_def_c} defective, false rejects {fp_c}/{n_all_c - n_def_c} good", flush=True)
    del det42, fusion42, det7, fusion7, deep_test42, deep_test7, deep_val42, deep_val7, val_maps42, val_maps7
    gc.collect()


def part_g():
    print("\n" + "=" * 115, flush=True)
    print("G) Fabric champion ensemble (seeds 42, 7)", flush=True)
    training, validation, captures = load_splits(DATASET, "fabric", 42)
    t0 = time.perf_counter()
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    print(f"[fit fabric seed 42] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test42 = test_maps(det42, captures)
    s42, base42 = eval_champion(fusion42, deep_test42, captures)
    report("fabric seed 42 champion", s42)

    t0 = time.perf_counter()
    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    print(f"[fit fabric seed 7] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test7 = test_maps(det7, captures)
    s7, base7 = eval_champion(fusion7, deep_test7, captures)
    report("fabric seed 7 champion", s7)

    ens = defaultdict(list)
    for light in base42:
        for r42, r7 in zip(base42[light], base7[light]):
            m = (r42.anomaly_map + r7.anomaly_map) / 2
            ens[light].append(r42.__class__(r42.image, r42.scene, r42.lighting, r42.is_defective,
                                            float(m.max()), m, r42.mask))
    report("fabric champion ensemble {42,7}", summarise_by_light(ens))
    del det42, fusion42, det7, fusion7, deep_test42, deep_test7, deep_val42, deep_val7
    gc.collect()


def main():
    part_e()
    part_f()
    part_g()
    print("\nDONE4", flush=True)


if __name__ == "__main__":
    main()
