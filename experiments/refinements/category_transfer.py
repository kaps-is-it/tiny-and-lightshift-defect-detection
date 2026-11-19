#!/usr/bin/env python
"""Round 8: fabric champion ensemble {42,7,99}; sheet_metal champion bright ensemble {42,7}."""
from __future__ import annotations

import gc
import time
from collections import defaultdict


from experiments.refinements.hybrid_maps import ensemble_eval
from local_contrast.run_experiment import load_splits
from local_contrast.benchmark import summarise_by_light
from local_contrast.ensemble_maps import DATASET, DEVICE, eval_champion, fit_champion, report, test_maps


def part_q():
    print("=" * 115, flush=True)
    print("Q) Fabric champion ensemble {42, 7, 99}", flush=True)
    training, validation, captures = load_splits(DATASET, "fabric", 42)
    base42, base7, base99 = [], [], []
    for seed in (42, 7, 99):
        t0 = time.perf_counter()
        det, fusion, _, _ = fit_champion(training, validation, seed)
        print(f"[fit fabric seed {seed}] {time.perf_counter() - t0:.0f}s", flush=True)
        deep_test = test_maps(det, captures)
        s, by_light = eval_champion(fusion, deep_test, captures)
        report(f"fabric seed {seed}", s)
        base42.append(by_light)
        del det, fusion, deep_test
        gc.collect()
    # fabric ensemble {42, 7, 99} = mean of maps
    ens = defaultdict(list)
    for light in base42[0]:
        for rs in zip(base42[0][light], base42[1][light], base42[2][light]):
            m = sum(r.anomaly_map for r in rs) / 3
            r0 = rs[0]
            ens[light].append(r0.__class__(r0.image, r0.scene, r0.lighting, r0.is_defective,
                                           float(m.max()), m, r0.mask))
    report("fabric champion ensemble {42,7,99}", summarise_by_light(ens))
    ensemble_eval([base42[0], base42[1]], "fabric ensemble {42, 7}")


def part_r():
    print("\n" + "=" * 115, flush=True)
    print("R) Sheet-metal champion bright ensemble {42, 7} (w=0.5, maxf=5, scale=2)", flush=True)
    training, validation, captures = load_splits(DATASET, "sheet_metal", 42)
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    deep_test42 = test_maps(det42, captures)
    s42e, base42e = eval_champion(fusion42, deep_test42, captures, weight=0.5, polarity="bright")
    report("sheet_metal seed 42 bright w=0.5", s42e)

    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    deep_test7 = test_maps(det7, captures)
    s7e, base7e = eval_champion(fusion7, deep_test7, captures, weight=0.5, polarity="bright")
    report("sheet_metal seed 7 bright w=0.5", s7e)
    ensemble_eval([base42e, base7e], "sheet_metal champion bright ensemble {42, 7} (w=0.5)")


def main():
    print(f"device: {DEVICE}", flush=True)
    part_q()
    part_r()
    print("\nDONE8", flush=True)


if __name__ == "__main__":
    main()
