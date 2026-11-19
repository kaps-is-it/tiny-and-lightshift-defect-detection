#!/usr/bin/env python
"""Round 5:
H) Can champion ensemble combos (42, 42e, 7, 7e)
I) Hybrid: champion + native fusion maps (Can)
J) Champion ensemble on wallplugs + vial
K) Champion + position bank (Can)
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import numpy as np
import torch

import research_extra2 as r2
from research_extra2 import (fit_champion, champion_scores, test_maps, eval_champion, report,
                             summarize_maps, champion_val_maps, DATASET, DEVICE, BATCH)
from run_experiment import load_splits, training_batches, training_views, read_image
from benchmark import summarise_by_light
from evaluation import ImageResult

INPUT_SIZE = 256


def ensemble_eval(maps, tag):
    ens = defaultdict(list)
    for light in maps[0]:
        for rs in zip(*[m[light] for m in maps]):
            m = sum(r.anomaly_map for r in rs) / len(rs)
            r0 = rs[0]
            ens[light].append(ImageResult(r0.image, r0.scene, r0.lighting, r0.is_defective,
                                          float(m.max()), m, r0.mask))
    report(tag, summarise_by_light(ens))


def part_h():
    print("=" * 115, flush=True)
    print("H) Can champion ensemble combos", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    deep_test42 = test_maps(det42, captures)
    s42, base42 = eval_champion(fusion42, deep_test42, captures)
    report("seed 42 default", s42)
    fusion42e = r2.BranchFusion(radius=25, eps=1.0).fit(deep_val42, val_images)
    s42e, base42e = eval_champion(fusion42e, deep_test42, captures)
    report("seed 42 eps=1.0", s42e)

    t0 = time.perf_counter()
    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    print(f"[fit seed 7] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test7 = test_maps(det7, captures)
    s7, base7 = eval_champion(fusion7, deep_test7, captures)
    report("seed 7 default", s7)
    fusion7e = r2.BranchFusion(radius=25, eps=1.0).fit(deep_val7, val_images7)
    s7e, base7e = eval_champion(fusion7e, deep_test7, captures)
    report("seed 7 eps=1.0", s7e)

    ensemble_eval([base42, base7], "ensemble {42, 7}")
    ensemble_eval([base42e, base7], "ensemble {42e, 7}")
    ensemble_eval([base42, base7, base7e], "ensemble {42, 7, 7e}")
    ensemble_eval([base42e, base7e], "ensemble {42e, 7e}")
    del det42, fusion42, fusion42e, det7, fusion7, fusion7e, deep_test42, deep_test7
    gc.collect()


def part_i():
    print("\n" + "=" * 115, flush=True)
    print("I) Hybrid: champion + native fusion maps (Can)", flush=True)
    from multiscale_detector import MultiScaleDetector
    from multiscale_experiment import predict_branches, add_fusion

    training, validation, captures = load_splits(DATASET, "can", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    deep_test = test_maps(det, captures)
    s, base = eval_champion(fusion, deep_test, captures)
    report("champion seed 42", s)

    native_det = MultiScaleDetector(DEVICE, 4096, 1024, 42)
    t0 = time.perf_counter()
    native_det.fit(training_batches(training_views(training, False, 42), INPUT_SIZE, BATCH))
    print(f"[native fit] {time.perf_counter() - t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    val_maps = [predict_branches(native_det, read_image(p), BATCH) for p in validation]
    print(f"[native val maps] {time.perf_counter() - t0:.0f}s", flush=True)
    scales = {name: max(1e-8, float(np.median([m[name].max() for m in val_maps])))
              for name in ("context", "fine")}
    print(f"  native scales: {scales}", flush=True)

    champ_maps = [champion_scores(fusion, deep_test[i], read_image(c.image)) for i, c in enumerate(captures)]
    for w in (0.25, 0.5, 1.0):
        hybrid = [c + w * n for c, n in zip(champ_maps, [add_fusion(predict_branches(native_det, read_image(c.image), BATCH), scales)["fusion"] for c in captures])]
        summary, _ = summarize_maps(captures, hybrid)
        report(f"champion + {w} x native_fusion", summary)
    del det, fusion, deep_test, champ_maps, native_det, val_maps
    gc.collect()


def part_j():
    print("\n" + "=" * 115, flush=True)
    print("J) Wallplugs/vial champion ensemble {42,7}", flush=True)
    for category in ("wallplugs", "vial"):
        training, validation, captures = load_splits(DATASET, category, 42)
        det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
        deep_test42 = test_maps(det42, captures)
        s42, base42 = eval_champion(fusion42, deep_test42, captures)
        report(f"{category} seed 42", s42)
        det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
        deep_test7 = test_maps(det7, captures)
        s7, base7 = eval_champion(fusion7, deep_test7, captures)
        report(f"{category} seed 7", s7)
        ensemble_eval([base42, base7], f"{category} ensemble {{42,7}}")
        del det42, fusion42, det7, fusion7, deep_test42, deep_test7, deep_val42, deep_val7
        gc.collect()


def part_k():
    print("\n" + "=" * 115, flush=True)
    print("K) Champion + position bank (Can)", flush=True)
    from position_map import PositionConditionedBank

    training, validation, captures = load_splits(DATASET, "can", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    deep_test = test_maps(det, captures)
    s, base = eval_champion(fusion, deep_test, captures)
    report("champion default", s)

    t0 = time.perf_counter()
    bank = PositionConditionedBank(DEVICE).fit(validation)
    print(f"[position bank fit on {len(validation)} validation] {time.perf_counter() - t0:.0f}s", flush=True)
    champ_maps = [champion_scores(fusion, deep_test[i], read_image(c.image)) for i, c in enumerate(captures)]
    for alpha in (0.5, 1.0):
        t0 = time.perf_counter()
        hybrid = [m + alpha * bank.score(np.asarray(read_image(c.image), dtype=np.float32) / 255.0)
                  for m, c in zip(champ_maps, captures)]
        print(f"[pos maps scored {time.perf_counter() - t0:.0f}s]", flush=True)
        summary, _ = summarize_maps(captures, hybrid)
        report(f"champion + {alpha} x position_bank", summary)
    del det, fusion, deep_test, champ_maps, bank
    gc.collect()


def main():
    print(f"device: {DEVICE}", flush=True)
    part_h()
    part_i()
    part_j()
    part_k()
    print("\nDONE5", flush=True)


if __name__ == "__main__":
    main()
