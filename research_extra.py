#!/usr/bin/env python
"""Exploratory sweeps around the Can champion (local_contrast) recipe, plus
champion measurements on other categories.

Targets (published champion, all-six lights): AU-PRO 63.94%, changed-5 63.85%,
tiny overlap 81.61%.

Usage: python research_extra.py
"""
from __future__ import annotations

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "local_contrast"))
sys.path.insert(1, str(REPO))

from detector import PatchDetector
from detail_branch import BranchFusion
from prepare_submission import finalize_map, roundtrip_map
from benchmark import summarise_by_light
from evaluation import ImageResult, summarize_decisions
from run_experiment import load_splits, predict_map, read_image, training_batches, training_views

TILE, BATCH = 256, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = REPO / "mvtec_ad_2_dataset"


def fit_champion(training, validation, seed, radius=None, eps=0.5):
    det = PatchDetector(DEVICE, 4096, seed)
    det.fit(training_batches(training_views(training, False, seed), TILE, BATCH))
    val_images = [np.asarray(read_image(p)) for p in validation]
    deep_val = [predict_map(det, Image.fromarray(im), TILE, BATCH) for im in val_images]
    fusion = BranchFusion(radius=radius, eps=eps).fit(deep_val, val_images)
    return det, fusion, val_images, deep_val


def champion_scores(fusion, deep, image, weight=0.75, maxf=5, scale=2.0, smooth=2.0, polarity="dark"):
    d = gaussian_filter(deep, smooth) if smooth > 0 else deep
    fused = fusion.fuse(d, np.asarray(image), weight, polarity)
    return roundtrip_map(finalize_map(fused, maxf, 99.5), scale)


def test_maps(det, captures):
    return [predict_map(det, read_image(c.image), TILE, BATCH) for c in captures]


def eval_champion(fusion, deep_test, captures, weight=0.75, maxf=5, scale=2.0, smooth=2.0, polarity="dark"):
    by_light = defaultdict(list)
    for i, c in enumerate(captures):
        scores = champion_scores(fusion, deep_test[i], read_image(c.image), weight, maxf, scale, smooth, polarity)
        mask = None
        if c.mask:
            with Image.open(c.mask) as t:
                mask = np.asarray(t) != 0
        by_light[c.lighting].append(
            ImageResult(c.image.name, c.scene, c.lighting, c.mask is not None, float(scores.max()), scores, mask))
    return summarise_by_light(by_light), by_light


def changed_mean(summary):
    lights = [l for l in summary["per_light"] if l != "regular"]
    vals = [summary["per_light"][l]["au_pro_05"] for l in lights]
    return float(np.mean(vals)) if vals else float("nan")


def report(tag, summary):
    cm = changed_mean(summary)
    cm_str = f"{cm * 100:6.2f}%" if cm == cm else "   nan"
    print(
        f"  {tag:46s} all_lights={summary['au_pro_mean'] * 100:6.2f}% changed_lights={cm_str} "
        f"worst={summary['au_pro_worst'] * 100:6.2f}% tiny={summary['tiny_pro_mean'] * 100:6.2f}% "
        f"dpct={summary['defect_percentile']:6.2f} fpag={summary['fp_agreement']:.3f}",
        flush=True,
    )


def per_light_table(summary):
    for light in sorted(summary["per_light"], key=lambda n: (n != "regular", n)):
        m = summary["per_light"][light]
        auroc = m["image_auroc"]
        auroc_s = f"{auroc:5.2f}" if auroc is not None and auroc == auroc else "  nan"
        print(f"      {light:10s} AU-PRO={m['au_pro_05'] * 100:6.2f}% tiny={m['tiny_pro_at_05'] * 100:6.2f}% "
              f"auroc={auroc_s} n={m['images']} def={m['defective']}", flush=True)


def decision_report(by_light, threshold):
    rows = []
    caught_all = fp_all = caught_changed = fp_changed = 0
    n_def_changed = n_all_changed = n_all = 0
    for light in sorted(by_light, key=lambda n: (n != "regular", n)):
        m = summarize_decisions(by_light[light], threshold)
        tp = m["defective"] - m["fn"]
        rows.append((light, tp, m["fn"], m["fp"], m["defective"], len(by_light[light])))
        caught_all += tp
        fp_all += m["fp"]
        if light != "regular":
            caught_changed += tp
            fp_changed += m["fp"]
            n_def_changed += m["defective"]
            n_all_changed += len(by_light[light])
        n_all += len(by_light[light])
    return caught_all, fp_all, caught_changed, fp_changed, n_def_changed, n_all_changed, n_all, rows


def can_part():
    print("=" * 115, flush=True)
    print("CAN: champion baseline + sweeps  (targets: all6 63.94%, changed5 63.85%, tiny 81.61%)", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    print(f"can: {len(training)} train / {len(validation)} validation / {len(captures)} public captures", flush=True)

    t0 = time.perf_counter()
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    print(f"[fit seed 42] {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    deep_test42 = test_maps(det42, captures)
    print(f"[test maps seed 42] {time.perf_counter() - t0:.0f}s", flush=True)

    print("\n-- baseline: champion defaults (r=25, eps=0.5, w=0.75, maxf=5, scale=2, smooth=2, seed=42)", flush=True)
    base, base_light = eval_champion(fusion42, deep_test42, captures)
    report("champion default", base)
    per_light_table(base)

    print("\n-- Stage A: zscore radius x eps (w=0.75, maxf=5, scale=2, smooth=2, seed=42)", flush=True)
    stage_a = []
    for radius in (25, 51, 101):
        for eps in (0.5, 1.0):
            fusion = BranchFusion(radius=radius, eps=eps).fit(deep_val42, val_images)
            s, _ = eval_champion(fusion, deep_test42, captures)
            report(f"r={radius} eps={eps}", s)
            stage_a.append((s["au_pro_mean"], radius, eps))
    stage_a.sort(key=lambda t: -t[0])
    print(f"  Stage A top-3: {[(r[1], r[2], round(r[0] * 100, 2)) for r in stage_a[:3]]}", flush=True)

    r_best, e_best = stage_a[0][1], stage_a[0][2]
    fusion_b = BranchFusion(radius=r_best, eps=e_best).fit(deep_val42, val_images)
    print(f"\n-- Stage B: weight/maxf/scale/smooth (r={r_best}, eps={e_best})", flush=True)
    stage_b = []
    for weight in (0.5, 0.75, 1.0):
        for maxf in (5, 9):
            for scale in (2.0, 1.0):
                s, _ = eval_champion(fusion_b, deep_test42, captures, weight=weight, maxf=maxf, scale=scale)
                report(f"w={weight} maxf={maxf} scale={scale}", s)
                stage_b.append((s["au_pro_mean"], weight, maxf, scale))
    for smooth in (0, 2):
        s, _ = eval_champion(fusion_b, deep_test42, captures, weight=0.75, maxf=5, scale=2.0, smooth=smooth)
        report(f"smooth={smooth} (w=0.75, maxf=5, scale=2)", s)
    stage_b.sort(key=lambda t: -t[0])
    print(f"  Stage B top-3: {[(r[1], r[2], r[3], round(r[0] * 100, 2)) for r in stage_b[:3]]}", flush=True)

    print("\n-- Stage C: seeds 7 and 1234 (champion defaults)", flush=True)
    seed_maps = {42: base_light}
    for seed in (7, 1234):
        t0 = time.perf_counter()
        det, fusion, _, _ = fit_champion(training, validation, seed)
        print(f"[fit seed {seed}] {time.perf_counter() - t0:.0f}s", flush=True)
        deep_test = test_maps(det, captures)
        s, by_light = eval_champion(fusion, deep_test, captures)
        report(f"seed {seed}", s)
        seed_maps[seed] = by_light
        del det, fusion, deep_test
        gc.collect()

    print("\n-- ensemble: mean of champion maps (seeds 42, 7, 1234)", flush=True)
    ens = defaultdict(list)
    for light in seed_maps[42]:
        acc = [None] * len(seed_maps[42][light])
        for s in seed_maps:
            for j, r in enumerate(seed_maps[s][light]):
                acc[j] = r.anomaly_map if acc[j] is None else acc[j] + r.anomaly_map
        n = len(seed_maps)
        ens[light] = [
            ImageResult(r.image, r.scene, r.lighting, r.is_defective, float((a / n).max()), a / n, r.mask)
            for r, a in zip(seed_maps[42][light], acc)
        ]
    summary_ens = summarise_by_light(ens)
    report("ensemble mean maps (42,7,1234)", summary_ens)
    per_light_table(summary_ens)

    del deep_test42
    gc.collect()

    print("\n-- Stage D: seeds 0, 1, 123 (champion defaults)", flush=True)
    for seed in (0, 1, 123):
        t0 = time.perf_counter()
        det, fusion, _, _ = fit_champion(training, validation, seed)
        print(f"[fit seed {seed}] {time.perf_counter() - t0:.0f}s", flush=True)
        deep_test = test_maps(det, captures)
        s, _ = eval_champion(fusion, deep_test, captures)
        report(f"seed {seed}", s)
        del det, fusion, deep_test
        gc.collect()

    print("\n-- Stage E: part-threshold rules on champion maps (seed 42 default)", flush=True)
    peaks_val = [champion_scores(fusion42, deep_val42[i], val_images[i]).max() for i in range(len(validation))]
    rules = {
        "p90 (champion)": float(np.quantile(peaks_val, 0.9)),
        "p95": float(np.quantile(peaks_val, 0.95)),
        "p99": float(np.quantile(peaks_val, 0.99)),
        "p99.5": float(np.quantile(peaks_val, 0.995)),
        "mean+3sd": float(np.mean(peaks_val) + 3 * np.std(peaks_val)),
    }
    for name, thr in rules.items():
        caught, fp, caught_c, fp_c, n_def_c, n_all_c, n_all, rows = decision_report(base_light, thr)
        print(
            f"  {name:16s} thr={thr:7.4f} | changed: caught {caught_c}/{n_def_c} defective, false rejects {fp_c}/{n_all_c - n_def_c} good | "
            f"all: caught {caught}/{sum(row[4] for row in rows)} defective",
            flush=True,
        )


def other_part():
    print("=" * 115, flush=True)
    print("Champion measurements on other categories (default champion recipe, seed 42)", flush=True)
    for category in ("fabric", "wallplugs", "vial", "rice", "sheet_metal"):
        print(f"\n== {category} ==", flush=True)
        training, validation, captures = load_splits(DATASET, category, 42)
        t0 = time.perf_counter()
        det, fusion, _, _ = fit_champion(training, validation, 42)
        print(f"[fit {category} seed 42] {time.perf_counter() - t0:.0f}s", flush=True)
        deep_test = test_maps(det, captures)
        s, _ = eval_champion(fusion, deep_test, captures)
        report(f"{category} champion", s)
        if category == "sheet_metal":
            s2, _ = eval_champion(fusion, deep_test, captures, polarity="bright")
            report(f"{category} champion bright", s2)
        del det, fusion, deep_test
        gc.collect()


def main():
    print(f"device: {DEVICE}", flush=True)
    can_part()
    other_part()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
