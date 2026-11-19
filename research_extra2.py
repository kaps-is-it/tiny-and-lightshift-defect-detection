#!/usr/bin/env python
"""Round 2: champion ensembles (Can), champion sweeps on sheet_metal/fabric,
SPARC-lite position correction on champion maps, region rule on champion maps."""
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
from run_experiment import load_splits, predict_map, read_image, training_batches, training_views, tile_origins

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


def summarize_maps(captures, maps):
    by_light = defaultdict(list)
    for c, m in zip(captures, maps, strict=True):
        mask = None
        if c.mask:
            with Image.open(c.mask) as t:
                mask = np.asarray(t) != 0
        by_light[c.lighting].append(
            ImageResult(c.image.name, c.scene, c.lighting, c.mask is not None, float(m.max()), m, mask))
    return summarise_by_light(by_light), by_light


def champion_val_maps(fusion, deep_val, val_images):
    return [champion_scores(fusion, deep_val[i], val_images[i]) for i in range(len(val_images))]


def ensemble(base_a, base_b, tag):
    ens = defaultdict(list)
    for light in base_a:
        for ra, rb in zip(base_a[light], base_b[light]):
            m = (ra.anomaly_map + rb.anomaly_map) / 2
            ens[light].append(ImageResult(ra.image, ra.scene, ra.lighting, ra.is_defective,
                                          float(m.max()), m, ra.mask))
    summary = summarise_by_light(ens)
    report(tag, summary)
    return summary, ens


def decision_counts(by_light, threshold):
    caught_c = fp_c = 0
    n_def_c = n_all_c = n_all = 0
    for light in sorted(by_light, key=lambda n: (n != "regular", n)):
        c = f = 0
        for r in by_light[light]:
            if r.score >= threshold:
                if r.is_defective:
                    c += 1
                else:
                    f += 1
        if light != "regular":
            caught_c += c
            fp_c += f
            n_def_c += sum(1 for r in by_light[light] if r.is_defective)
            n_all_c += len(by_light[light])
        n_all += len(by_light[light])
    return caught_c, fp_c, n_def_c, n_all_c, n_all


def part_a():
    print("=" * 115, flush=True)
    print("A) Can champion: ensembles and seed-7 eps=1.0", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    print(f"can: {len(training)} train / {len(validation)} validation / {len(captures)} public captures", flush=True)

    t0 = time.perf_counter()
    det42, fusion42, val_images, deep_val42 = fit_champion(training, validation, 42)
    print(f"[fit seed 42] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test42 = test_maps(det42, captures)
    s42, base42 = eval_champion(fusion42, deep_test42, captures)
    report("seed 42 (champion default)", s42)

    t0 = time.perf_counter()
    det7, fusion7, val_images7, deep_val7 = fit_champion(training, validation, 7)
    print(f"[fit seed 7] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test7 = test_maps(det7, captures)
    s7, base7 = eval_champion(fusion7, deep_test7, captures)
    report("seed 7 (champion default)", s7)

    fusion7e = BranchFusion(radius=25, eps=1.0).fit(deep_val7, val_images7)
    deep_test7e = test_maps(det7, captures)
    s7e, base7e = eval_champion(fusion7e, deep_test7e, captures)
    report("seed 7 eps=1.0", s7e)

    print("\n-- ensembles: mean of champion maps --", flush=True)
    _, ens42_7 = ensemble(base42, base7, "ensemble {42, 7} (default maps)")
    _, ens42_7e = ensemble(base42, base7e, "ensemble {42, 7@eps=1.0}")

    val_maps42 = champion_val_maps(fusion42, deep_val42, val_images)
    for tag, ens, second_fusion in (
        ("ensemble {42, 7}", ens42_7, fusion7),
        ("ensemble {42, 7@eps=1.0}", ens42_7e, fusion7e),
    ):
        val_maps7 = champion_val_maps(second_fusion, deep_val7, val_images7)
        peaks = [float(((a + b) / 2).max()) for a, b in zip(val_maps42, val_maps7, strict=True)]
        threshold = float(np.quantile(peaks, 0.9))
        caught, fp, n_def_c, n_all_c, _ = decision_counts(ens, threshold)
        print(f"  {tag:26s} p90={threshold:.4f}: caught {caught}/{n_def_c} defective, "
              f"false rejects {fp}/{n_all_c - n_def_c} good", flush=True)

    del det42, fusion42, det7, fusion7, fusion7e, deep_test42, deep_test7, deep_test7e, deep_val42, deep_val7, val_maps42, val_maps7
    gc.collect()


def part_b():
    print("\n" + "=" * 115, flush=True)
    print("B) sheet_metal champion: dark (baseline) + bright sweep", flush=True)
    training, validation, captures = load_splits(DATASET, "sheet_metal", 42)
    t0 = time.perf_counter()
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    print(f"[fit sheet_metal seed 42] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test = test_maps(det, captures)
    s_dark, _ = eval_champion(fusion, deep_test, captures, polarity="dark")
    report("sheet_metal champion dark (default)", s_dark)
    for weight in (0.5, 0.75, 1.0):
        for maxf in (5, 9):
            for scale in (2.0, 1.0):
                s, _ = eval_champion(fusion, deep_test, captures, weight=weight, maxf=maxf, scale=scale, polarity="bright")
                report(f"sheet_metal champion bright w={weight} maxf={maxf} scale={scale}", s)
    del det, fusion, deep_test
    gc.collect()


def part_c():
    print("\n" + "=" * 115, flush=True)
    print("C) fabric champion: dark/both sweep", flush=True)
    training, validation, captures = load_splits(DATASET, "fabric", 42)
    t0 = time.perf_counter()
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    print(f"[fit fabric seed 42] {time.perf_counter() - t0:.0f}s", flush=True)
    deep_test = test_maps(det, captures)
    s_dark, _ = eval_champion(fusion, deep_test, captures, polarity="dark")
    report("fabric champion dark", s_dark)
    s_both, _ = eval_champion(fusion, deep_test, captures, polarity="both")
    report("fabric champion both", s_both)
    for weight in (0.5, 0.75, 1.0):
        for maxf in (5, 9):
            s, _ = eval_champion(fusion, deep_test, captures, weight=weight, maxf=maxf, polarity="both")
            report(f"fabric champion both w={weight} maxf={maxf}", s)
    del det, fusion, deep_test
    gc.collect()


def bounds(x, y, radius, width, height):
    return max(0, x - radius), max(0, y - radius), min(width, x + radius + 1), min(height, y + radius + 1)


def hotspots(scores, RADIUS=16, MAX_REGIONS=5):
    height, width = scores.shape
    remaining = scores.copy()
    regions = []
    for _ in range(MAX_REGIONS):
        y, x = map(int, np.unravel_index(remaining.argmax(), remaining.shape))
        if not np.isfinite(remaining[y, x]):
            break
        x0, y0, x1, y1 = bounds(x, y, RADIUS, width, height)
        dx0, dy0, dx1, dy1 = bounds(x, y, 2 * RADIUS, width, height)
        surround = scores[dy0:dy1, dx0:dx1]
        patch = scores[y0:y1, x0:x1]
        peak = float(scores[y, x])
        ring = np.ones(surround.shape, dtype=bool)
        ring[y0 - dy0:y1 - dy0, x0 - dx0:x1 - dx0] = False
        background = float(np.median(surround[ring])) if ring.any() else float(patch.min())
        contrast = max(0.0, peak - background)
        mass = float(np.maximum(patch - background, 0).mean())
        regions.append(dict(center=[x, y], box=[x0, y0, x1, y1],
                            features=[peak, contrast, mass], position=[x / width, y / height]))
        remaining[dy0:dy1, dx0:dx1] = -np.inf
    return regions


class NormalRegions:
    def __init__(self, examples):
        self.features = np.array([r["features"] for r in examples])
        self.positions = np.array([r["position"] for r in examples])
        self.spread = np.maximum(np.diff(np.percentile(self.features, [25, 75], axis=0), axis=0)[0], 1e-6)

    def score(self, region):
        excess = np.maximum(0, region["features"] - self.features) / self.spread
        spatial = np.square(region["position"] - self.positions).sum(axis=1)
        distances = np.sqrt(np.square(excess).mean(axis=1) + spatial)
        return float(np.partition(distances, 2)[:3].mean())


def sparc_lite_maps(test_maps_list, val_maps_list, val_images, tile_size=256):
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
        H, W = m.shape
        corr = np.zeros_like(m)
        for y in tile_origins(H, tile_size):
            for x in tile_origins(W, tile_size):
                ex, ey = min(x + tile_size, W), min(y + tile_size, H)
                corr[y:ey, x:ex] = means.get((y, x), 0.0)
        out.append(m - corr)
    return out


def region_rule_counts(by_light, scorer, threshold):
    caught_c = fp_c = 0
    n_def_c = n_all_c = 0
    for light in sorted(by_light, key=lambda n: (n != "regular", n)):
        c = f = 0
        for r in by_light[light]:
            novelty = max((scorer.score(reg) for reg in hotspots(r.anomaly_map)), default=0.0)
            if novelty >= threshold:
                if r.is_defective:
                    c += 1
                else:
                    f += 1
        if light != "regular":
            caught_c += c
            fp_c += f
            n_def_c += sum(1 for r in by_light[light] if r.is_defective)
            n_all_c += len(by_light[light])
    return caught_c, fp_c, n_def_c, n_all_c


def part_d():
    print("\n" + "=" * 115, flush=True)
    print("D) SPARC-lite: per-cell mean correction on Can champion maps", flush=True)
    training, validation, captures = load_splits(DATASET, "can", 42)
    det, fusion, val_images, deep_val = fit_champion(training, validation, 42)
    deep_test = test_maps(det, captures)
    s42, base42 = eval_champion(fusion, deep_test, captures)
    report("champion default (re-measured)", s42)
    val_maps = champion_val_maps(fusion, deep_val, val_images)
    test_maps_list = [champion_scores(fusion, deep_test[i], read_image(c.image)) for i, c in enumerate(captures)]
    corrected = sparc_lite_maps(test_maps_list, val_maps, val_images)
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
    print(f"device: {DEVICE}", flush=True)
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("\nDONE2", flush=True)


if __name__ == "__main__":
    main()
