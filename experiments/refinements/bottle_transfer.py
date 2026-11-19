#!/usr/bin/env python
"""Round 10: champion on MVTec AD Bottle (original-dataset pipeline check)."""
from __future__ import annotations

import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from local_contrast.evaluation import ImageResult, summarize
from PIL import Image
from local_contrast.detail_branch import BranchFusion
from local_contrast.detector import PatchDetector
from local_contrast.ensemble_maps import BATCH, DEVICE, TILE, champion_scores, report
from local_contrast.run_experiment import load_splits, predict_map, read_image, training_batches, training_views


def bottle_summary(by_light):
    lights = sorted(by_light, key=lambda n: (n != "regular", n))
    per = {l: summarize(by_light[l], 0.0) for l in lights}
    au = [per[l]["au_pro_05"] for l in lights if per[l]["au_pro_05"] is not None]
    tiny = [per[l]["tiny_pro_at_05"] for l in lights if per[l]["tiny_pro_at_05"] is not None]
    return {
        "per_light": per,
        "au_pro_mean": float(np.mean(au)) if au else float("nan"),
        "au_pro_worst": float(np.min(au)) if au else float("nan"),
        "tiny_pro_mean": float(np.mean(tiny)) if tiny else float("nan"),
        "defect_percentile": float("nan"),
        "fp_agreement": float("nan"),
    }


def bottle_champion(seed, eps):
    training, validation, captures = load_splits(Path("data/mvtec_ad"), "bottle", 42)
    det = PatchDetector(DEVICE, 4096, seed)
    det.fit(training_batches(training_views(training, False, seed), TILE, BATCH))
    val_images = [np.asarray(read_image(p)) for p in validation]
    deep_val = [predict_map(det, Image.fromarray(im), TILE, BATCH) for im in val_images]
    fusion = BranchFusion(radius=25, eps=eps).fit(deep_val, val_images)
    by_light = defaultdict(list)
    for c in captures:
        im = read_image(c.image)
        deep = predict_map(det, im, TILE, BATCH)
        scores = champion_scores(fusion, deep, im)
        mask = None
        if c.mask:
            with Image.open(c.mask) as t:
                mask = np.asarray(t) != 0
        by_light[c.lighting].append(ImageResult(c.image.name, c.scene, c.lighting, c.mask is not None,
                                                float(scores.max()), scores, mask))
    return bottle_summary(by_light), by_light


def bottle_ensemble(b42e, b7):
    ens = defaultdict(list)
    for light in b42e:
        for r42, r7 in zip(b42e[light], b7[light]):
            m = (r42.anomaly_map + r7.anomaly_map) / 2
            ens[light].append(ImageResult(r42.image, r42.scene, r42.lighting, r42.is_defective,
                                          float(m.max()), m, r42.mask))
    return bottle_summary(ens), ens


def main():
    print(f"device: {DEVICE}", flush=True)
    results = {}
    for tag, seed, eps in (("bottle seed 42 eps=0.5", 42, 0.5), ("bottle seed 42 eps=1.0", 42, 1.0),
                           ("bottle seed 7 eps=1.0", 7, 1.0)):
        t0 = time.perf_counter()
        s, b = bottle_champion(seed, eps)
        print(f"[fit {tag}] {time.perf_counter() - t0:.0f}s", flush=True)
        report(tag, s)
        results[tag] = b
        del s, b
        gc.collect()
    s, _ = bottle_ensemble(results["bottle seed 42 eps=1.0"], results["bottle seed 7 eps=1.0"])
    report("bottle champion ensemble {42e, 7e}", s)
    print("\nDONE10", flush=True)


if __name__ == "__main__":
    main()
