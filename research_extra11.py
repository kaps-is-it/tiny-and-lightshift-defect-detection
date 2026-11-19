#!/usr/bin/env python
"""Round 11: champion ensemble {42e,7e,99e}: multi-view voting (P2/P3) + INP-lite variants (N)."""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import research_extra2 as r2
from research_extra2 import (fit_champion, champion_scores, test_maps, eval_champion, report,
                             DATASET, DEVICE)
from run_experiment import load_splits
from benchmark import summarise_by_light
from evaluation import ImageResult


def champion_ensemble(training, validation, captures):
    summed = None
    val_sum = None
    masks = []
    seed_metrics = {}
    for seed in (42, 7, 99):
        t0 = time.perf_counter()
        det, fusion, val_images, deep_val = fit_champion(training, validation, seed, eps=1.0)
        print(f"[fit seed {seed}] {time.perf_counter() - t0:.0f}s", flush=True)
        val = [champion_scores(fusion, deep, im)
               for deep, im in zip(deep_val, val_images, strict=True)]
        if val_sum is None:
            val_sum = [m.copy() for m in val]
        else:
            for acc, m in zip(val_sum, val, strict=True):
                acc += m
        del deep_val, val_images, val
        summary, by_light = eval_champion(fusion, test_maps(det, captures), captures)
        report(f"seed {seed}, eps 1.0", summary)
        seed_metrics[str(seed)] = summary["au_pro_mean"]
        by_id = {(r.scene, r.lighting): r for rows in by_light.values() for r in rows}
        rows = [by_id[c.scene, c.lighting] for c in captures]
        if summed is None:
            summed = [r.anomaly_map.copy() for r in rows]
            masks = [r.mask for r in rows]
        else:
            for acc, row in zip(summed, rows, strict=True):
                acc += row.anomaly_map
        del det, fusion, rows, by_light, by_id
        gc.collect()
    for acc in summed:
        acc /= 3
    threshold = float(np.quantile([m.max() / 3.0 for m in val_sum], 0.9))
    return list(zip(summed, masks, strict=True)), threshold, seed_metrics


def part_m(training, validation, captures, out):
    aligned, threshold, seeds = champion_ensemble(training, validation, captures)
    by_scene = defaultdict(list)
    single = defaultdict(list)
    multi = defaultdict(list)
    for c, (m, mask) in zip(captures, aligned, strict=True):
        row = ImageResult(c.image.name, c.scene, c.lighting, c.mask is not None, float(m.max()), m, mask)
        single[c.lighting].append(row)
        by_scene[c.scene].append(row)
    # All six registered captures of the same physical part are available together.
    # The grouping is given by the manifest. No mask enters the map average.
    votes = {"caught": 0, "false_rejects": 0, "defective": 0, "good": 0}
    for scene, rows in by_scene.items():
        if len(rows) != 6 or {r.lighting for r in rows} != set(single):
            raise ValueError(f"Incomplete light set for {scene}")
        mean = sum(r.anomaly_map for r in rows) / len(rows)
        changed = [r for r in rows if r.lighting != "regular"]
        reject = any(r.score >= threshold for r in changed)
        for r in changed:
            votes["defective" if r.is_defective else "good"] += 1
            votes["caught" if r.is_defective else "false_rejects"] += int(reject)
        for r in rows:
            multi[r.lighting].append(ImageResult(r.image, scene, r.lighting, r.is_defective,
                                                 float(mean.max()), mean, r.mask))
        if scene == "can/bad/003":
            example = {r.lighting: r.anomaly_map for r in rows}
            example["scene_mean"] = mean
            np.savez_compressed(out / "can_003.npz", **example)
    result = {"category": "can", "split": "test_public", "seed_all_six_au_pro": seeds,
              "training_good": len(training), "validation_good": len(validation),
              "scenes": len(by_scene), "captures": len(captures), "p90_single_view_threshold": threshold,
              "changed_light_or_vote": votes}
    for name, groups in (("single_view", single), ("six_view_mean", multi)):
        metrics = summarise_by_light(groups, threshold)
        report(name, metrics)
        metrics["changed_light_au_pro"] = r2.changed_mean(metrics)
        result[name] = metrics
    return aligned, result


def inp_v1(m, q=0.2):
    normal = m[m < np.quantile(m, q)]
    return m - float(normal.mean()) if normal.size else m


def inp_v2(m, q=0.1):
    normal = m[m < np.quantile(m, q)]
    return m - float(normal.mean()) if normal.size else m


def inp_v3(m, q=0.2, alpha=0.5):
    normal = m[m < np.quantile(m, q)]
    base = float(normal.mean()) if normal.size else 0.0
    return m + alpha * (m - base)


def part_n(captures, aligned):
    print("\n" + "=" * 115, flush=True)
    print("N) INP-lite variants on champion ensemble maps (Can)", flush=True)
    variants = (
        ("control (champion ensemble)", lambda m: m),
        ("v1: m - mean(m[m < q20])", inp_v1),
        ("v2: m - mean(m[m < q10])", inp_v2),
        ("v3: m + 0.5*(m - mean(normal))", lambda m: inp_v3(m, 0.2, 0.5)),
    )
    outcomes = {}
    for tag, fn in variants:
        by_light = defaultdict(list)
        for i, c in enumerate(captures):
            m = fn(aligned[i][0])
            by_light[c.lighting].append(ImageResult(c.image.name, c.scene, c.lighting, c.mask is not None,
                                                    float(m.max()), m, aligned[i][1]))
        metrics = summarise_by_light(by_light)
        report(f"N {tag}", metrics)
        outcomes[tag] = metrics["au_pro_mean"]
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=Path("artifacts/can_multiview"))
    parser.add_argument("--skip-inp", action="store_true", help="Only run the two ensemble protocols and voting.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}", flush=True)
    training, validation, captures = load_splits(args.dataset, "can", 42)
    aligned, result = part_m(training, validation, captures, args.out)
    if not args.skip_inp:
        result["within_image_controls_all_light_au_pro"] = part_n(captures, aligned)
    output = args.out / "public_can.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved measurements to {output}", flush=True)
    print("\nDONE11", flush=True)


if __name__ == "__main__":
    main()
