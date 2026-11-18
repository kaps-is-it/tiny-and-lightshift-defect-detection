"""Summarize region proposals and decisions, preserving repeated views of each scene."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from PIL import Image

from hotspot_study import atomic_json
from multiscale_experiment import CATEGORIES


def shortlist(rows):
    def hit(row, k, field):
        return any(r[field] for r in row["candidates"][:k])
    return dict(n=len(rows), box=[sum(hit(row, k, "hits") for row in rows) for k in (1, 3, 5)],
                center=[sum(hit(row, k, "center_hit") for row in rows) for k in (1, 3, 5)])


def counts(rows, name, threshold):
    return {label: sum(r["bad"] == bad and r[name] >= threshold for r in rows)
            for label, bad in (("caught", True), ("false_rejects", False))}


def ranking(rows, name, budget):
    good = np.array([r[name] for r in rows if not r["bad"]])
    bad = np.array([r[name] for r in rows if r["bad"]])
    if not len(bad) or not len(good):
        raise ValueError("Both classes required for ranking evaluation")
    ranks = rankdata(np.r_[bad, good])[:len(bad)]
    auc = (ranks.sum() - len(bad)*(len(bad)+1)/2) / (len(bad)*len(good))
    # Highest attainable sensitivity on this ROC within the budget. Crossing a
    # tied group of negatives would exceed the budget, so use strict > boundary.
    boundary = np.sort(good)[::-1][budget] if budget < len(good) else -np.inf
    return dict(auroc=float(auc), caught_at_budget=int(sum(bad > boundary)),
                budget_false_rejects=budget)


def scene_interval(rows, thresholds, bad, method="region", baseline="raw"):
    by_scene = {}
    for r in rows:
        if r["bad"] == bad:
            delta = int(r[method] >= thresholds[method]) - int(r[baseline] >= thresholds[baseline])
            by_scene[r["scene"]] = by_scene.get(r["scene"], 0) + delta
    values = np.array(list(by_scene.values()))
    rng = np.random.default_rng(42)
    draws = rng.choice(values, size=(5000, len(values))).sum(axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def good_scene_control(rows, dataset):
    """Compare own proposals with those from every good part under the same light."""
    good_by_light = {}
    for row in rows:
        if not row["bad"]:
            good_by_light.setdefault(row["lighting"], []).append(row)
    expected = 0.
    for row in rows:
        if not row["bad"]:
            continue
        image = dataset / row["image"]
        path = image.parent.parent / "ground_truth" / "bad" / f"{image.stem}_mask.png"
        with Image.open(path) as stream:
            mask = np.asarray(stream) != 0
        height, width = mask.shape
        hits = []
        for good in good_by_light[row["lighting"]]:
            gh, gw = good["shape"]
            overlaps = False
            for c in good["candidates"]:
                x0, y0, x1, y1 = c["box"]
                x0, x1 = x0*width//gw, (x1*width+gw-1)//gw
                y0, y1 = y0*height//gh, (y1*height+gh-1)//gh
                if mask[y0:y1, x0:x1].any():
                    overlaps = True
                    break
            hits.append(overlaps)
        expected += float(np.mean(hits))
    return expected


def summarize(data, dataset):
    rows = data["captures"]
    thresholds = data["thresholds"]
    shifted = [r for r in rows if r["lighting"] != "regular"]
    bad = [r for r in shifted if r["bad"]]
    methods = {n: counts(shifted, n, t) for n, t in thresholds.items()}
    budget = methods["raw"]["false_rejects"]
    by_light = {}
    for lighting in sorted({r["lighting"] for r in rows}):
        group = [r for r in rows if r["lighting"] == lighting]
        defective = [r for r in group if r["bad"]]
        by_light[lighting] = dict(shortlist=shortlist(defective),
                                  good=len(group)-len(defective),
                                  decisions={n: counts(group, n, t) for n, t in thresholds.items()},
                                  auroc={n: ranking(group, n, 0)["auroc"] for n in thresholds})

    for name in methods:
        methods[name].update(ranking(shifted, name, budget))
        caught = [r for r in bad if r[name] >= thresholds[name]]
        leading = [(r["candidates"][0] if name == "raw" else
                    max(r["candidates"], key=lambda c: c["novelty"])) for r in caught]
        methods[name]["caught_with_box_hit"] = sum(c["hits"] for c in leading)
        methods[name]["caught_with_center_hit"] = sum(c["center_hit"] for c in leading)
        methods[name]["mean_shift_auroc"] = float(np.mean([
            v["auroc"][name] for light, v in by_light.items() if light != "regular"]))
    area = []
    for row in shifted:
        pixels = sum((x1-x0)*(y1-y0) for x0, y0, x1, y1 in
                     (c["box"] for c in row["candidates"]))
        area.append(pixels / np.prod(row["shape"]))
    return dict(total=len(rows), scenes=len({r["scene"] for r in rows}),
                good=len(shifted)-len(bad), bad=len(bad), shortlist=shortlist(bad),
                expected_hits_from_good_parts=good_scene_control(shifted, dataset),
                baseline_misses=shortlist([r for r in bad if r["raw"] < thresholds["raw"]]),
                tiny_only=shortlist([r for r in bad if r["tiny_only"]]),
                other=shortlist([r for r in bad if not r["tiny_only"]]),
                median_fraction_marked=float(np.median(area)), methods=methods,
                delta_caught_ci=scene_interval(shifted, thresholds, True),
                delta_false_rejects_ci=scene_interval(shifted, thresholds, False),
                thresholds=thresholds, fusion_scales=data["scales"], by_light=by_light,
                map_ms=np.percentile([r["map_ms"] for r in shifted], [50, 95]).tolist(),
                region_ms=np.percentile([r["region_ms"] for r in shifted], [50, 95]).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/hotspots"))
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--output", type=Path, default=Path("results/hotspot_study.json"))
    parser.add_argument("categories", nargs="*", default=list(CATEGORIES))
    args = parser.parse_args()
    results = {}
    for category in args.categories:
        data = json.loads((args.input/f"{category}.json").read_text(encoding="utf-8"))
        results[category] = summarize(data, args.dataset)
        v = results[category]
        print(category, v["shortlist"], v["baseline_misses"], v["methods"])
    atomic_json(args.output, results, indent=2)
