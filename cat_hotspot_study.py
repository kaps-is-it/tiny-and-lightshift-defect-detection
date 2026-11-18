"""Compare CAT and native fusion with and without spatial agreement on public Can."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from hotspot_study import annotate, atomic_json, hotspots, native_options
from multiscale_experiment import add_fusion, build_reference, infer
from report_hotspots import counts, ranking, scene_interval, shortlist
from run_cat import (CHECKPOINT_SHA256, DEFAULT_SOURCE, DEFAULT_WEIGHTS, gaussian_kernel,
                     load_cat, predict_map, released_mask)
from run_experiment import load_splits, read_image


def agreement(first, second):
    """Use the same five proposal boxes from each detector for a ten-box union."""
    seeds = hotspots(first) + hotspots(second)
    for r in seeds:
        x0, y0, x1, y1 = r["box"]
        a = float(first[y0:y1, x0:x1].max())
        b = float(second[y0:y1, x0:x1].max())
        r["agreement"] = (a*b)**.5
    ordered = []
    for r in sorted(seeds, key=lambda r: -r["agreement"]):
        x0, y0, x1, y1 = r["box"]
        # Collapse intersecting proposals from the two models.
        if all(x1 <= a or c <= x0 or y1 <= b or d <= y0 for a,b,c,d in
               (v["box"] for v in ordered)):
            ordered.append(r)
    return ordered[:5]


def run(args):
    device = torch.device(args.device)
    with args.weights.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Unexpected CAT checkpoint")
    cat, transform = load_cat(args.source, args.weights, device)
    kernel = gaussian_kernel(device)
    data = json.loads((args.input/"can.json").read_text(encoding="utf-8"))
    training, validation, captures = load_splits(args.dataset, "can", 42)
    opts = native_options()
    native, selected, _ = build_reference(training, opts, args.device)
    assert [p.relative_to(args.dataset).as_posix() for p in selected] == data["training"]

    def inspect(path):
        im = read_image(path)
        start = time.perf_counter()
        a = add_fusion(infer(native, im, opts), data["scales"])["fusion"]
        b = predict_map(cat, transform, kernel, im, device, 4)
        released = bool(released_mask(b).any())
        b = b.astype(np.float32)/255
        regions = agreement(a, b)
        elapsed = (time.perf_counter()-start)*1000
        return dict(image=path.relative_to(args.dataset).as_posix(), native=float(a.max()),
                    cat_peak=float(b.max()), independent=float((float(a.max())*float(b.max()))**.5),
                    spatial=regions[0]["agreement"], released=released,
                    candidates=regions, compute_ms=elapsed)

    valid = [inspect(p) for p in validation]
    thresholds = {k: float(np.quantile([r[k] for r in valid], .90)) for k in
                  ("native", "cat_peak", "independent", "spatial")}
    rows = []
    for c in captures:
        row = inspect(c.image)
        row.update(scene=c.scene, lighting=c.lighting, bad=c.mask is not None)
        row["either"] = row["native"] >= thresholds["native"] or row["released"]
        if c.mask:
            with Image.open(c.mask) as stream:
                annotate(row["candidates"], np.asarray(stream) != 0)
        rows.append(row)
        if len(rows) % 36 == 0:
            print(f"Verified {len(rows)} public captures", flush=True)
    atomic_json(args.input/"can_cat_agreement_captures.json", dict(validation=valid, captures=rows))
    shifted = [r for r in rows if r["lighting"] != "regular"]
    bad = [r for r in shifted if r["bad"]]
    budget = counts(shifted, "native", thresholds["native"])["false_rejects"]
    summary = {k: {**counts(shifted, k, t), **ranking(shifted, k, budget)} for k,t in thresholds.items()}
    for k in thresholds:
        summary[k]["delta_caught_ci"] = scene_interval(shifted, thresholds, True, k, "native")
        summary[k]["delta_false_rejects_ci"] = scene_interval(shifted, thresholds, False, k, "native")
    for k in ("either", "released"):
        summary[k] = counts(shifted, k, 1)
    return dict(thresholds=thresholds, summary=summary, shortlist=shortlist(bad),
                baseline_misses=shortlist([r for r in bad if r["native"]<thresholds["native"]]),
                compute_ms=np.percentile([r["compute_ms"] for r in shifted],[50,95]).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    p.add_argument("--input", type=Path, default=Path("artifacts/hotspots"))
    p.add_argument("--output", type=Path, default=Path("results/can_cat_agreement.json"))
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    atomic_json(args.output, run(args), indent=2)
