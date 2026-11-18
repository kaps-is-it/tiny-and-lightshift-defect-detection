"""Diagnose native-fusion hotspots and test a normal-only region decision rule."""
import argparse
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label

from multiscale_experiment import CATEGORIES, add_fusion, build_reference, infer
from run_experiment import load_splits, read_image

RADIUS = 16
MAX_REGIONS = 5


def bounds(x, y, radius, width, height):
    return max(0, x-radius), max(0, y-radius), min(width, x+radius+1), min(height, y+radius+1)


def hotspots(scores):
    """Greedy non-overlapping 33x33 boxes, clipped at the image boundary."""
    height, width = scores.shape
    remaining = scores.copy()
    regions = []
    for _ in range(MAX_REGIONS):
        y, x = map(int, np.unravel_index(remaining.argmax(), remaining.shape))
        if not np.isfinite(remaining[y, x]):
            break
        x0, y0, x1, y1 = bounds(x, y, RADIUS, width, height)
        dx0, dy0, dx1, dy1 = bounds(x, y, 2*RADIUS, width, height)
        surround = scores[dy0:dy1, dx0:dx1]
        patch = scores[y0:y1, x0:x1]
        peak = float(scores[y, x])
        # Exclude the candidate itself when estimating its local background.
        ring = np.ones(surround.shape, dtype=bool)
        ring[y0-dy0:y1-dy0, x0-dx0:x1-dx0] = False
        background = float(np.median(surround[ring])) if ring.any() else float(patch.min())
        contrast = max(0., peak-background)
        mass = float(np.maximum(patch-background, 0).mean())
        regions.append(dict(center=[x, y], box=[x0, y0, x1, y1],
                            features=[peak, contrast, mass], position=[x/width, y/height]))
        remaining[dy0:dy1, dx0:dx1] = -np.inf
    return regions


class NormalRegions:
    """Compare a hotspot with similar peaks from separate good training parts."""
    def __init__(self, examples):
        self.features = np.array([r["features"] for r in examples])
        self.positions = np.array([r["position"] for r in examples])
        self.spread = np.maximum(np.diff(np.percentile(self.features, [25, 75], axis=0),
                                         axis=0)[0], 1e-6)

    def score(self, region):
        # Weaker evidence than a normal reference incurs no penalty.
        excess = np.maximum(0, region["features"] - self.features) / self.spread
        spatial = np.square(region["position"] - self.positions).sum(axis=1)
        distances = np.sqrt(np.square(excess).mean(axis=1) + spatial)
        return float(np.partition(distances, 2)[:3].mean())


def annotate(regions, mask):
    """Record any mask intersection and the stricter peak-center hit."""
    for region in regions:
        x, y = region["center"]
        x0, y0, x1, y1 = region["box"]
        region["hits"] = bool(mask[y0:y1, x0:x1].any())
        region["center_hit"] = bool(mask[y, x])


def tiny_only(mask):
    regions, count = label(mask, structure=np.ones((3, 3)))
    if not count:
        raise ValueError("Empty mask on a defective capture")
    return bool(np.bincount(regions.ravel())[1:].max() <= mask.size * 0.0001)


def atomic_json(path, content, *, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False, mode="w",
                                encoding="utf-8")
    temp = Path(stream.name)
    try:
        with stream:
            json.dump(content, stream, allow_nan=False, indent=indent)
            stream.write("\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def native_options():
    """Fixed native RGB configuration from the original controlled screen."""
    return SimpleNamespace(seed=42, treatment="native", backbone="resnet", bank_size=4096,
                           fine_bank_size=1024, batch_size=32, train_limit=96)


def fit_inspector(dataset, category, device):
    args = native_options()
    root = dataset / category
    train = sorted((root/"train"/"good").glob("*_regular.png"))
    validation = sorted((root/"validation"/"good").glob("*_regular.png"))
    if len(train) < 128 or not validation:
        raise ValueError(f"Need 96 references, 32 additional normal images and validation under {root}")
    model, selected, _ = build_reference(train, args, device)
    used = set(selected)
    reserve = [p for p in train if p not in used]
    if len(reserve) < 32:
        raise ValueError("Need 32 good training images outside the baseline reference bank")
    reserve = [reserve[i] for i in np.linspace(0, len(reserve)-1, 32, dtype=int)]
    # Use exactly the original map scaling. The region bank never sees public images.
    maxima = [dict((key, float(m.max())) for key, m in infer(model, read_image(p), args).items())
              for p in validation]
    scales = {key: max(1e-8, float(np.median([r[key] for r in maxima]))) for key in maxima[0]}

    def candidates(path):
        image = read_image(path)
        start = time.perf_counter()
        fusion = add_fusion(infer(model, image, args), scales)["fusion"]
        mapped = time.perf_counter()
        regions = hotspots(fusion)
        return regions, fusion.shape, (mapped-start)*1000, (time.perf_counter()-mapped)*1000

    examples = [region for p in reserve for region in candidates(p)[0]]
    scorer = NormalRegions(examples)

    def inspect(path):
        regions, shape, map_ms, proposal_ms = candidates(path)
        start = time.perf_counter()
        for region in regions:
            region["novelty"] = scorer.score(region)
        scoring_ms = (time.perf_counter()-start)*1000
        return dict(shape=list(shape),
                    raw=regions[0]["features"][0], region=max(r["novelty"] for r in regions),
                    map_ms=map_ms, region_ms=proposal_ms+scoring_ms, candidates=regions)

    valid = [dict(image=p.relative_to(dataset).as_posix(), **inspect(p)) for p in validation]
    thresholds = {name: float(np.quantile([r[name] for r in valid], .90)) for name in ("raw", "region")}
    metadata = dict(category=category, scales=scales, thresholds=thresholds,
                    training=[p.relative_to(dataset).as_posix() for p in selected],
                    region_training=[p.relative_to(dataset).as_posix() for p in reserve],
                    validation=valid)
    return inspect, metadata


def run(category, dataset, output, device):
    inspect, metadata = fit_inspector(dataset, category, device)
    _, _, captures = load_splits(dataset, category, 42)
    rows = []
    for capture in captures:
        row = inspect(capture.image)
        row.update(image=capture.image.relative_to(dataset).as_posix(), scene=capture.scene,
                   lighting=capture.lighting, bad=capture.mask is not None)
        if capture.mask:
            with Image.open(capture.mask) as stream:
                mask = np.asarray(stream) != 0
            if list(mask.shape) != row["shape"]:
                raise ValueError(f"Mask and capture dimensions differ for {capture.image}")
            annotate(row["candidates"], mask)
            row["tiny_only"] = tiny_only(mask)
        rows.append(row)
    atomic_json(output/f"{category}.json", dict(**metadata, captures=rows))
    print(f"{category}: {len(rows)} public captures scored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="+", choices=CATEGORIES)
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/hotspots"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    opts = parser.parse_args()
    if opts.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    for category in dict.fromkeys(opts.categories):
        run(category, opts.dataset, opts.output, opts.device)
