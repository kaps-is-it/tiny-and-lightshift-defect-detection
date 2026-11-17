"""sheet_metal: does the right polarity fused on the ensemble beat either alone?"""
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from benchmark import summarise_by_light
from detail_branch import BranchFusion
from detector import PatchDetector
from evaluation import ImageResult
from run_experiment import load_splits, predict_map, read_image

CAT = "sheet_metal"
training, validation, captures = load_splits(Path("mvtec_ad_2_dataset"), CAT, 42)
device = "cuda" if torch.cuda.is_available() else "cpu"
from run_experiment import training_batches, training_views

detectors = []
for seed in (42, 7, 1234):
    det = PatchDetector(device, bank_size=4096, seed=seed)
    det.fit(training_batches(training_views(training, False, seed), 256, 32))
    detectors.append(det)

def map_for(detector, arr):
    im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return predict_map(detector, im, 256, 32)

print("scoring validation")
val_arrays = [np.asarray(read_image(p), dtype=np.float32) for p in validation]
val_ens = [np.mean([map_for(d, a) for d in detectors], axis=0) for a in val_arrays]
fusion = BranchFusion().fit(val_ens, val_arrays)

test_arrays = [np.asarray(read_image(c.image), dtype=np.float32) for c in captures]
test_ens = [np.mean([map_for(d, a) for d in detectors], axis=0) for a in test_arrays]
test_ens = [gaussian_filter(m, sigma=2.0) for m in test_ens]
test_deep = [map_for(detectors[0], a) for a in test_arrays]
val_deep = [map_for(detectors[0], a) for a in val_arrays]

masks = []
for c in captures:
    mask = None
    if c.mask:
        with Image.open(c.mask) as t:
            mask = np.asarray(t) != 0
    masks.append(mask)

def pack(maps):
    return [ImageResult(c.image.name, c.scene, c.lighting, m is not None,
                        float(a.max()), a, m)
            for c, m, a in zip(captures, masks, maps, strict=True)]

tests = {
    "control_seed42": test_deep,
    "ensemble_3seed": test_ens,
    "ensemble_bright_w0.5": [fusion.fuse(m, a, 0.5, "bright") for m, a in zip(test_ens, test_arrays)],
    "ensemble_bright_w0.75": [fusion.fuse(m, a, 0.75, "bright") for m, a in zip(test_ens, test_arrays)],
    "ensemble_bright_w1.0": [fusion.fuse(m, a, 1.0, "bright") for m, a in zip(test_ens, test_arrays)],
    "single_bright_w0.75": [fusion.fuse(m, a, 0.75, "bright") for m, a in zip(test_deep, test_arrays)],
    "single_bright_w1.0": [fusion.fuse(m, a, 1.0, "bright") for m, a in zip(test_deep, test_arrays)],
}

from collections import defaultdict
def by_light(rows):
    g = defaultdict(list)
    for r in rows:
        g[r.lighting].append(r)
    return g

for name, maps in tests.items():
    rows = pack(maps)
    s = summarise_by_light(by_light(rows))
    print(f"{name:<22} mean {s['au_pro_mean']*100:5.1f}% | worst {s['au_pro_worst']*100:5.1f}% | tiny {s['tiny_pro_mean']*100:4.1f}%")
