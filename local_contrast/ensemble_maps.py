"""Fit and evaluate one normal-reference bank for the Can ensemble."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from local_contrast.benchmark import summarise_by_light
from local_contrast.detail_branch import BranchFusion
from local_contrast.detector import PatchDetector
from local_contrast.evaluation import ImageResult
from local_contrast.prepare_submission import finalize_map, roundtrip_map
from local_contrast.run_experiment import predict_map, read_image, training_batches, training_views

TILE, BATCH = 256, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = Path(__file__).resolve().parents[1] / "data" / "mvtec_ad_2"


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
