#!/usr/bin/env python
"""Round 12: anomalib comparison on the Can public split.

Runs anomalib Patchcore, Padim, Stfpm on the public Can split via the anomalib
Engine (image_size set to Can's native aspect, (256, 558): native 2232x1024 is
infeasible on this GPU), then evaluates the resulting anomaly maps with the
project's own harness (local_contrast evaluation / benchmark.summarise_by_light,
threshold 0.0). Our champion for reference: ensemble {42e,7e,99e}
all6 = 67.53%, changed5 = 67.53%.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from local_contrast.benchmark import summarise_by_light
from local_contrast.evaluation import ImageResult
from local_contrast.run_experiment import load_splits, read_image

from anomalib.data import MVTecAD2
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, Stfpm

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO = EXPERIMENT_DIR.parent
RUN_OUTPUT = REPO / "artifacts" / "anomalib_testing"

# --- Workarounds for anomalib 2.7.x + pandas 3: good-sample mask_path becomes
# NaN (truthy) and anomalib's validator rejects it. Patch at both levels.
import anomalib.data.datasets.image.mvtecad2 as _m2
from anomalib.data.datasets.image.mvtecad2 import TestType as _TestType

_orig_make = _m2.make_mvtec2_dataset


def _make_fixed(root, split=None, test_type=_TestType.PUBLIC, extensions=None):
    df = _orig_make(root, split=split, test_type=test_type, extensions=extensions)
    col = df["mask_path"]
    df["mask_path"] = col.astype(object).where(col.notna(), None)
    return df


_m2.make_mvtec2_dataset = _make_fixed

import anomalib.data.validators.torch.image as _avimg

_orig_validate_path = _avimg.validate_path


def _validate_path_fixed(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return _orig_validate_path(v)


_avimg.validate_path = _validate_path_fixed

CAN_NATIVE = (2232, 1024)


def flatten_preds(preds):
    """Normalize engine.predict output into per-image records.

    Anomalib returns lists of Batch objects (dataclasses with ``keys``).
    """
    entries = []
    for batch in (preds or []):
        if batch is None:
            continue
        if isinstance(batch, dict):
            pairs = list(batch.items())
        elif hasattr(batch, "keys"):
            pairs = [(k, getattr(batch, k, None)) for k in batch.keys()]
        else:
            continue
        n = None
        for _, v in pairs:
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
                n = len(v)
                break
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] > 1:
                n = v.shape[0]
                break
        if n is None:
            for _, v in pairs:
                if isinstance(v, (list, tuple)):
                    n = len(v)
                    break
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    n = v.shape[0]
                    break
        if n is None:
            n = 1
        for i in range(n):
            rec = {}
            for k, v in pairs:
                if v is None:
                    continue
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == n:
                    rec[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == n:
                    rec[k] = v[i]
                else:
                    rec[k] = v
            entries.append(rec)
    return entries


def to_np(v):
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        return np.asarray(v.numpy())
    return np.asarray(v)


def report(tag, summary):
    lights = summary["per_light"]
    changed = [l for l in lights if l != "regular"]
    cm = float(np.mean([summary["per_light"][l]["au_pro_05"] for l in changed])) if changed else float("nan")
    cm_str = f"{cm * 100:6.2f}%" if cm == cm else "   nan"
    print(f"  {tag:38s} all6={summary['au_pro_mean'] * 100:6.2f}% changed5={cm_str} "
          f"worst={summary['au_pro_worst'] * 100:6.2f}% tiny={summary['tiny_pro_mean'] * 100:6.2f}%", flush=True)


def make_model(cls, image_size):
    kwargs = {}
    try:
        kwargs["pre_processor"] = cls.configure_pre_processor(image_size=image_size)
        print(f"  {cls.__name__}: pre_processor set to {image_size}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {cls.__name__}: no native pre_processor ({e})", flush=True)
    return cls(**kwargs)


def resize_map(m, shape):
    if m.shape == shape:
        return m
    im = PILImage.fromarray(m.astype(np.float32))
    return np.asarray(im.resize((shape[1], shape[0]), PILImage.Resampling.BILINEAR))


def run_anomalib(model_name, model, ckpt_path=None):
    print("=" * 110, flush=True)
    print(f"anomalib {model_name} on Can public (project harness)", flush=True)
    _, _, captures = load_splits(REPO / "data" / "mvtec_ad_2", "can", 42)
    dataset_root = REPO / "data" / "mvtec_ad_2"
    cap_by_path = {c.image: c for c in captures}

    def find_cap(rec):
        p = None
        for k in ("image_path", "image_paths", "path"):
            if rec.get(k):
                p = Path(str(rec[k]))
                break
        if p is None:
            return None
        cap = cap_by_path.get(p)
        if cap is not None:
            return cap
        try:
            rel = p.relative_to(dataset_root)
        except ValueError:
            rel = None
        matches = []
        for cand, cc in cap_by_path.items():
            crel = cand.relative_to(dataset_root)
            if rel is not None and crel == rel:
                matches.append(cc)
            elif str(p).endswith(str(crel)):
                matches.append(cc)
        return matches[0] if matches else None

    dm = MVTecAD2(root=dataset_root, category="can", test_type="public",
                  train_batch_size=8, eval_batch_size=16)
    dm.setup()
    engine = Engine(default_root_dir=RUN_OUTPUT)
    if ckpt_path is None:
        t0 = time.perf_counter()
        engine.fit(model=model, datamodule=dm)
        print(f"[anomalib {model_name} fit] {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        print(f"[anomalib {model_name}] reusing checkpoint: {ckpt_path}", flush=True)
    t0 = time.perf_counter()
    preds = engine.predict(model=model, datamodule=dm, return_predictions=True, ckpt_path=ckpt_path)
    print(f"[anomalib {model_name} predict] {time.perf_counter() - t0:.0f}s", flush=True)
    entries = flatten_preds(preds)
    print(f"  prediction entries: {len(entries)}", flush=True)
    if entries:
        print(f"  first entry keys: {sorted(entries[0].keys())}", flush=True)
        print("  first entry: " + ", ".join(
            f"{k}={type(v).__name__}{getattr(v, 'shape', '')}" for k, v in list(entries[0].items())[:8]), flush=True)

    by_light = defaultdict(list)
    unmatched = 0
    for rec in entries:
        cap = find_cap(rec)
        if cap is None:
            unmatched += 1
            continue
        m = None
        for k in ("anomaly_map", "anomaly"):
            if rec.get(k) is not None:
                m = to_np(rec[k])
                break
        if m is None or not hasattr(m, "ndim"):
            continue
        if m.ndim == 3:
            m = m.squeeze()
        if m.ndim != 2:
            continue
        mask = np.asarray(read_image(cap.mask).convert("L")) != 0 if cap.mask else None
        if mask is not None and mask.shape != m.shape:
            m = resize_map(m, mask.shape)
        by_light[cap.lighting].append(ImageResult(cap.image.name, cap.scene, cap.lighting, cap.mask is not None,
                                                  float(m.max()), m.astype(np.float32), mask))
    print(f"  lighting groups: {sorted(by_light)}; results: {sum(len(v) for v in by_light.values())}; unmatched: {unmatched}",
          flush=True)
    if not by_light:
        return None
    summary = summarise_by_light(by_light)
    report(f"anomalib {model_name}", summary)
    return summary


def main():
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    # All three anomalib models run at (256, 558): native aspect (2.18:1,
    # matches the 2232x1024 Can images). Native 2232x1024 is infeasible: the
    # Patchcore memory bank alone would be ~90 GB (the store holds a 219 MB
    # patch embedding per training image at native size).
    anomalib_size = (256, 558)
    run_anomalib("Patchcore", make_model(Patchcore, anomalib_size))
    run_anomalib("Padim", make_model(Padim, anomalib_size))
    run_anomalib("Stfpm", make_model(Stfpm, anomalib_size))
    print("\nDONE12", flush=True)


if __name__ == "__main__":
    main()
