"""Inspect a Can image or reproduce the public Can localization experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from local_contrast.detail_branch import BranchFusion
from local_contrast.detector import PatchDetector
from local_contrast.evaluation import ImageResult, summarize
from local_contrast.prepare_submission import finalize_map, roundtrip_map
from local_contrast.run_experiment import load_splits, predict_map, read_image, training_batches, training_views


SEED = 42
TILE = 256
BATCH = 32
WEIGHT = 0.75


def score_map(fusion, deep, image):
    fused = fusion.fuse(deep, np.asarray(image), WEIGHT, "dark")
    return roundtrip_map(finalize_map(fused), 2.0)


def fit(training, validation, device):
    detector = PatchDetector(device, 4096, SEED)
    detector.fit(training_batches(training_views(training, False, SEED), TILE, BATCH))
    images = [np.asarray(read_image(path)) for path in validation]
    deep_maps = [predict_map(detector, Image.fromarray(im), TILE, BATCH) for im in images]
    fusion = BranchFusion().fit(deep_maps, images)
    peaks = [float(score_map(fusion, deep, im).max())
             for deep, im in zip(deep_maps, images, strict=True)]
    return detector, fusion, float(np.quantile(peaks, 0.9))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_preview(path, image, scores, threshold):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4.8), constrained_layout=True)
    axes = fig.subplots(1, 2)
    for ax in axes:
        ax.imshow(image)
        ax.set_axis_off()
    axes[0].set_title("Input")
    axes[1].set_title("Local contrast + deep features")
    heat = axes[1].imshow(scores, cmap="inferno", vmin=0, vmax=threshold, alpha=0.72)
    fig.colorbar(heat, ax=axes[1], shrink=0.66, label="Anomaly score")
    peak = float(scores.max())
    decision = "reject" if peak >= threshold else "pass"
    fig.suptitle(f"Can inspection   |   {decision}   |   peak {peak:.3f}   ·   threshold {threshold:.3f}")
    fig.savefig(path, dpi=150)


def public_run(args, detector, fusion, threshold, training, validation, captures):
    per_light = {}
    decisions = []
    for light in sorted({c.lighting for c in captures}):
        records = []
        for c in (item for item in captures if item.lighting == light):
            im = read_image(c.image)
            deep = predict_map(detector, im, TILE, BATCH)
            scores = score_map(fusion, deep, im)
            mask = None
            if c.mask:
                with Image.open(c.mask) as truth:
                    mask = np.asarray(truth) != 0
            peak = float(scores.max())
            records.append(ImageResult(c.image.name, c.scene, light, c.mask is not None,
                                       peak, scores, mask))
            decisions.append(dict(image=c.image.relative_to(args.dataset).as_posix(),
                                  scene=c.scene, lighting=light, defective=c.mask is not None,
                                  peak=peak, rejected=peak >= threshold))
            if c.scene == "can/bad/003" and light == "shift_1":
                from local_contrast.showcase import defect_figure
                components = fusion.components(deep, np.asarray(im))
                defect_figure(args.out / "can_detail.png", im, components, scores, mask, threshold)
                save_preview(args.out / "inspection.png", im, scores, threshold)
        per_light[light] = summarize(records, threshold)
        print(f"{light}: AU-PRO {per_light[light]['au_pro_05']:.2%}", flush=True)
    aggregates = {}
    for name, lights in (("all_six", list(per_light)),
                         ("changed_five", [key for key in per_light if key != "regular"])):
        rows = [per_light[key] for key in lights]
        averages = {key: float(np.mean([row[key] for row in rows]))
                    for key in ("au_pro_05", "tiny_pro_at_05", "image_auroc")}
        counts = {key: sum(row[key] for row in rows)
                  for key in ("images", "defective", "good", "tp", "fp", "fn", "tn")}
        aggregates[name] = {**averages, **counts}
    report = dict(category="can", split="test_public", selection="public development masks",
                  seed=SEED, training_good=len(training), validation_good=len(validation),
                  scenes=len({c.scene for c in captures}), captures=len(captures),
                  bank_size=4096, tile_pixels=TILE, weight=WEIGHT, polarity="dark",
                  detail_window=25, detail_eps=0.5, max_filter=5, image_percentile=99.5,
                  map_downscale=2, part_threshold_rule="p90 of validation map maxima",
                  part_threshold=threshold, pixel_fpr_limit=0.05, histogram_bins=4096,
                  tiny_area_ratio=0.0001, per_light=per_light, aggregates=aggregates)
    write_json(args.out / "public_can.json", report)
    write_json(args.out / "decisions.json", decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--public", action="store_true", help="Evaluate every labeled public Can capture.")
    mode.add_argument("--image", type=Path, help="Inspect one Can image without its annotation.")
    parser.add_argument("--dataset", type=Path, default=Path("data/mvtec_ad_2"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/local_contrast/inspection"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.image and not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.public:
        training, validation, captures = load_splits(args.dataset, "can", SEED)
    else:
        training, validation = [sorted((args.dataset / "can" / split / "good").glob("*.png"))
                                for split in ("train", "validation")]
        if not training or not validation:
            parser.error("Expected normal Can reference images under can/{train,validation}/good.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Fitting on {len(training)} normal images and calibrating on {len(validation)}.", flush=True)
    detector, fusion, threshold = fit(training, validation, args.device)
    if args.public:
        public_run(args, detector, fusion, threshold, training, validation, captures)
    else:
        im = read_image(args.image)
        scores = score_map(fusion, predict_map(detector, im, TILE, BATCH), im)
        peak = float(scores.max())
        decision = dict(image=args.image.as_posix(), peak=peak, threshold=threshold,
                        decision="reject" if peak >= threshold else "pass")
        np.save(args.out / "scores.npy", scores)
        save_preview(args.out / "inspection.png", im, scores, threshold)
        write_json(args.out / "decision.json", decision)
        print(json.dumps(decision, indent=2))
    print(f"Saved results in {args.out}", flush=True)


if __name__ == "__main__":
    main()
