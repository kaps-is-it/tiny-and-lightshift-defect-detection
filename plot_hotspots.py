"""Show the actual regions offered to the decision model on two Can captures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from make_project_visuals import crop_box


def render(data, dataset, destination):
    missed = sorted((r for r in data["captures"] if r["bad"] and r["lighting"] != "regular"
                     and r["raw"] < data["thresholds"]["raw"]), key=lambda r: r["image"])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), width_ratios=[1.4, 1], layout="constrained")
    for i, want_hit in enumerate((True, False)):
        row = next(r for r in missed if any(c["hits"] for c in r["candidates"]) == want_hit)
        source = dataset/row["image"]
        with Image.open(source) as stream:
            image = np.asarray(stream.convert("RGB")).copy()
        mask_path = source.parent.parent/"ground_truth"/"bad"/f"{source.stem}_mask.png"
        with Image.open(mask_path) as stream:
            mask = np.asarray(stream) != 0
        # Cyan contour sits outside the annotation to keep the original defect visible.
        image[binary_dilation(mask, iterations=2) & ~mask] = [20, 221, 224]
        box = crop_box(mask)
        left, top, right, bottom = box
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(image[top:bottom, left:right], extent=(left, right, bottom, top),
                        interpolation="nearest")
        for j, candidate in enumerate(row["candidates"], 1):
            x0, y0, x1, y1 = candidate["box"]
            for axis in ax[i]:
                axis.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False,
                                         edgecolor="#ff9933", lw=1.3))
            ax[i, 0].annotate(str(j), (x0, y0), color="white", fontsize=8,
                              bbox=dict(boxstyle="square,pad=.05", fc="#563508", ec="none"))
        ax[i, 1].set_xlim(left, right)
        ax[i, 1].set_ylim(bottom, top)
        ax[i, 0].set_title(f"Can {source.stem}  ·  five proposed locations", fontsize=10)
        outcome = "reaches the annotation" if want_hit else "misses the annotation"
        ax[i, 1].set_title(f"Detail  ·  shortlist {outcome}", fontsize=10)
        for axis in ax[i]:
            axis.set_axis_off()
    fig.suptitle("The same pass decision can hide two different failures", fontsize=15, weight="bold")
    fig.supxlabel("Orange = proposed boxes in score order     Cyan = annotation outline\n"
                  "First shifted Can miss in filename order for each outcome. MVTec AD 2 · CC BY-NC-SA 4.0",
                  fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=165)
    plt.close(fig)


def recovered_sheet(data, dataset, destination):
    thresholds = data["thresholds"]
    eligible = sorted((r for r in data["captures"] if r["bad"] and r["lighting"] != "regular"
                       and r["raw"] < thresholds["raw"] and r["region"] >= thresholds["region"]),
                      key=lambda r: r["image"])
    row = next(r for r in eligible if max(r["candidates"], key=lambda c: c["novelty"])["hits"])
    candidate = max(row["candidates"], key=lambda c: c["novelty"])
    source = dataset/row["image"]
    with Image.open(source) as stream:
        image = np.asarray(stream.convert("RGB")).copy()
    with Image.open(source.parent.parent/"ground_truth"/"bad"/f"{source.stem}_mask.png") as stream:
        mask = np.asarray(stream) != 0
    x0, y0, x1, y1 = candidate["box"]
    height, width = mask.shape
    crop = (max(0, x0-60), max(0, y0-60), min(width, x1+60), min(height, y1+60))
    left, top, right, bottom = crop
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(image)
    axes[0].add_patch(Rectangle((left, top), right-left, bottom-top, fill=False,
                                edgecolor="#ff9933", lw=1.5))
    axes[1].imshow(image[top:bottom, left:right], interpolation="nearest")
    image[binary_dilation(mask, iterations=1) & ~mask] = [20, 221, 224]
    axes[2].imshow(image[top:bottom, left:right], interpolation="nearest")
    axes[2].add_patch(Rectangle((x0-left-.5, y0-top-.5), x1-x0, y1-y0, fill=False,
                                edgecolor="#ff9933", lw=2))
    for axis, title in zip(axes, ("Original capture and crop", "Original pixels", "Leading region and annotation")):
        axis.set_title(title, fontsize=11)
        axis.set_axis_off()
    raw_ratio = row["raw"]/thresholds["raw"]
    region_ratio = row["region"]/thresholds["region"]
    fig.suptitle(f"Sheet Metal {source.stem}  ·  peak rule PASS ({raw_ratio:.3f}×)  ·  "
                 f"region rule REJECT ({region_ratio:.3f}×)", weight="bold", fontsize=13)
    fig.supxlabel("Orange box = highest region score     Cyan = annotation outline     "
                  "Ratios are relative to each rule’s threshold. MVTec AD 2 · CC BY-NC-SA 4.0", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=165)
    plt.close(fig)


def position_example(dataset, directory, results, device):
    from hotspot_study import native_options
    from make_project_visuals import heat_overlay
    from multiscale_experiment import add_fusion, build_reference, infer
    from run_experiment import read_image
    from sampling_study import PositionDetector, tile_geometry

    region = json.loads((directory/"can.json").read_text(encoding="utf-8"))
    meta = json.loads((results/"can_sampling.json").read_text(encoding="utf-8"))
    capture_path = "can/test_public/bad/003_shift_1.png"
    saved = json.loads((directory/"can_sampling_captures.json").read_text(encoding="utf-8"))
    row = next(r for r in saved if r["image"] == capture_path)
    options = native_options()
    paths = [dataset/p for p in region["training"]]
    native, _, _ = build_reference(paths, options, device)
    scales = region["scales"]
    position = PositionDetector(device, scales)
    position.fit(position.batches(paths, options.batch_size))
    image = read_image(dataset/capture_path)
    position.geometry = tile_geometry(image)
    source = dataset/capture_path
    with Image.open(source.parent.parent/"ground_truth"/"bad"/f"{source.stem}_mask.png") as stream:
        mask = np.asarray(stream) != 0
    box = crop_box(mask)
    left, top, right, bottom = box
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.6), layout="constrained")
    for i, (name, model) in enumerate((("native",native),("position",position))):
        scores = add_fusion(infer(model, image, options), scales)["fusion"]
        if abs(float(scores.max())-row[name]) > .0002:
            raise ValueError(f"Regenerated {name} does not match the recorded score")
        threshold = meta["thresholds"][name]
        prediction = scores >= threshold
        preview = heat_overlay(image, scores, threshold, mask=mask)
        axes[i, 0].imshow(image.crop(box) if i == 0 else mask[top:bottom, left:right],
                          cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title("Original pixels" if i==0 else "Dataset annotation")
        axes[i, 1].imshow(preview.crop(box), interpolation="nearest")
        axes[i, 1].set_title("Original heatmap" if i==0 else "Position-aware heatmap")
        axes[i, 2].imshow(prediction[top:bottom,left:right], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
        inside = int((prediction & mask).sum())
        axes[i, 2].set_title(f"Mask · {inside}/{int(mask.sum())} defect pixels")
        for axis in axes[i]:
            axis.set_axis_off()
    fig.suptitle("Can 003, Shift 1 · the same defect under two matching rules", weight="bold")
    fig.supxlabel("Cyan outlines the annotation. White marks scores at or above the normal-validation threshold. "
                  "MVTec AD 2 · CC BY-NC-SA 4.0", fontsize=8)
    dest = results/"visuals"/"can_position_matching.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/hotspots/can.json"))
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--output", type=Path, default=Path("results/visuals/hotspot_shortlist.png"))
    parser.add_argument("--sheet-input", type=Path, default=Path("artifacts/hotspots/sheet_metal.json"))
    parser.add_argument("--sheet-output", type=Path, default=Path("results/visuals/sheet_region_decision.png"))
    parser.add_argument("--position-example", action="store_true",
                        help="Also regenerate the Can map comparison from its recorded banks")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    render(json.loads(args.input.read_text(encoding="utf-8")), args.dataset, args.output)

    recovered_sheet(json.loads(args.sheet_input.read_text(encoding="utf-8")), args.dataset, args.sheet_output)
    if args.position_example:
        position_example(args.dataset, args.input.parent, args.results, args.device)
