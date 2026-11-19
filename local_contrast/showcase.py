"""Render measured Can outputs for the project README."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def defect_figure(path, image, components, scores, mask, threshold):
    y, x = np.where(mask)
    size = 100
    x0 = int(np.clip((x.min() + x.max() - size) // 2, 0, image.width - size))
    y0 = int(np.clip((y.min() + y.max() - size) // 2, 0, image.height - size))
    box = (x0, y0, x0 + size, y0 + size)
    crop = np.s_[y0:y0 + size, x0:x0 + size]
    fig = Figure(figsize=(13, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1])
    overview = fig.add_subplot(grid[:, 0])
    overview.imshow(image)
    overview.add_patch(Rectangle((x0, y0), size, size, fill=False, linewidth=1.8, edgecolor="#00e2dd"))
    overview.set_title("Can 003 · Shift 1")
    overview.set_axis_off()
    first = fig.add_subplot(grid[0, 1])
    first.imshow(image.crop(box))
    first.set_title("Native crop")
    second = fig.add_subplot(grid[0, 2])
    second.imshow(image.crop(box))
    second.contour(mask[crop], levels=[0.5], colors=["#00e2dd"], linewidths=1.3)
    second.set_title("Annotation")
    decision = fig.add_subplot(grid[0, 3])
    decision.set_axis_off()
    peak = float(scores.max())
    label = "REJECT" if peak >= threshold else "PASS"
    decision.text(0.04, 0.87, label, transform=decision.transAxes, fontsize=20, weight="bold", va="top")
    decision.text(0.04, 0.60,
                  f"Image peak    {peak:.3f}\nPart cutoff     {threshold:.3f}\n\n{int(mask.sum())} annotated pixels",
                  transform=decision.transAxes, fontsize=11, linespacing=1.5, va="top")
    deep, detail, final = [fig.add_subplot(grid[1, column]) for column in (1, 2, 3)]
    for ax, name, values in ((deep, "Deep distance", components["deep"]),
                             (detail, "Dark detail", components["dark"])):
        ax.imshow(values[crop], cmap="inferno", vmin=0, vmax=2.5, interpolation="nearest")
        ax.set_title(name)
    heat = final.imshow(scores[crop], cmap="inferno", vmin=0, vmax=threshold, interpolation="nearest")
    final.set_title("Final score")
    fig.colorbar(heat, ax=final, shrink=0.78)
    for ax in (first, second, deep, detail, final):
        ax.set_axis_off()
    fig.suptitle("Where the detector looked in a 100 × 100 pixel crop", fontsize=19, weight="bold")
    fig.savefig(path, dpi=150)


def light_figure(report, baseline, output):
    lights = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
    positions = np.arange(len(lights))
    fig = Figure(figsize=(9.4, 4.0), constrained_layout=True)
    ax = fig.subplots()
    for key, label, color, marker in (("native", "Native fusion", "#677888", "o"),
                                      ("position", "Position-aware", "#aa6513", "s"),
                                      ("local", "Local contrast", "#08776e", "D")):
        rows = [report["per_light"][light] if key == "local" else baseline["by_light"][light][key]
                for light in lights]
        ax.plot(positions, [100 * row["au_pro_05"] for row in rows], color=color, label=label,
                marker=marker, markersize=6, linewidth=2)
    ax.set_xticks(positions, ["Regular", "Overexposed", "Underexposed", "Shift 1", "Shift 2", "Shift 3"])
    ax.set_ylabel("AU-PRO at 5% pixel FPR (%)")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="lower center", frameon=False)
    ax.set_title("The same 27 Can scenes under each light", fontsize=16, weight="bold")
    fig.savefig(output, dpi=150)


def scene_figure(archive, dataset, output):
    """Show the same Can 003 crop under six lights, plus their mean result."""
    from PIL import Image

    public = dataset / "can" / "test_public"
    with Image.open(public / "ground_truth/bad/003_regular_mask.png") as image:
        mask = np.asarray(image) != 0
    y, x = np.where(mask)
    cy, cx = int(y.mean()), int(x.mean())
    crop = np.s_[cy - 50:cy + 50, cx - 50:cx + 50]
    lights = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
    fig = Figure(figsize=(13.8, 4.4), constrained_layout=True)
    axes = fig.subplots(2, 7)
    with np.load(archive) as maps:
        for col, light in enumerate(lights):
            with Image.open(public / f"bad/003_{light}.png") as image:
                pixels = np.asarray(image.convert("RGB"))[crop]
            axes[0, col].imshow(pixels)
            axes[0, col].set_title(light.replace("_", " ").capitalize())
            axes[1, col].imshow(maps[light][crop], cmap="inferno", vmin=0, vmax=2.5)
            if light == "regular":
                axes[0, -1].imshow(pixels)
                axes[0, -1].contour(mask[crop], levels=[0.5], colors=["#00f2de"], linewidths=1.6)
        heat = axes[1, -1].imshow(maps["scene_mean"][crop], cmap="inferno", vmin=0, vmax=2.5)
    axes[0, -1].set_title("Annotation", color="#096957", weight="bold")
    axes[1, -1].set_title("Six-view mean", color="#096957", weight="bold")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel("Input crop")
    axes[1, 0].set_ylabel("Model score")
    fig.colorbar(heat, ax=axes[1, :].tolist(), orientation="horizontal", shrink=0.35,
                 aspect=40, label="Anomaly score · same scale in every heatmap")
    fig.suptitle("One 131-pixel defect seen under six lights", fontsize=18, weight="bold")
    fig.savefig(output, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/can_local_contrast.json"))
    parser.add_argument("--baseline", type=Path, default=Path("results/can_sampling.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/local_contrast/per_light.png"))
    parser.add_argument("--scene-maps", type=Path, help="Render the registered Can 003 views saved by round 11.")
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.scene_maps:
        scene_figure(args.scene_maps, args.dataset, args.out)
    else:
        light_figure(json.loads(args.run.read_text(encoding="utf-8")),
                     json.loads(args.baseline.read_text(encoding="utf-8")), args.out)
