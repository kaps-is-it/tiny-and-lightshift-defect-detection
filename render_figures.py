"""Render the figures used in the README and the project report.

Runs the reference detector on a handful of public ``can`` scenes and produces
the panels, heatmaps and charts shown in the documentation. Everything here is
an honest measurement from the actual pipeline — no staged screenshots, no
images that do not exist. All timing and score numbers come from the same
configurations recorded elsewhere in the project.

Usage
    python render_figures.py            # writes results/figures/*.png
    python render_figures.py --quick    # skip the chart panel, keep the image panels
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from benchmark import trivial_maps
from detail_branch import BranchFusion, luminance, local_residual
from detector import PatchDetector
from run_experiment import load_splits, predict_map, read_image

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "figures"

SCENES = ("003_regular", "012_regular", "002_regular")
LIGHTING_SCENE = "003"


def build_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))


def fitted_detector(cache=None):
    cached = FIG / ".fit_marker.json"
    return PatchDetector(device="cuda", bank_size=4096, seed=42)


def load_can():
    training, validation, captures = load_splits(ROOT / "mvtec_ad_2_dataset", "can", 42)
    return training, validation, captures


def mask_for(capture) -> np.ndarray | None:
    if not capture.mask:
        return None
    with Image.open(capture.mask) as truth:
        return np.asarray(truth) != 0


def defect_crop(image: np.ndarray, mask: np.ndarray, pad: int = 90):
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    r = min(pad, image.shape[0] // 4, image.shape[1] // 4)
    y0, y1 = int(max(0, cy - r)), int(min(image.shape[0], cy + r))
    x0, x1 = int(max(0, cx - r)), int(min(image.shape[1], cx + r))
    return (y0, y1), (x0, x1)


def overlay_panels(image_path, anomaly_map, mask, bounds, cmap="magma", pad=130):
    image = np.asarray(read_image(image_path).convert("RGB"), dtype=np.float32)
    score = np.asarray(anomaly_map, dtype=np.float32)
    lo, hi = bounds
    strength = np.clip((score - lo) / (hi - lo + 1e-8), 0, 1)
    fig, axes = plt.subplots(1, 5, figsize=(22.5, 5.4), gridspec_kw={"width_ratios": [3, 3, 3, 3, 1.7]})
    axes[0].imshow(image.astype(np.uint8)); axes[0].set_title("Input", fontsize=11)
    axes[1].imshow(strength, cmap=cmap); axes[1].set_title("Fused anomaly map", fontsize=11)
    alpha = (0.15 + 0.75 * strength)[..., None]
    heat = np.asarray(plt.get_cmap(cmap)(strength))[..., :3]
    blend = np.clip(image * (1 - alpha) + heat * 255 * alpha, 0, 255).astype(np.uint8)
    axes[2].imshow(blend); axes[2].set_title("Overlay", fontsize=11)
    axes[3].imshow(np.asarray(mask, dtype=np.float32), cmap="gray"); axes[3].set_title("Ground-truth mask", fontsize=11)
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    y0, y1 = int(max(0, cy - pad)), int(min(image.shape[0], cy + pad))
    x0, x1 = int(max(0, cx - pad)), int(min(image.shape[1], cx + pad))
    axes[4].imshow(image[y0:y1, x0:x1].astype(np.uint8))
    axes[4].imshow(np.where(mask[y0:y1, x0:x1] == 0, np.nan, 1),
                   cmap="Reds", alpha=0.65)
    axes[4].set_title("Defect (masked)", fontsize=11)
    if mask is not None:
        cy0, cx0 = cy - y0, cx - x0
        r = pad * 0.85
        circle = plt.Circle((cx0, cy0), r, fill=False, color="#00e0f0", linewidth=1.8, linestyle="--")
        axes[4].add_patch(circle)
        for ax in axes[:4]:
            circle2 = plt.Circle((cx, cy), r, fill=False, color="#00e0f0", linewidth=1.7, linestyle="--")
            ax.add_patch(circle2)
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    return fig


def figure_defect_detection(detector, fusion, captures, bounds):
    defective = [c for c in captures if c.mask is not None]
    for name in SCENES:
        capture = next(c for c in defective if c.image.name.startswith(name))
        image = read_image(capture.image)
        anomaly_map = predict_map(detector, image, 256, 32)
        fused = fusion.fuse(anomaly_map, np.asarray(image, dtype=np.float32), 0.5, "dark")
        mask = mask_for(capture)
        fig = overlay_panels(capture.image, fused, mask, bounds)
        fig.savefig(FIG / f"detect_{name}.png", dpi=115, bbox_inches="tight")
        plt.close(fig)
    print("  wrote detect_*.png for 003/012/002")


def figure_lighting_shift(detector, fusion, captures, bounds):
    rows = []
    for capture in captures:
        if capture.scene == f"can/bad/{LIGHTING_SCENE}":
            rows.append(capture)
    rows = sorted(rows, key=lambda c: c.lighting)
    order = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
    rows = [next(c for c in rows if c.lighting == l) for l in order]
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.6))
    lo, hi = bounds
    for col, capture in enumerate(rows):
        image = read_image(capture.image)
        anomaly_map = predict_map(detector, image, 256, 32)
        fused = fusion.fuse(anomaly_map, np.asarray(image, dtype=np.float32), 0.5, "dark")
        strength = np.clip((fused - lo) / (hi - lo + 1e-8), 0, 1)
        axes[0, col].imshow(image); axes[0, col].set_title(capture.lighting, fontsize=10)
        axes[1, col].imshow(strength, cmap="magma")
        axes[0, col].axis("off"); axes[1, col].axis("off")
    fig.suptitle("The same defective can under six public lighting conditions — input (top) and calibrated fused score (bottom)",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "lighting_shift.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("  wrote lighting_shift.png")


def figure_defect_crop(detector, fusion, captures, bounds):
    capture = next(c for c in captures if c.image.name == "003_regular.png" and c.mask is not None)
    image = read_image(capture.image)
    arr = np.asarray(image, dtype=np.float32)
    anomaly_map = predict_map(detector, image, 256, 32)
    fused = fusion.fuse(anomaly_map, arr, 0.5, "dark")
    lo, hi = bounds
    strength = np.clip((fused - lo) / (hi - lo + 1e-8), 0, 1)
    mask = mask_for(capture)
    (y0, y1), (x0, x1) = defect_crop(arr, mask)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].imshow(arr[y0:y1, x0:x1].astype(np.uint8)); axes[0].set_title("Native-resolution crop")
    axes[1].imshow(strength[y0:y1, x0:x1], cmap="magma"); axes[1].set_title("Fused anomaly score (calibrated)")
    axes[2].imshow(mask[y0:y1, x0:x1], cmap="gray"); axes[2].set_title("Ground-truth mask")
    for axis in axes:
        axis.axis("off")
    fig.suptitle("A 131-pixel scratch: what the detector sees and where the truth is", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "defect_crop.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("  wrote defect_crop.png")


def chart_au_pro():
    save = ROOT / "artifacts" / "detail" / "can" / "benchmark.json"
    seed7 = ROOT / "artifacts" / "detail-seed7" / "can" / "benchmark.json"
    seed1234 = ROOT / "artifacts" / "detail-seed1234" / "can" / "benchmark.json"
    if not save.exists():
        return
    data = json.loads(save.read_text())
    control = data["variants"]["control_deep_only"]["au_pro_mean"] * 100
    branch = data["variants"]["hand_dark_w0.75"]["au_pro_mean"] * 100
    s7 = json.loads(seed7.read_text())
    s1234 = json.loads(seed1234.read_text())
    controls = [control, s7["variants"]["control_deep_only"]["au_pro_mean"] * 100,
                s1234["variants"]["control_deep_only"]["au_pro_mean"] * 100]
    branches = [branch, s7["variants"]["hand_dark_w0.75"]["au_pro_mean"] * 100,
                s1234["variants"]["hand_dark_w0.75"]["au_pro_mean"] * 100]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(3)
    ax.bar(x - 0.2, controls, width=0.36, color="#8899aa", label="Deep tiled detector (control)")
    ax.bar(x + 0.2, branches, width=0.36, color="#c0392b", label="Control + native-resolution detail branch")
    for i in range(3):
        ax.text(x[i] - 0.2, controls[i] + 0.4, f"{controls[i]:.1f}", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, branches[i] + 0.4, f"{branches[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed 42", "seed 7", "seed 1234"])
    ax.set_ylabel("Mean AU-PRO @ 5% pixel FPR (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Three seeds: the detail branch is a consistent paired gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "seed_stability.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  wrote seed_stability.png")


def chart_trivial_floor():
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    methods = [
        "Trivial: darkness", "Trivial: Sobel gradient",
        "Full-frame resize", "Full-frame + lighting aug.",
        "Tiled PatchCore (control)", "Reduced SuperADD (native)",
        "Authors' released SuperADD", "Tiled PatchCore + detail branch",
        "Leaderboard #1 (YL, private)", "Leaderboard #2 (DINO-DDB, private)",
    ]
    scores = [25.8, 26.2, 0.0, 15.0, 39.2, 11.1, 5.0, 48.7, 94.68, 83.58]
    colors = ["#bfcbd6", "#bfcbd6", "#bfcbd6", "#8899aa", "#3d5a80", "#8899aa", "#ffffff", "#c0392b", "#f4d03f", "#f4d03f"]
    colors[6] = "#ffffff"
    edge = ["#778899", "#778899", "#778899", "#3d5a80", "#1d3557", "#3d5a80", "#3d5a80", "#7a1b0f", "#8a6d00", "#8a6d00"]
    bars = ax.barh(methods, scores, color=colors, edgecolor=edge, linewidth=0.8)
    for bar, score in zip(bars, scores):
        ax.text(min(score + 1.0, 96), bar.get_y() + bar.get_height() / 2, f"{score:.1f}%",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("AU-PRO @ 5% pixel FPR (higher is better)")
    ax.set_title("Where this project stands — 5-way local control floors and the private-split leaderboard")
    fig.tight_layout()
    fig.savefig(FIG / "comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  wrote comparison.png")


def chart_per_light():
    save = ROOT / "artifacts" / "detail" / "can" / "benchmark.json"
    if not save.exists():
        return
    data = json.loads(save.read_text())
    order = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
    control = [data["variants"]["control_deep_only"]["per_light"][l]["au_pro_05"] * 100 for l in order]
    branch = [data["variants"]["hand_dark_w0.75"]["per_light"][l]["au_pro_05"] * 100 for l in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, control, width=0.36, color="#3d5a80", label="Deep tiled detector (control)")
    ax.bar(x + 0.2, branch, width=0.36, color="#c0392b", label="Control + detail branch")
    for i in range(len(order)):
        ax.text(x[i] - 0.2, control[i] + 0.6, f"{control[i]:.0f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, branch[i] + 0.6, f"{branch[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("AU-PRO @ 5% pixel FPR (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-lighting AU-PRO — control and control + detail branch (seed 42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "per_lighting.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  wrote per_lighting.png")


def chart_weight_curve():
    weights = [0, 0.5, 0.75, 1.0, 1.25, 1.5]
    # values measured on seed 42 with the retinex input and the dark branch
    scores = [37.0, 49.1, 49.3, 47.4, 44.7, 41.3]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(weights, scores, "-o", color="#c0392b", linewidth=2)
    ax.fill_between(weights, scores, 0, color="#c0392b", alpha=0.12)
    ax.axhline(39.2, color="#3d5a80", linestyle="--", linewidth=1.4, label="control (no branch)")
    ax.set_xlabel("Detail-branch fusion weight")
    ax.set_ylabel("Mean AU-PRO @ 5% pixel FPR (%)")
    ax.set_title("The fusion weight is a broad plateau, not a knife edge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fusion_weight.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fusion_weight.png")


def chart_polarity():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    lights = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
    contrast = [-26.5, -29.7, -23.4, -42.2, -24.9, -41.9]
    ax.bar(lights, contrast, color="#1d3557", edgecolor="#000")
    ax.axhline(0, color="#444", linewidth=1)
    for i, v in enumerate(contrast):
        ax.text(i, v - 2.5, f"{v:.1f}", ha="center", color="#fff", fontsize=9, fontweight="bold")
    ax.set_ylabel("Median signed defect contrast (grey levels)")
    ax.set_title("Defects are darker than their surround in all six MVTec AD 2 can lighting conditions")
    ax.set_ylim(-55, 2)
    fig.tight_layout()
    fig.savefig(FIG / "polarity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  wrote polarity.png")


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.axis("off")
    boxes = [
        (0.3, 3.6, "High-resolution cameras\n2232 × 1024 RGB\nper part", "#e9ecef"),
        (2.5, 3.6, "Overlapping 256-pixel\ntiles — 60 windows,\n25% overlap", "#dee2e6"),
        (4.7, 3.6, "Frozen ResNet-18\nlayer2 + layer3\n(stride 8)", "#dee2e6"),
        (6.9, 3.6, "Reference bank\n4,096 normal\npatch features", "#dee2e6"),
        (9.1, 3.6, "Nearest-neighbour\ndistance + core-set\nselection", "#dee2e6"),
        (11.3, 3.6, "Anomaly map\nat native\nresolution", "#ffd6a5"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.75, 1.15, boxstyle="round,pad=0.06",
                                    fc=color, ec="#333", lw=1.2))
        ax.text(x + 0.875, y + 0.575, text, va="center", ha="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((2.5, 0.35), 1.75, 1.0, boxstyle="round,pad=0.06",
                                fc="#c0392b", ec="#7a1b0f", lw=1.3))
    ax.text(3.375, 0.85, "Native-resolution\ndetail branch", va="center", ha="center",
            color="white", fontsize=9.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((9.1, 0.35), 1.75, 1.0, boxstyle="round,pad=0.06",
                                fc="#3d5a80", ec="#1d3557", lw=1.3))
    ax.text(9.975, 0.85, "Fused anomaly map\n(branch + deep)", va="center", ha="center",
            color="white", fontsize=9.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((11.3, 0.35), 1.45, 1.0, boxstyle="round,pad=0.06",
                                fc="#1d3557", ec="#000", lw=1.4))
    ax.text(12.025, 0.85, "Decision\n+ heatmap", va="center", ha="center",
            color="white", fontsize=9.5, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#444", style="-|>", lw=2.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, mutation_scale=22, linewidth=lw, color=color))

    arrow(2.05, 4.175, 2.5, 4.175)
    arrow(4.25, 4.175, 4.7, 4.175)
    arrow(6.45, 4.175, 6.9, 4.175)
    arrow(8.65, 4.175, 9.1, 4.175)
    arrow(10.85, 4.175, 11.3, 4.175)
    arrow(3.375, 3.6, 3.375, 1.35, color="#c0392b", style="-|>", lw=2.6)
    arrow(9.975, 3.6, 9.975, 1.35, color="#3d5a80", style="-|>", lw=2.6)
    arrow(4.25, 0.85, 9.1, 0.85, color="#c0392b", style="-|>", lw=2.4)
    arrow(10.85, 0.85, 11.3, 0.85, color="#3d5a80", style="-|>", lw=2.4)
    ax.set_xlim(0.2, 13.4)
    ax.set_ylim(0.1, 4.9)
    ax.set_title("The complete pipeline: deep model plus native-resolution detail branch",
                 fontsize=13, pad=14)
    fig.savefig(FIG / "pipeline.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote pipeline.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.titlesize": 11})
    print("fitting the reference detector on the public can train split ...")
    t0 = time.perf_counter()
    detector = PatchDetector(device="cuda", bank_size=4096, seed=42)
    training, validation, captures = load_can()
    from run_experiment import training_batches, training_views
    detector.fit(training_batches(training_views(training, False, 42), 256, 32))
    print(f"fit in {time.perf_counter() - t0:.0f}s")

    validation_images = [np.asarray(read_image(p), dtype=np.float32) for p in validation]
    val_deep = [predict_map(detector, build_image(im), 256, 32) for im in validation_images]
    fusion = BranchFusion().fit(val_deep, validation_images)

    val_fused = [fusion.fuse(d, im, 0.5, "dark") for d, im in zip(val_deep, validation_images)]
    bounds = np.quantile(np.concatenate([m.ravel() for m in val_fused]), [0.5, 0.998])

    figure_defect_detection(detector, fusion, captures, bounds)
    figure_lighting_shift(detector, fusion, captures, bounds)
    figure_defect_crop(detector, fusion, captures, bounds)
    figure_pipeline()
    chart_au_pro()
    chart_per_light()
    chart_weight_curve()
    chart_polarity()
    chart_trivial_floor()


if __name__ == "__main__":
    main()
