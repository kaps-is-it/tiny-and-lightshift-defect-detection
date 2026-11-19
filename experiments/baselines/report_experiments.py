"""Aggregate completed runs without weighting categories by capture count."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.baselines.multiscale_experiment import CATEGORIES, MAP_VARIANTS, VARIANTS


LABELS = {"context": "Context", "fine": "Fine", "fusion": "Fusion",
          "context_local": "Context local contrast", "fusion_local": "Fusion local contrast"}


def mean_metric(groups, name):
    values = [row[name] for row in groups if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def aggregate(groups):
    return {
        **{name: mean_metric(groups, name) for name in ("au_pro_05", "tiny_pro_at_05", "image_auroc", "pixel_f1")},
        **{name: sum(row[name] for row in groups) for name in ("tp", "fp", "tn", "fn")},
        "lights": len(groups),
        "tiny_regions": sum(row.get("tiny_regions", 0) for row in groups),
    }


def review_replay(rows):
    defective = np.array([row["is_defective"] == "True" for row in rows])
    rejected = np.array([row["fusion_rejected"] == "True" for row in rows])
    defer = np.array([row["context_rejected"] != row["fine_rejected"] for row in rows])
    baseline_missed = int((defective & ~rejected).sum())
    baseline_false_rejects = int((~defective & rejected).sum())
    review_rate = float(defer.mean())
    return {
        "captures": len(rows), "defective": int(defective.sum()), "good": int((~defective).sum()),
        "referred": int(defer.sum()), "referred_good": int((defer & ~defective).sum()), "review_rate": review_rate,
        "baseline_missed": baseline_missed, "baseline_false_rejects": baseline_false_rejects,
        "auto_passed_defective": int((defective & ~rejected & ~defer).sum()),
        "auto_rejected_good": int((~defective & rejected & ~defer).sum()),
        "random_review_expected_auto_passed_defective": baseline_missed * (1.0 - review_rate),
        "random_review_expected_auto_rejected_good": baseline_false_rejects * (1.0 - review_rate),
    }


def summarize_run(report, predictions):
    lights = report["by_lighting"]
    variants = {}
    for name in VARIANTS:
        groups = [row["variants"][name] for row in lights.values()]
        variants[name] = {
            "all": aggregate(groups),
            "changed": aggregate([row["variants"][name] for light, row in lights.items() if light != "regular"]),
            "by_lighting": {light: row["variants"][name] for light, row in lights.items()},
        }
    return {
        "train_count": len(report["training_images"]), "validation_count": report["validation_count"],
        "device": report["device"], "fit_seconds": report["fit_seconds"],
        "balanced_light_median_ms": float(np.median([row["paired_map_ms"]["median"] for row in lights.values()])),
        "variants": variants,
        "fusion_disagreement_review": {
            scope: review_replay([row for row in predictions if (row["lighting"] != "regular") == shifted])
            for scope, shifted in (("regular", False), ("changed", True))
        },
    }


def read_runs(root):
    runs = {}
    for path in sorted(root.glob("*/*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        category = report["category"]
        with (path.parent / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        runs.setdefault(path.parent.parent.name, {})[category] = summarize_run(report, predictions)
    if not runs:
        raise ValueError(f"No completed run reports under {root}")
    return runs


def make_chart(results, destination):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    native = results.get("native", {})
    can = {name: cats["can"] for name, cats in results.items() if "can" in cats}
    if not native or not can:
        raise ValueError("Chart needs at least one completed native run and a Can run")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.right": False,
                         "axes.spines.top": False, "axes.labelcolor": "#253142", "text.color": "#152535"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.7), layout="constrained")
    categories = [category for category in CATEGORIES if category in native]
    colors = ("#627a93", "#d27740", "#167663")
    for offset, (variant, color) in enumerate(zip(MAP_VARIANTS, colors, strict=True)):
        values = [native[category]["variants"][variant]["changed"]["au_pro_05"] for category in categories]
        left.bar(np.arange(len(categories)) + (offset - 1) * 0.24, values, 0.22,
                 color=color, label=LABELS[variant])
    left.set_xticks(np.arange(len(categories)), [category.replace("_", " ") for category in categories],
                    rotation=30, ha="right")
    left.set_title("Localization across shifted lights", loc="left", fontweight="bold")
    left.set_ylabel("Mean AU-PRO at 5% pixel FPR")
    left.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    left.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    left.set_ylim(0, 1)

    names = {"native": "Native RGB", "full": "Full frame", "augmented": "Lighting augmentation",
             "luma": "Gamma", "retinex": "Retinex", "coreset": "Coreset", "convnext-native": "DINOv3 ConvNeXt",
             "gradient-native": "Gradient detail", "synthetic-native": "Synthetic defect head",
             "distillation-native": "Student-teacher"}
    ordered = sorted(can, key=lambda name: can[name]["variants"]["fusion"]["changed"]["au_pro_05"])
    for index, name in enumerate(ordered):
        variants = can[name]["variants"]
        bars = [variants[variant]["changed"]["au_pro_05"] for variant in MAP_VARIANTS]
        for offset, (value, color) in enumerate(zip(bars, colors, strict=True)):
            right.barh(index + (offset - 1) * 0.24, value, 0.22, color=color)
    right.set_title("Can ablations", loc="left", fontweight="bold")
    right.set_yticks(range(len(ordered)), [names.get(name, name) for name in ordered])
    right.set_xlabel("Mean shifted AU-PRO at 5% pixel FPR")
    right.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    right.set_xlim(0, 1)
    for axis, grid_axis in ((left, "y"), (right, "x")):
        axis.set_axisbelow(True)
        axis.grid(axis=grid_axis, alpha=0.25)
    fig.savefig(destination, dpi=164, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/multiscale"))
    parser.add_argument("--output", type=Path, default=Path("results/multiscale.json"))
    parser.add_argument("--chart", type=Path)
    args = parser.parse_args()
    results = read_runs(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.chart:
        args.chart.parent.mkdir(parents=True, exist_ok=True)
        make_chart(results, args.chart)
    for treatment, categories in results.items():
        for category in CATEGORIES:
            if category not in categories:
                continue
            sample = categories[category]
            rows = sample["variants"]
            message = []
            for variant in VARIANTS:
                stats = rows[variant]["changed"]
                pro, tiny = stats["au_pro_05"], stats["tiny_pro_at_05"]
                metrics = f"{pro:.1%}/{tiny:.1%}" if pro is not None and tiny is not None else str(pro)
                message.append(f"{variant} {metrics} FN {stats['fn']}/{stats['tp'] + stats['fn']} FP {stats['fp']}/{stats['tn'] + stats['fp']}")
            print(f"{treatment}/{category} | {sample['balanced_light_median_ms']:.0f} ms | " + " | ".join(message))


if __name__ == "__main__":
    main()
