"""Compare paired raw-max and local-contrast part scores on MVTec AD 2."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("raw_max", "local_contrast")
REPLICATES = 10_000
SEED = 42


def read_predictions(path: Path, lights: list[str], thresholds: np.ndarray):
    """Return scores indexed by scene, light and method, and one label per scene."""
    by_scene: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, bool] = {}
    images = set()
    required = {
        "image", "scene", "lighting", "is_defective", "score", "rejected",
        "local_contrast_score", "local_contrast_rejected",
    }
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing prediction columns in {path}")

        for line, row in enumerate(reader, start=2):
            if any(not row[field] for field in required):
                raise ValueError(f"Missing prediction value on line {line} in {path}")
            scene, light = row["scene"], row["lighting"]
            flags = (row["is_defective"], row["rejected"], row["local_contrast_rejected"])
            if any(flag not in {"True", "False"} for flag in flags):
                raise ValueError(f"Invalid boolean on line {line} in {path}")
            label, raw_rejected, local_rejected = (flag == "True" for flag in flags)
            scores = np.array([float(row["score"]), float(row["local_contrast_score"])])
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite score on line {line} in {path}")
            if tuple(scores >= thresholds) != (raw_rejected, local_rejected):
                raise ValueError(f"Saved decisions disagree with the run thresholds on line {line} in {path}")

            captures = by_scene.setdefault(scene, {})
            if light not in lights or light in captures or row["image"] in images:
                raise ValueError(f"Unexpected or duplicate capture on line {line} in {path}")
            if labels.setdefault(scene, label) != label:
                raise ValueError(f"Inconsistent label for scene {scene} in {path}")
            captures[light] = scores
            images.add(row["image"])

    for scene, captures in by_scene.items():
        missing = set(lights) - captures.keys()
        if missing:
            raise ValueError(f"Scene {scene} is missing lighting views {sorted(missing)} in {path}")

    scenes = sorted(by_scene)
    defective = np.array([labels[scene] for scene in scenes], dtype=bool)
    if not defective.any() or defective.all():
        raise ValueError(f"Both good and defective scenes are required in {path}")
    scores = np.array([[by_scene[scene][light] for light in lights] for scene in scenes])
    return scores, defective


def summarize_subset(wins: np.ndarray, rejected: np.ndarray, defective: np.ndarray) -> dict:
    light_count = rejected.shape[1]
    good_count = int((~defective).sum()) * light_count
    defective_count = int(defective.sum()) * light_count
    auc = wins.mean(axis=(0, 1, 2))
    misses = (~rejected[defective]).sum(axis=(0, 1))
    false_rejects = rejected[~defective].sum(axis=(0, 1))
    return {
        "captures": good_count + defective_count,
        "good": good_count,
        "defective": defective_count,
        **{
            method: {
                "image_auroc": float(auc[index]),
                "misses": int(misses[index]),
                "false_rejects": int(false_rejects[index]),
            }
            for index, method in enumerate(METHODS)
        },
        "image_auroc_delta": float(auc[1] - auc[0]),
        "misses_delta": int(misses[1] - misses[0]),
        "false_rejects_delta": int(false_rejects[1] - false_rejects[0]),
    }


def scene_weights(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw scenes with replacement, returning their relative multiplicities."""
    return rng.multinomial(size, np.full(size, 1 / size), size=REPLICATES) / size


def summarize_run(run: Path) -> dict:
    metadata = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    reported = (metadata["by_lighting"], metadata["local_contrast_by_lighting"])
    lights = sorted(reported[0], key=lambda light: (light != "regular", light))
    if set(reported[1]) != set(lights) or len(lights) < 2 or lights[0] != "regular":
        raise ValueError(f"Expected the same regular and shifted lights for both methods in {run}")
    thresholds = np.array([metadata["score_threshold"], metadata["local_contrast_threshold"]], dtype=float)
    if not np.isfinite(thresholds).all():
        raise ValueError(f"Non-finite thresholds in {run}")

    scores, defective = read_predictions(run / "predictions.csv", lights, thresholds)
    rejected = scores >= thresholds
    bad_scores = scores[defective][:, None, :, :]
    good_scores = scores[~defective][None, :, :, :]
    # AUROC is the fraction of bad/good score pairs in the right order. Ties get half credit.
    wins = (bad_scores > good_scores) + 0.5 * (bad_scores == good_scores)

    by_lighting = {
        light: summarize_subset(wins[:, :, index:index + 1], rejected[:, index:index + 1], defective)
        for index, light in enumerate(lights)
    }
    for method, groups in zip(METHODS, reported, strict=True):
        for light, values in groups.items():
            summary = by_lighting[light]
            if (summary["good"], summary["defective"], summary[method]["misses"], summary[method]["false_rejects"]) != (
                values["good"], values["defective"], values["fn"], values["fp"],
            ):
                raise ValueError(f"Prediction counts disagree with {method} metrics for {light} in {run}")

    rng = np.random.default_rng(SEED)
    bad_weights = scene_weights(rng, wins.shape[0])
    good_weights = scene_weights(rng, wins.shape[1])
    subsets = {}
    for name, scope in (("all", slice(None)), ("regular", slice(0, 1)), ("changed", slice(1, None))):
        scoped_wins = wins[:, :, scope]
        summary = summarize_subset(scoped_wins, rejected[:, scope], defective)
        # Reuse scene weights across all lights and both methods to preserve pairing.
        pair_deltas = (scoped_wins[..., 1] - scoped_wins[..., 0]).mean(axis=2)
        sampled_deltas = ((bad_weights @ pair_deltas) * good_weights).sum(axis=1)
        summary["image_auroc_delta_95_ci"] = np.quantile(sampled_deltas, [0.025, 0.975]).tolist()
        subsets[name] = summary

    return {
        "scene_counts": {"good": wins.shape[1], "defective": wins.shape[0]},
        "thresholds": dict(zip(METHODS, thresholds.tolist(), strict=True)),
        "by_lighting": by_lighting,
        "subsets": subsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="+")
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, help="Defaults to <runs-root>/hybrid_summary.json")
    args = parser.parse_args()
    output = args.output or args.runs_root / "hybrid_summary.json"
    try:
        categories = {
            category: summarize_run(args.runs_root / category / "tiled-paired")
            for category in args.categories
        }
        report = {
            "bootstrap": {"replicates": REPLICATES, "seed": SEED, "unit": "scene", "stratified_by": "label"},
            "categories": categories,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    for category, summary in categories.items():
        for name, scope in summary["subsets"].items():
            raw, local = (scope[method] for method in METHODS)
            low, high = scope["image_auroc_delta_95_ci"]
            print(
                f"{category} {name}: AUROC {raw['image_auroc']:.1%} -> {local['image_auroc']:.1%}, "
                f"delta {scope['image_auroc_delta'] * 100:+.1f} pp (95% CI {low * 100:+.1f}, {high * 100:+.1f}); "
                f"misses {raw['misses']} -> {local['misses']} / {scope['defective']}; "
                f"false rejects {raw['false_rejects']} -> {local['false_rejects']} / {scope['good']}"
            )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
