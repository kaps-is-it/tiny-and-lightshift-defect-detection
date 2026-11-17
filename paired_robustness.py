"""Analyze decisions on paired public MVTec AD 2 captures."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("context", "fusion")
LABELS = ("good", "defective")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42


def read_run(run: Path):
    metadata = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    lighting = metadata["by_lighting"]
    if "regular" not in lighting or len(lighting) < 2:
        raise ValueError(f"Expected regular and shifted captures in {run}")
    lights = ["regular", *sorted(lighting.keys() - {"regular"})]
    thresholds = np.array([metadata["score_thresholds"][key] for key in METHODS], dtype=float)
    if not np.isfinite(thresholds).all():
        raise ValueError(f"Non-finite thresholds in {run}")

    by_scene = {}
    labels = {}
    images = set()
    columns = {"image", "scene", "lighting", "is_defective", "paired_map_ms", *METHODS}
    columns.update(f"{method}_rejected" for method in METHODS)
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        if not columns.issubset(rows.fieldnames or []):
            raise ValueError(f"Missing prediction columns in {run}")
        for line, row in enumerate(rows, start=2):
            if any(not row[column] for column in columns):
                raise ValueError(f"Missing prediction value on line {line} in {run}")
            scene, light = row["scene"], row["lighting"]
            flags = [row["is_defective"], *(row[f"{method}_rejected"] for method in METHODS)]
            if any(flag not in {"True", "False"} for flag in flags):
                raise ValueError(f"Invalid boolean on line {line} in {run}")
            label, *rejected = [flag == "True" for flag in flags]
            scores = np.array([float(row[method]) for method in METHODS])
            ms = float(row["paired_map_ms"])
            if not np.isfinite(scores).all() or not np.isfinite(ms) or ms < 0:
                raise ValueError(f"Invalid score or timing on line {line} in {run}")
            if not np.array_equal(scores >= thresholds, rejected):
                raise ValueError(f"Prediction and fixed threshold disagree on line {line} in {run}")

            captures = by_scene.setdefault(scene, {})
            if light not in lights or light in captures or row["image"] in images:
                raise ValueError(f"Unexpected or duplicate capture on line {line} in {run}")
            if labels.setdefault(scene, label) != label:
                raise ValueError(f"Conflicting labels for {scene} in {run}")
            captures[light] = (rejected, ms)
            images.add(row["image"])

    scenes = sorted(by_scene)
    for scene in scenes:
        missing = set(lights) - by_scene[scene].keys()
        if missing:
            raise ValueError(f"Missing lighting views for {scene}: {sorted(missing)}")
    defective = np.array([labels[scene] for scene in scenes], dtype=bool)
    if not defective.any() or defective.all():
        raise ValueError(f"Expected good and defective scenes in {run}")
    rejected = np.array([[by_scene[scene][light][0] for light in lights] for scene in scenes])
    times = np.array([[by_scene[scene][light][1] for light in lights] for scene in scenes])
    errors = rejected != defective[:, None, None]
    for index, light in enumerate(lights):
        for method_index, method in enumerate(METHODS):
            expected = lighting[light]["variants"][method]
            counts = (
                int((~defective).sum()),
                int(defective.sum()),
                int(errors[~defective, index, method_index].sum()),
                int(errors[defective, index, method_index].sum()),
            )
            if counts != tuple(expected[key] for key in ("good", "defective", "fp", "fn")):
                raise ValueError(f"Metrics and predictions disagree for {light} {method} in {run}")
    return metadata, lights, thresholds, defective, rejected, times


def paired_error_rates(errors, boot_weights):
    """Compare context and fusion while keeping every view of a scene together."""
    counts = errors.sum(axis=(0, 1))
    rates = errors.mean(axis=(0, 1))
    captures = errors.shape[0] * errors.shape[1]
    scene_gaps = errors[:, :, 1].sum(axis=1) - errors[:, :, 0].sum(axis=1)
    sampled_gaps = (boot_weights @ scene_gaps) / captures
    return {
        "scenes": errors.shape[0],
        "captures": captures,
        **{
            method: {"errors": int(counts[index]), "error_rate": float(rates[index])}
            for index, method in enumerate(METHODS)
        },
        "fusion_minus_context": float((counts[1] - counts[0]) / captures),
        "fusion_minus_context_95_ci": np.quantile(sampled_gaps, [0.025, 0.975]).tolist(),
    }


def transitions(regular_error, shifted_error):
    return {
        "correct_to_correct": int((~regular_error & ~shifted_error).sum()),
        "correct_to_error": int((~regular_error & shifted_error).sum()),
        "error_to_correct": int((regular_error & ~shifted_error).sum()),
        "error_to_error": int((regular_error & shifted_error).sum()),
    }


def reinspect(regular, alternate, defective, regular_ms, alternate_ms):
    """Simulate policies that acquire the second view only when it can change the decision."""
    output = {}
    for name, inspect_again in (("or", ~regular), ("two_vote", regular)):
        rejected = np.where(inspect_again, alternate, regular)
        captured_views = len(regular) + int(inspect_again.sum())
        added_ms = np.where(inspect_again, alternate_ms, 0.0)
        output[name] = {
            "misses": int((defective & ~rejected).sum()),
            "false_rejects": int((~defective & rejected).sum()),
            "extra_captures": int(inspect_again.sum()),
            "mean_views_per_scene": captured_views / len(regular),
            "mean_added_map_compute_ms": float(added_ms.mean()),
            "mean_total_map_compute_ms": float((regular_ms + added_ms).mean()),
        }
    return output


def summarize_run(run: Path) -> dict:
    metadata, lights, thresholds, defective, rejected, times = read_run(run)
    errors = rejected != defective[:, None, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = {"good": ~defective, "defective": defective}
    bootstraps = {}
    for label, mask in groups.items():
        size = int(mask.sum())
        bootstraps[label] = rng.multinomial(size, np.full(size, 1 / size), size=BOOTSTRAP_DRAWS)

    scopes = {"changed_lights": slice(1, None), **{light: slice(i, i + 1) for i, light in enumerate(lights)}}
    paired_rates = {
        scope: {
            label: paired_error_rates(errors[mask, views], bootstraps[label])
            for label, mask in groups.items()
        }
        for scope, views in scopes.items()
    }
    shifts = {}
    for index, light in enumerate(lights[1:], start=1):
        shifts[light] = {}
        for method_index, method in enumerate(METHODS):
            shifts[light][method] = {
                "transitions": {
                    label: transitions(errors[mask, 0, method_index], errors[mask, index, method_index])
                    for label, mask in groups.items()
                },
                "reinspection": reinspect(
                    rejected[:, 0, method_index], rejected[:, index, method_index], defective,
                    times[:, 0], times[:, index],
                ),
            }

    return {
        "category": metadata["category"],
        "treatment": metadata["treatment"],
        "thresholds": dict(zip(METHODS, thresholds.tolist(), strict=True)),
        "paired_error_rates": paired_rates,
        "regular_to_shifted": shifts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing predictions.csv and metrics.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/paired_robustness.json"))
    args = parser.parse_args()

    try:
        runs = {}
        for path in args.runs:
            result = summarize_run(path)
            key = f"{result['treatment']}/{result['category']}"
            if key in runs:
                raise ValueError(f"Multiple runs for {key}")
            runs[key] = result
        report = {
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "scene",
                "stratified_by": "good or defective", "thresholds": "fixed",
            },
            "decision_errors": {"good": "false reject", "defective": "miss"},
            "reinspection_note": (
                "The alternate view is a preexisting benchmark capture. The OR policy retries regular-view "
                "passes and rejects if either view rejects. The two-vote policy retries regular-view rejects "
                "and rejects only if both views reject. Costs sum recorded map compute times for the views used "
                "and exclude camera and station overhead."
            ),
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    for key, result in runs.items():
        rates = result["paired_error_rates"]["changed_lights"]
        for label in LABELS:
            rate = rates[label]
            low, high = rate["fusion_minus_context_95_ci"]
            print(
                f"{key} shifted {label}: {rate['context']['errors']} -> {rate['fusion']['errors']} errors "
                f"across {rate['captures']} captures "
                f"({rate['fusion_minus_context'] * 100:+.1f} pp, 95% interval {low * 100:+.1f} to {high * 100:+.1f})"
            )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
