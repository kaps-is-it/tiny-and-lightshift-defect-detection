"""Replay few-shot lighting calibration with disjoint good support and evaluation scenes."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np


SUPPORT_SIZE = 4
SCALE_BOUNDS = (0.5, 2.0)


def read_run(run: Path):
    metadata = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    category = metadata["category"]
    variants = list(metadata["score_thresholds"])
    thresholds = np.array([metadata["score_thresholds"][v] for v in variants], dtype=float)
    lights = sorted(metadata["by_lighting"], key=lambda light: (light != "regular", light))
    if not variants or not np.isfinite(thresholds).all() or len(lights) < 2 or lights[0] != "regular":
        raise ValueError(f"Expected finite fitted thresholds and regular and shifted captures in {run}")

    data: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, bool] = {}
    required = {"image", "scene", "lighting", "is_defective", *variants, *(f"{v}_rejected" for v in variants)}
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("predictions.csv does not contain paired scores and fixed decisions for every variant")
        for line, row in enumerate(reader, start=2):
            if any(not row[column] for column in required) or row["is_defective"] not in {"True", "False"}:
                raise ValueError(f"Missing value or invalid label on prediction line {line}")
            scene, light = row["scene"], row["lighting"]
            defective = row["is_defective"] == "True"
            folder = "bad" if defective else "good"
            number = scene.rsplit("/", 1)[-1]
            expected = f"{category}/test_public/{folder}/{number}_{light}.png"
            captures = data.setdefault(scene, {})
            if (
                light not in lights or light in captures or not number.isdecimal()
                or row["image"] != expected or scene != f"{category}/{folder}/{number}"
            ):
                raise ValueError(f"Unexpected capture, scene or duplicate on prediction line {line}")
            scores = np.array([float(row[variant]) for variant in variants])
            decisions = [row[f"{variant}_rejected"] for variant in variants]
            if not np.isfinite(scores).all() or decisions != [str(bool(flag)) for flag in scores >= thresholds]:
                raise ValueError(f"Invalid scores or inconsistent fixed decisions on prediction line {line}")
            captures[light] = scores
            labels[scene] = defective

    for scene, captures in data.items():
        if captures.keys() != set(lights):
            raise ValueError(f"Lighting views missing from {scene}")
    good = sorted(scene for scene, defective in labels.items() if not defective)
    bad = sorted(scene for scene, defective in labels.items() if defective)
    if len(good) <= SUPPORT_SIZE or not bad:
        raise ValueError("Expected more good scenes than the support size and at least one defective scene")
    for light in lights:
        for variant in variants:
            reference = metadata["by_lighting"][light]["variants"][variant]
            if (reference["good"], reference["defective"]) != (len(good), len(bad)):
                raise ValueError(f"Scene counts disagree with metrics.json for {light} and {variant}")
    good_scores = np.asarray([[data[scene][light] for light in lights] for scene in good])
    bad_scores = np.asarray([[data[scene][light] for light in lights] for scene in bad])
    return metadata, lights, variants, thresholds, good, good_scores, bad_scores


def support_splits(good_count: int) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(list(combinations(range(good_count), SUPPORT_SIZE)))
    eligible = np.ones((len(support), good_count), dtype=bool)
    eligible[np.arange(len(support))[:, None], support] = False
    return support, np.nonzero(eligible)[1].reshape(len(support), good_count - SUPPORT_SIZE)


def adapted_thresholds(references: np.ndarray, fixed: np.ndarray):
    """Map a fixed regular-light threshold into each light's raw score units."""
    regular = references[:, :, :1, :]
    offset = np.median(references - regular, axis=1)
    left, right = np.asarray(list(combinations(range(SUPPORT_SIZE), 2))).T
    source_delta = regular[:, right] - regular[:, left]
    target_delta = references[:, right] - references[:, left]
    comparable = np.broadcast_to(source_delta != 0, target_delta.shape)
    slopes = np.divide(target_delta, source_delta, out=np.zeros_like(target_delta), where=comparable)
    estimate = np.ma.median(np.ma.array(slopes, mask=~comparable), axis=1).filled(1.0)
    scale = np.clip(estimate, *SCALE_BOUNDS)
    # Robust intercept from matched pairs, written directly at the operating threshold.
    affine = np.median(references + scale[:, None] * (fixed - regular), axis=1)
    affine[:, 0] = fixed
    return fixed + offset, affine, scale, estimate != scale


def read_review_gate(directory: Path, category: str, good: list[str], lights: list[str], support: np.ndarray):
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    monitor = report["monitor"]
    threshold = float(monitor["batch_threshold"])
    if report["category"] != category or monitor["batch_size"] != SUPPORT_SIZE or not np.isfinite(threshold):
        raise ValueError("Lighting audit category, batch size or fixed threshold does not match the support replay")
    expected = {(scene, light) for scene in good for light in lights}
    scores = {}
    with (directory / "good_scene_alerts.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["scene"], row["lighting"]
            score = float(row["score"])
            if key in scores or not np.isfinite(score):
                raise ValueError("Lighting audit has duplicate captures or non-finite scores")
            scores[key] = score
    if scores.keys() != expected:
        raise ValueError("Lighting audit does not contain exactly the good scenes and lights in the run")
    values = np.array([[scores[scene, light] for light in lights] for scene in good])
    return np.median(values[support], axis=1) > threshold, threshold


def errors_for(good: np.ndarray, bad: np.ndarray, thresholds: np.ndarray):
    return good >= thresholds[:, None], bad < thresholds[:, None]


def paired_errors(fixed, candidate) -> dict:
    fixed_fp, fixed_fn = fixed
    fp, fn = candidate
    false_rejects = fp.sum(axis=(1, 2))
    missed_defects = fn.sum(axis=(1, 2))
    fp_delta = false_rejects - fixed_fp.sum(axis=(1, 2))
    fn_delta = missed_defects - fixed_fn.sum(axis=(1, 2))
    better = (fp_delta <= 0) & (fn_delta <= 0) & ((fp_delta < 0) | (fn_delta < 0))
    return {
        "mean_false_rejects": float(false_rejects.mean()),
        "mean_missed_defects": float(missed_defects.mean()),
        "mean_false_reject_delta": float(fp_delta.mean()),
        "mean_missed_defect_delta": float(fn_delta.mean()),
        "mean_good_captures_rescued": float((fixed_fp & ~fp).sum(axis=(1, 2)).mean()),
        "mean_good_captures_newly_rejected": float((~fixed_fp & fp).sum(axis=(1, 2)).mean()),
        "mean_defects_newly_caught": float((fixed_fn & ~fn).sum(axis=(1, 2)).mean()),
        "mean_defects_newly_missed": float((~fixed_fn & fn).sum(axis=(1, 2)).mean()),
        "support_sets_with_more_misses": int((fn_delta > 0).sum()),
        "support_sets_with_more_false_rejects": int((fp_delta > 0).sum()),
        "support_sets_improving_errors_without_tradeoff": int(better.sum()),
        "max_missed_defect_delta": int(fn_delta.max()),
    }


def summarize_scope(scope: slice, variants: list[str], errors: dict, scale, clipped, gate) -> dict:
    fixed_fp, fixed_fn = (array[:, :, scope] for array in errors["fixed"])
    result = {
        "good_captures_per_support_set": fixed_fp.shape[1] * fixed_fp.shape[2],
        "defective_captures_per_support_set": fixed_fn.shape[1] * fixed_fn.shape[2],
        "variants": {},
    }
    for i, variant in enumerate(variants):
        fixed = (fixed_fp[..., i], fixed_fn[..., i])
        metrics = {
            "fixed": {
                "mean_false_rejects": float(fixed[0].sum(axis=(1, 2)).mean()),
                "mean_missed_defects": float(fixed[1].sum(axis=(1, 2)).mean()),
            },
            **{
                method: paired_errors(fixed, tuple(array[:, :, scope, i] for array in errors[method]))
                for method in ("median_offset", "bounded_affine")
            },
        }
        metrics["bounded_affine"]["median_scale"] = float(np.median(scale[:, scope, i]))
        metrics["bounded_affine"]["clipped_scale_fits"] = int(clipped[:, scope, i].sum())
        if gate is not None:
            chosen = gate[:, None, scope]
            metrics["drift_review"] = {
                "mean_automatically_false_rejected_good": float((fixed[0] & ~chosen).sum(axis=(1, 2)).mean()),
                "mean_automatically_missed_defects": float((fixed[1] & ~chosen).sum(axis=(1, 2)).mean()),
            }
        result["variants"][variant] = metrics
    if gate is not None:
        flagged = gate[:, scope].sum(axis=1)
        result["review"] = {
            "triggered_support_set_light_pairs": int(flagged.sum()),
            "support_set_light_pairs": gate[:, scope].size,
            "mean_good_captures_referred": float(flagged.mean() * fixed_fp.shape[1]),
            "mean_defective_captures_referred": float(flagged.mean() * fixed_fn.shape[1]),
        }
    return result


def replay(run: Path, lighting_audit: Path | None) -> dict:
    metadata, lights, variants, thresholds, good, good_scores, bad_scores = read_run(run)
    support, held_out = support_splits(len(good))
    offset, affine, scale, clipped = adapted_thresholds(good_scores[support], thresholds)
    fixed = np.broadcast_to(thresholds, offset.shape)
    errors = {
        method: errors_for(good_scores[held_out], bad_scores[None], values)
        for method, values in (("fixed", fixed), ("median_offset", offset), ("bounded_affine", affine))
    }
    gate, gate_threshold = None, None
    if lighting_audit is not None:
        gate, gate_threshold = read_review_gate(lighting_audit, metadata["category"], good, lights, support)
    return {
        "category": metadata["category"], "treatment": metadata["treatment"],
        "good_scenes": len(good), "defective_scenes": len(bad_scores),
        "support_size": SUPPORT_SIZE, "support_sets": len(support),
        "fitted_regular_thresholds": dict(zip(variants, thresholds.tolist(), strict=True)),
        "affine_scale_bounds": SCALE_BOUNDS, "review_gate_threshold": gate_threshold,
        "changed_lights": summarize_scope(slice(1, None), variants, errors, scale, clipped, gate),
        "by_lighting": {
            light: summarize_scope(slice(i, i + 1), variants, errors, scale, clipped, gate)
            for i, light in enumerate(lights)
        },
        "limitations": [
            "The replay uses labeled good support scenes as verified references and keeps their captures out of both comparison arms.",
            "Every support scene has a regular reference capture. Support and evaluation captures share the same acquisition condition within a replay group.",
            "Support sets and lighting views overlap. Counts are descriptive sensitivity to support choice, not independent sample sizes.",
            "Median offset and positive bounded affine calibration are monotone within each light. They move the operating point without improving within-light score ordering or AUROC.",
            "The optional review policy defers the entire remaining batch when the support set exceeds the source-calibrated acquisition threshold. Referred captures have no simulated review outcome.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Multiscale run folder with metrics.json and predictions.csv")
    parser.add_argument("--lighting-audit", type=Path, help="Optional lighting_study category directory for drift-triggered review")
    parser.add_argument("--output", type=Path, help="Output JSON (defaults to support_adaptation.json in the run folder)")
    args = parser.parse_args()
    output = args.output or args.run / "support_adaptation.json"
    try:
        report = replay(args.run, args.lighting_audit)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(f"{report['category']}: replayed {report['support_sets']} verified-good support sets")
    for name, variants in report["changed_lights"]["variants"].items():
        summaries = ", ".join(
            f"{method}: {row['mean_false_rejects']:.2f} FP / {row['mean_missed_defects']:.2f} FN"
            for method, row in variants.items() if method != "drift_review"
        )
        print(f"  {name}: {summaries}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
