"""Compare fixed and recalibrated thresholds on paired MVTec AD 2 scenes."""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median


def read_predictions(
    path: Path, category: str
) -> tuple[dict[str, dict[str, float]], dict[str, bool]]:
    scores: dict[str, dict[str, float]] = {}
    labels: dict[str, bool] = {}
    required = {"image", "scene", "lighting", "is_defective", "score"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("predictions.csv is missing required columns")

        for line, row in enumerate(reader, start=2):
            if any(not row[field] for field in required):
                raise ValueError(f"Missing prediction field on line {line}")
            scene = row["scene"]
            light = row["lighting"]
            score = float(row["score"])
            label = row["is_defective"]
            if label not in {"True", "False"} or not math.isfinite(score):
                raise ValueError(f"Invalid score or label on line {line}")
            defective = label == "True"
            folder = "bad" if defective else "good"
            number = scene.rsplit("/", 1)[-1]
            expected_image = f"{category}/test_public/{folder}/{number}_{light}.png"
            if (
                not number.isdecimal()
                or scene != f"{category}/{folder}/{number}"
                or row["image"] != expected_image
            ):
                raise ValueError(f"Scene, label, and capture path disagree on line {line}")
            captures = scores.setdefault(scene, {})
            if light in captures:
                raise ValueError(f"Duplicate capture for {scene} under {light} lighting")
            captures[light] = score
            labels[scene] = defective

    return scores, labels


def calibrate(run_dir: Path, seed: int) -> dict:
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    category = metadata["category"]
    baseline = float(metadata["score_threshold"])
    if not math.isfinite(baseline):
        raise ValueError("The baseline threshold must be finite")

    scores, labels = read_predictions(run_dir / "predictions.csv", category)
    lights = set(metadata["by_lighting"])
    if "regular" not in lights or len(lights) < 2:
        raise ValueError("Expected a public MVTec AD 2 run with regular and shifted lighting")
    for scene, captures in scores.items():
        if captures.keys() != lights:
            raise ValueError(f"Lighting conditions do not match metrics.json for {scene}")

    good = sorted(scene for scene, defective in labels.items() if not defective)
    bad = sorted(scene for scene, defective in labels.items() if defective)
    if len(good) < 5 or not bad:
        raise ValueError("Need four good reference scenes, a held-out good scene, and defects")
    for light, counts in metadata["by_lighting"].items():
        if counts["good"] != len(good) or counts["defective"] != len(bad):
            raise ValueError(f"Capture counts disagree with metrics.json for {light}")

    references = random.Random(seed).sample(good, 4)
    held_out = [scene for scene in good if scene not in references]
    by_lighting = {}
    for light in ["regular", *sorted(lights - {"regular"})]:
        reference_deltas = [scores[scene][light] - scores[scene]["regular"] for scene in references]
        offsets = {
            "fixed": 0.0,
            "two_reference": median(reference_deltas[:2]),
            "four_reference": median(reference_deltas),
        }
        comparison = {"good": len(held_out), "defective": len(bad)}
        for method, offset in offsets.items():
            threshold = baseline + offset
            false_rejects = sum(scores[scene][light] >= threshold for scene in held_out)
            missed = sum(scores[scene][light] < threshold for scene in bad)
            comparison[method] = {
                "threshold": threshold,
                "offset": offset,
                "false_rejects": false_rejects,
                "false_reject_rate": false_rejects / len(held_out),
                "missed_defects": missed,
                "miss_rate": missed / len(bad),
            }
        by_lighting[light] = comparison

    return {
        "category": category,
        "mode": metadata["mode"],
        "seed": seed,
        "baseline_threshold": baseline,
        "reference_scenes": {"2": references[:2], "4": references},
        "evaluation_scenes": {"good": held_out, "defective": bad},
        "by_lighting": by_lighting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Folder with predictions.csv and metrics.json")
    parser.add_argument("--seed", type=int, default=42, help="Reference scene sampling seed (42)")
    args = parser.parse_args()

    try:
        report = calibrate(args.run_dir, args.seed)
        output = args.run_dir / "lighting_calibration.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved paired lighting calibration to {output}")


if __name__ == "__main__":
    main()
