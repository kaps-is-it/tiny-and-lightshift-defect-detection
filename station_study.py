"""Validate matched station captures and evaluate a locked inspection rule."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from lighting_study import FEATURES, quality_features, visibility


CONDITIONS = ("nominal", "diffuse", "grazing", "polarized")
SPLITS = ("train", "calibration", "test")
FIELDS = ("part_id", "split", "condition", "label", "image", "mask", "exposure_us", "gain_db")


@dataclass(frozen=True)
class Capture:
    part_id: str
    split: str
    condition: str
    label: str
    image: Path
    mask: Path | None


def number(value, name, *, positive=False) -> float:
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid numeric setting {name}")
    return result


def read_station(root: Path) -> dict:
    station = json.loads((root / "station.json").read_text(encoding="utf-8"))
    if station["data_origin"] not in ("physical_station", "synthetic_test"):
        raise ValueError("Set data_origin to physical_station or synthetic_test")
    for field in ("study_id", "part_family", "camera_model", "lens_model", "fixture_id", "image_pipeline"):
        if not isinstance(station[field], str) or not station[field].strip():
            raise ValueError(f"A nonempty {field} is required")
    for field in ("working_distance_mm", "field_width_mm", "field_height_mm", "minimum_defect_width_mm"):
        station[field] = number(station[field], field, positive=True)
    if set(station["lighting"]) != set(CONDITIONS):
        raise ValueError(f"Lighting settings must be defined for {', '.join(CONDITIONS)}")
    for condition, settings in station["lighting"].items():
        if not isinstance(settings["description"], str) or not settings["description"].strip():
            raise ValueError(f"Describe the light configuration for {condition}")
        settings["exposure_us"] = number(settings["exposure_us"], f"{condition} exposure_us", positive=True)
        settings["gain_db"] = number(settings["gain_db"], f"{condition} gain_db")
    return station


def local_png(root: Path, value: str) -> Path:
    relative = Path(value)
    path = (root / relative).resolve()
    if (
        relative.is_absolute() or not path.is_relative_to(root)
        or path.suffix.lower() != ".png" or not path.is_file()
    ):
        raise ValueError(f"Expected a PNG path relative to the station folder, found {value!r}")
    return path


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as mask:
        if mask.format != "PNG" or mask.size != size or mask.mode not in ("L", "1"):
            raise ValueError(f"Expected a native-size, single-channel PNG mask in {path}")
        pixels = np.array(mask)
    values = np.unique(pixels)
    if len(values) != 2 or values[0] != 0 or values[1] not in (1, 255):
        raise ValueError(f"Mask must contain binary foreground and background in {path}")
    return pixels != 0


def read_captures(root: Path, station: dict) -> tuple[list[Capture], tuple[int, int]]:
    captures = []
    part_groups: dict[str, tuple[str, str]] = {}
    views: dict[str, set[str]] = {}
    images: set[Path] = set()
    digests: dict[str, str] = {}
    mask_owners: dict[Path, str] = {}
    size = None
    mode = None
    with (root / "captures.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Expected capture columns {', '.join(FIELDS)}")
        for line, row in enumerate(reader, start=2):
            part, split, condition, label = (row[field] for field in FIELDS[:4])
            if (
                not part or part.strip() != part or split not in SPLITS
                or condition not in CONDITIONS or label not in ("good", "defective")
            ):
                raise ValueError(f"Invalid capture identity on line {line}")
            if split != "test" and (condition != "nominal" or label != "good"):
                raise ValueError(f"Use verified-good nominal captures only in {split}, line {line}")
            if part_groups.setdefault(part, (split, label)) != (split, label):
                raise ValueError(f"Part {part!r} changes split or label")
            part_views = views.setdefault(part, set())
            if condition in part_views:
                raise ValueError(f"Duplicate {condition} capture for part {part!r}")
            part_views.add(condition)

            settings = station["lighting"][condition]
            exposure = number(row["exposure_us"], "exposure_us", positive=True)
            gain = number(row["gain_db"], "gain_db")
            if exposure != settings["exposure_us"] or gain != settings["gain_db"]:
                raise ValueError(f"Recorded exposure or gain differs from the locked {condition} recipe on line {line}")
            image = local_png(root, row["image"])
            if image in images:
                raise ValueError(f"Capture file reused on line {line}")
            images.add(image)
            with image.open("rb") as content:
                digest = hashlib.file_digest(content, "sha256").hexdigest()
            if digests.setdefault(digest, part) != part:
                raise ValueError(f"Identical capture bytes assigned to different parts on line {line}")
            with Image.open(image) as opened:
                if opened.format != "PNG" or opened.mode not in ("RGB", "L"):
                    raise ValueError(f"Expected an 8-bit RGB or grayscale PNG on line {line}")
                size = size or opened.size
                mode = mode or opened.mode
                if opened.size != size or opened.mode != mode:
                    raise ValueError(f"Capture dimensions or channels changed on line {line}")
                opened.verify()

            mask = local_png(root, row["mask"]) if row["mask"] else None
            if (mask is not None) != (label == "defective"):
                raise ValueError(f"Supply a registered mask for each defective capture only, line {line}")
            if mask is not None:
                if mask_owners.setdefault(mask, part) != part:
                    raise ValueError(f"Mask path reused for different parts on line {line}")
                read_mask(mask, size)
            captures.append(Capture(part, split, condition, label, image, mask))

    if size is None or set(split for split, _ in part_groups.values()) != set(SPLITS):
        raise ValueError("Train, calibration and test each need at least one independent part")
    if {label for split, label in part_groups.values() if split == "test"} != {"good", "defective"}:
        raise ValueError("Test must contain good and defective parts")
    if images.intersection(mask_owners):
        raise ValueError("A path is used as both a camera capture and a defect mask")
    for part, (split, _) in part_groups.items():
        expected = set(CONDITIONS) if split == "test" else {"nominal"}
        if views[part] != expected:
            raise ValueError(f"Missing views for {part!r}: {', '.join(sorted(expected - views[part]))}")
    return captures, size


def describe(station: dict, captures: list[Capture], size: tuple[int, int]) -> dict:
    width, height = size
    return {
        "station": station,
        "native_size_px": [width, height],
        "nominal_sampling_mm_per_pixel": [station["field_width_mm"] / width, station["field_height_mm"] / height],
        "parts_by_split": {split: len({row.part_id for row in captures if row.split == split}) for split in SPLITS},
        "test_captures": sum(row.split == "test" for row in captures),
    }


def analyze_acquisition(captures: list[Capture]) -> dict:
    observations = {}
    for row in captures:
        with Image.open(row.image) as image:
            gray = image.convert("L")
        features = quality_features(gray)
        measures = None
        if row.mask:
            with Image.open(row.mask) as mask:
                measures = visibility(gray, mask)
        observations[row.part_id, row.condition] = features, measures

    report = {}
    for condition in CONDITIONS:
        parts = [row for row in captures if row.condition == condition]
        current = [observations[row.part_id, condition] for row in parts]
        nominal = [observations[row.part_id, "nominal"] for row in parts]
        values = np.stack([features for features, _ in current])
        reference = np.stack([features for features, _ in nominal])
        contrasts = [metrics["native_luma_contrast"] for _, metrics in current if metrics is not None]
        differences = [
            shifted["native_luma_contrast"] - baseline["native_luma_contrast"]
            for (_, shifted), (_, baseline) in zip(current, nominal, strict=True) if shifted is not None
        ]
        resized = [metrics["resized_luma_contrast"] for _, metrics in current if metrics is not None]
        paired_changes = np.median(values - reference, axis=0)
        report[condition] = {
            "parts": len(parts),
            "good_parts": len(parts) - len(contrasts),
            "defective_parts": len(contrasts),
            "median_image_features": dict(zip(FEATURES, np.median(values, axis=0).tolist(), strict=True)),
            "median_paired_feature_change_vs_nominal": dict(zip(FEATURES, paired_changes.tolist(), strict=True)),
            "median_native_defect_contrast": float(np.median(contrasts)),
            "median_paired_defect_contrast_change_vs_nominal": float(np.median(differences)),
            "median_resized_defect_contrast": float(np.median(resized)),
        }
    return report


def evaluate_detector(root: Path, captures: list[Capture], backbone: str) -> dict:
    from evaluation import ImageResult, summarize
    from inspection import selected_variant
    from multiscale_experiment import build_reference, calibrate, infer
    from run_experiment import read_image
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    options = argparse.Namespace(
        dataset=root, treatment="native", backbone=backbone, seed=42,
        train_limit=0, validation_limit=0, bank_size=4096, fine_bank_size=1024, batch_size=16,
    )
    training = [row.image for row in captures if row.split == "train"]
    validation = [row.image for row in captures if row.split == "calibration"]
    detector, _, _ = build_reference(training, options, "cpu")
    scales, thresholds, _ = calibrate(detector, validation, options)
    scores = {}
    for condition in CONDITIONS:
        results = []
        for row in captures:
            if row.split != "test" or row.condition != condition:
                continue
            image = read_image(row.image)
            _, heatmap, score = selected_variant("fusion", infer(detector, image, options), scales)
            mask = read_mask(row.mask, image.size) if row.mask else None
            results.append(ImageResult(
                row.image.relative_to(root).as_posix(), row.part_id, condition,
                row.label == "defective", score, heatmap, mask,
            ))
        scores[condition] = summarize(results, thresholds["fusion"])
    return {
        "backbone": backbone, "device": "cpu", "score": "native_fusion_peak",
        "threshold": thresholds["fusion"], "fit_parts": len(training), "calibration_parts": len(validation),
        "by_condition": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check metadata, native PNGs, masks and split integrity")
    validate.add_argument("study", type=Path)
    analyze = commands.add_parser("analyze", help="Report matched test acquisition metrics and optionally score defects")
    analyze.add_argument("study", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--detector", choices=("gradient", "resnet"), help="Fit and evaluate an existing detector on CPU")
    args = parser.parse_args()
    try:
        root = args.study.resolve(strict=True)
        station = read_station(root)
        captures, size = read_captures(root, station)
        report = describe(station, captures, size)
        if args.command == "analyze":
            output = args.output.resolve()
            inputs = {(root / name).resolve() for name in ("station.json", "captures.csv")}
            inputs.update(path for row in captures for path in (row.image, row.mask) if path is not None)
            if output in inputs:
                raise ValueError("Choose a report path separate from the station inputs")
            report["acquisition_by_condition"] = analyze_acquisition([row for row in captures if row.split == "test"])
            if args.detector:
                report["detector"] = evaluate_detector(root, captures, args.detector)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Saved {args.output}")
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
