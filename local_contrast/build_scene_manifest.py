"""List matched lighting captures from an MVTec AD 2 public test split."""

import argparse
import csv
from pathlib import Path


def build_manifest(dataset: Path, category: str) -> list[dict[str, str]]:
    public = dataset / category / "test_public"
    mask_dir = public / "ground_truth" / "bad"
    rows = []
    lighting_by_scene: dict[str, set[str]] = {}
    expected_masks = set()

    for label in ("good", "bad"):
        images = sorted((public / label).glob("*.png"))
        if not images:
            raise ValueError(f"No PNG images found in {public / label}")

        for image in images:
            number, _, lighting = image.stem.partition("_")
            if not number.isdecimal() or not lighting:
                raise ValueError(f"Unexpected capture filename: {image.name}")

            # Good and defective scenes use separate numbering sequences.
            scene = f"{category}/{label}/{number}"
            lighting_by_scene.setdefault(scene, set()).add(lighting)

            mask_path = ""
            if label == "bad":
                mask = mask_dir / f"{image.stem}_mask.png"
                expected_masks.add(mask)
                mask_path = mask.relative_to(dataset).as_posix()

            rows.append({
                "scene": scene,
                "lighting": lighting,
                "label": label,
                "image": image.relative_to(dataset).as_posix(),
                "mask": mask_path,
            })

    actual_masks = set(mask_dir.glob("*.png"))
    if actual_masks != expected_masks:
        raise ValueError(
            f"Mask mismatch in {mask_dir}: {len(expected_masks - actual_masks)} missing, "
            f"{len(actual_masks - expected_masks)} without a matching image"
        )

    all_conditions = set().union(*lighting_by_scene.values())
    if "regular" not in all_conditions or len(all_conditions) < 2:
        raise ValueError("Expected regular lighting and at least one shifted condition")

    for scene, conditions in lighting_by_scene.items():
        if conditions != all_conditions:
            raise ValueError(f"Scene {scene} is missing {sorted(all_conditions - conditions)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/mvtec_ad_2"))
    parser.add_argument("--category", default="can")
    parser.add_argument("--output", type=Path, default=Path("artifacts/can_manifest.csv"))
    args = parser.parse_args()

    rows = build_manifest(args.dataset, args.category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scene_count = len({row["scene"] for row in rows})
    print(f"Wrote {len(rows)} captures from {scene_count} scenes to {args.output}")


if __name__ == "__main__":
    main()
