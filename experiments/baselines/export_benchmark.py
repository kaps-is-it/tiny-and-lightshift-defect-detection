"""Write and validate a local MVTec AD 2 benchmark submission."""

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
import tifffile


# The official checker expects the same numbered range in each private split.
CATEGORY_COUNTS = {
    "can": 321,
    "fabric": 314,
    "fruit_jelly": 255,
    "rice": 277,
    "sheet_metal": 142,
    "vial": 276,
    "wallplugs": 232,
    "walnuts": 228,
}
PRIVATE_SPLITS = {"test_private": "regular", "test_private_mixed": "mixed"}
SCORE_DIR = "anomaly_images"
MASK_DIR = "anomaly_images_thresholded"


@dataclass(frozen=True)
class BenchmarkImage:
    category: str
    split: str
    source: Path
    relative: Path
    shape: tuple[int, int]


def _expected_names(category: str, split: str, extension: str) -> set[str]:
    return {
        f"{number:03d}_{PRIVATE_SPLITS[split]}{extension}"
        for number in range(CATEGORY_COUNTS[category])
    }


def _check_children(folder: Path, names: set[str], *, directories: bool = False) -> None:
    entries = {path.name: path for path in folder.iterdir()}
    missing = names - entries.keys()
    extra = entries.keys() - names
    if missing or extra:
        raise ValueError(
            f"Unexpected contents in {folder}: {len(missing)} missing, "
            f"{len(extra)} extra. Examples: missing {sorted(missing)[:3]}, "
            f"extra {sorted(extra)[:3]}"
        )
    for path in entries.values():
        correct_type = path.is_dir() if directories else path.is_file()
        if not correct_type or path.resolve() != path:
            raise ValueError(f"Expected a regular {'folder' if directories else 'file'}: {path}")


def _write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class BenchmarkExport:
    """Store original-image maps, or explicitly limit their longest side.

    Each input map must cover the entire image. Undo padding or cropping in
    the model before calling this writer. The benchmark server resizes reduced
    score maps bilinearly. Keep native size when small defects matter.
    """

    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        *,
        with_masks: bool = False,
        max_side: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.with_masks = with_masks
        self.max_side = max_side
        if self.output_root.is_relative_to(self.dataset_root):
            raise ValueError("Choose a submission folder outside the dataset")
        if max_side is not None and max_side < 1:
            raise ValueError("max_side must be positive")

    def entries(self, category: str | None = None) -> Iterator[BenchmarkImage]:
        """Yield dataset metadata for one category or all eight categories."""
        if category is not None and category not in CATEGORY_COUNTS:
            raise ValueError(f"Unknown benchmark category: {category}")
        categories = [category] if category is not None else CATEGORY_COUNTS
        for current_category in categories:
            for split in PRIVATE_SPLITS:
                folder = self.dataset_root / current_category / split
                names = _expected_names(current_category, split, ".png")
                _check_children(folder, names)
                for name in sorted(names):
                    source = folder / name
                    with Image.open(source) as image:
                        width, height = image.size
                        if image.format != "PNG":
                            raise ValueError(f"Expected PNG source image: {source}")
                    yield BenchmarkImage(
                        current_category, split, source,
                        source.relative_to(self.dataset_root), (height, width),
                    )

    def _shape(self, image: BenchmarkImage) -> tuple[int, int]:
        largest = max(image.shape)
        if self.max_side is None or largest <= self.max_side:
            return image.shape
        height, width = image.shape
        return (
            max(1, round(height * self.max_side / largest)),
            max(1, round(width * self.max_side / largest)),
        )

    def _path(self, tree: str, image: BenchmarkImage, extension: str) -> Path:
        path = self.output_root / tree / image.relative.with_suffix(extension)
        if path.resolve() != path:
            raise ValueError(f"Unexpected link in submission path: {path}")
        return path

    def write(
        self,
        image: BenchmarkImage,
        anomaly_map: np.ndarray,
        *,
        binary_mask: np.ndarray | None = None,
        resize: bool = False,
    ) -> None:
        """Write one map and optional mask.

        Set resize=True if the supplied full-image map has different dimensions.
        Scores use bilinear interpolation and binary masks use nearest neighbor.
        """
        if image.category not in CATEGORY_COUNTS or image.split not in PRIVATE_SPLITS:
            raise ValueError(f"Unsupported benchmark image: {image.relative}")
        expected_relative = Path(image.category) / image.split / image.relative.name
        if (
            image.relative != expected_relative
            or image.source != self.dataset_root / expected_relative
            or image.relative.name not in _expected_names(image.category, image.split, ".png")
        ):
            raise ValueError(f"Unexpected benchmark image path: {image.relative}")

        scores = np.asarray(anomaly_map)
        stored_shape = self._shape(image)
        if scores.ndim != 2 or not scores.size or scores.dtype.kind not in "fiu":
            raise ValueError("Anomaly map must be a nonempty 2D numeric array")
        if scores.shape not in {image.shape, stored_shape} and not resize:
            raise ValueError(
                f"Map shape {scores.shape} differs from image {image.shape} "
                f"and export {stored_shape}. Use resize=True for an aligned full-image map."
            )
        minimum, maximum = float(scores.min()), float(scores.max())
        limit = float(np.finfo(np.float16).max)
        if not -limit <= minimum <= maximum <= limit:
            raise ValueError("Anomaly scores must be finite and fit in float16")
        if self.with_masks != (binary_mask is not None):
            raise ValueError("Supply a binary mask exactly when with_masks=True")

        destination = (stored_shape[1], stored_shape[0])
        mask = None
        if binary_mask is not None:
            mask = np.asarray(binary_mask)
            if mask.shape != scores.shape:
                raise ValueError("Binary mask and anomaly map must have the same shape")
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8) * 255
            elif mask.dtype.kind not in "fiu" or not ((mask == 0) | (mask == 255)).all():
                raise ValueError("Binary mask must contain only 0 and 255, or boolean values")
            mask = Image.fromarray(mask.astype(np.uint8, copy=False))
            if mask.size != destination:
                mask = mask.resize(destination, Image.Resampling.NEAREST)

        if scores.shape != stored_shape:
            resized = Image.fromarray(scores.astype(np.float32, copy=False))
            scores = np.asarray(resized.resize(destination, Image.Resampling.BILINEAR))
        scores = scores.astype(np.float16, copy=False)
        score_path = self._path(SCORE_DIR, image, ".tiff")
        mask_path = self._path(MASK_DIR, image, ".png") if mask is not None else None

        _write_atomic(
            score_path,
            lambda path: tifffile.imwrite(
                path, scores, photometric="minisblack", compression="deflate", metadata=None
            ),
        )
        if mask is not None:
            _write_atomic(mask_path, lambda path: mask.save(path, format="PNG", compress_level=1))

    def validate(self, checker: Path | None = None) -> int:
        """Read and validate one capture at a time, with an optional official check."""
        trees = {SCORE_DIR: ".tiff"}
        if self.with_masks:
            trees[MASK_DIR] = ".png"
        _check_children(self.output_root, set(trees), directories=True)
        for tree, extension in trees.items():
            root = self.output_root / tree
            _check_children(root, set(CATEGORY_COUNTS), directories=True)
            for category in CATEGORY_COUNTS:
                _check_children(root / category, set(PRIVATE_SPLITS), directories=True)
                for split in PRIVATE_SPLITS:
                    _check_children(root / category / split, _expected_names(category, split, extension))

        count = 0
        for image in self.entries():
            expected_shape = self._shape(image)
            path = self._path(SCORE_DIR, image, ".tiff")
            with tifffile.TiffFile(path) as source:
                if len(source.pages) != 1:
                    raise ValueError(f"Expected one TIFF page: {path}")
                scores = source.asarray()
            if scores.dtype != np.float16 or scores.shape != expected_shape:
                raise ValueError(f"Expected a float16 TIFF of shape {expected_shape}: {path}")
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite score in {path}")

            if self.with_masks:
                path = self._path(MASK_DIR, image, ".png")
                with Image.open(path) as source:
                    if source.format != "PNG" or source.mode != "L":
                        raise ValueError(f"Expected an 8-bit grayscale PNG: {path}")
                    mask = np.asarray(source)
                if mask.shape != expected_shape or not ((mask == 0) | (mask == 255)).all():
                    raise ValueError(f"Expected a 0/255 binary PNG of shape {expected_shape}: {path}")
            count += 1

        if checker is not None:
            _run_checker(Path(checker).resolve(), self.output_root)
        return count

    def package(self, archive: Path, checker: Path | None = None) -> Path:
        """Validate then write one gzip archive using MVTec's parent-directory layout."""
        archive = Path(archive).resolve()
        if not archive.name.endswith(".tar.gz"):
            raise ValueError("Archive name must end in .tar.gz")
        if archive.is_relative_to(self.output_root) or archive.is_relative_to(self.dataset_root):
            raise ValueError("Put the archive outside the submission folder and the dataset")

        self.validate(checker)

        def compress(path: Path) -> None:
            with tarfile.open(path, "w:gz", compresslevel=1) as output:
                output.add(self.output_root, arcname=self.output_root.name)

        _write_atomic(archive, compress)
        return archive


def _run_checker(checker: Path, output_root: Path) -> None:
    if not checker.is_file() or checker.name != "check_and_prepare_data_for_upload.py":
        raise ValueError(f"Choose MVTec's check_and_prepare_data_for_upload.py: {checker}")
    # A separate interpreter avoids importing MVTec's generic utils module into the runner.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from check_and_prepare_data_for_upload import check_submission; "
        "check_submission(sys.argv[2])"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(checker.parent), str(output_root)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="MVTec AD 2 image root")
    parser.add_argument("output_root", type=Path, help="Existing folder of benchmark maps")
    parser.add_argument("--with-masks", action="store_true", help="Require the optional binary PNG tree")
    parser.add_argument("--max-side", type=int, help="Expected longest side for reduced-size exports")
    parser.add_argument("--archive", type=Path, help="Validate and package to this .tar.gz path")
    parser.add_argument("--checker", type=Path, help="Run the supplied MVTec checker after local validation")
    args = parser.parse_args()

    try:
        export = BenchmarkExport(
            args.dataset_root, args.output_root, with_masks=args.with_masks, max_side=args.max_side
        )
        if args.archive is None:
            count = export.validate(args.checker)
            print(f"Validated {count:,} score maps in {export.output_root}")
        else:
            archive = export.package(args.archive, args.checker)
            print(f"Wrote validated archive to {archive}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
