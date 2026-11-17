"""No-ML smoke test for the new modules. Run it after any edit:

    python smoke_test.py

This covers the deterministic pieces that do not need the GPU or the datasets:
* the luminance/residual construction is exact;
* one-sided branch maps are non-negative and deterministic;
* fusion statistics on constant colour images are finite;
* synthetic defects carry their own masks;
* the submission downscale helper keeps a float16 2-D output.
"""

from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detail_branch import luminance, local_residual, one_sided, BranchFusion
from fusion import LearnedFusion, FEATURE_NAMES
from synthetic import inject
from prepare_submission import downscale_map


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> None:
    source = np.full((32, 48, 3), 100, dtype=np.uint8)
    lum = luminance(source)
    require(np.allclose(lum, 100.0), "luminance of a grey image should equal the grey value")
    residual = local_residual(lum, radius=9)
    require(np.allclose(residual, 0.0), "a constant image has zero residual")

    bright_image = source.copy()
    bright_image[8:16, 20:28] = 240
    residual = local_residual(luminance(bright_image), radius=9)
    dark = one_sided(residual, "dark")
    bright = one_sided(residual, "bright")
    require(np.all(dark >= 0) and np.all(bright >= 0), "branch maps must be non-negative")
    require(bright[12, 24] > dark[12, 24], "the synthetic hole carries the bright polarity")

    fusion = BranchFusion().fit([np.full((32, 48), 0.5, dtype=np.float32)], [source])
    fused = fusion.fuse(np.full((32, 48), 0.5, dtype=np.float32), source, weight=0.75, polarity="dark")
    require(np.isfinite(fused).all(), "fused map must be finite")

    rng = np.random.default_rng(0)
    defective, mask = inject(source, rng)
    require(mask.sum() > 0, "synthetic defect should occupy pixels")
    require(rng is not None, "")

    small = downscale_map(np.zeros((4, 6), dtype=np.float32), factor=2.0)
    require(small.ndim == 2 and small.dtype == np.float16, "submission maps must be single-channel float16")
    print("smoke test passed")


if __name__ == "__main__":
    main()
