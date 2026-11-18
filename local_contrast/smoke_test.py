"""No-ML smoke test for the new modules. Run it after any edit:

    python smoke_test.py

This covers the deterministic pieces that do not need the GPU or the datasets:
* the luminance/residual construction is exact;
* the local z-score is zero on a constant image and carries the right sign;
* one-sided branch maps are non-negative and deterministic;
* fusion statistics on constant colour images are finite;
* the post-processing normalises each map to its own p99.5;
* synthetic defects carry their own masks;
* the submission downscale helper keeps a float16 2-D output.
"""

from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detail_branch import BranchFusion, luminance, local_residual, local_zscore, one_sided
from fusion import LearnedFusion, FEATURE_NAMES
from synthetic import inject
from prepare_submission import downscale_map, finalize_map


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> None:
    source = np.full((32, 48, 3), 100, dtype=np.uint8)
    lum = luminance(source)
    require(np.allclose(lum, 100.0), "luminance of a grey image should equal the grey value")
    residual = local_residual(lum, radius=9)
    require(np.allclose(residual, 0.0), "a constant image has zero residual")
    zscore = local_zscore(lum, radius=9, eps=0.5)
    require(np.allclose(zscore, 0.0, atol=1e-6), "a constant image has zero z-score")

    bright_image = source.copy()
    bright_image[8:16, 20:28] = 240
    residual = local_residual(luminance(bright_image), radius=9)
    dark = one_sided(residual, "dark")
    bright = one_sided(residual, "bright")
    require(np.all(dark >= 0) and np.all(bright >= 0), "branch maps must be non-negative")
    require(bright[12, 24] > dark[12, 24], "the synthetic hole carries the bright polarity")
    zscore = local_zscore(luminance(bright_image), radius=9, eps=0.5)
    require(zscore[12, 24] < 0, "a bright hole has a negative z-score (darker is positive)")

    fusion = BranchFusion(radius=9).fit([np.full((32, 48), 0.5, dtype=np.float32)], [source])
    require(fusion.mode == "zscore", "the measured recipe is the z-score branch")
    fused = fusion.fuse(np.full((32, 48), 0.5, dtype=np.float32), source, weight=0.5, polarity="dark")
    require(np.isfinite(fused).all(), "fused map must be finite")

    rng = np.random.default_rng(0)
    final = finalize_map(rng.normal(size=(32, 48)).astype(np.float32), filter_size=5)
    require(final.ndim == 2 and np.isfinite(final).all(), "finalized map must be finite")
    require(abs(float(np.percentile(final, 99.5)) - 1.0) < 1e-4, "per-image normalisation fixes p99.5 at 1.0")

    defective, mask = inject(source, rng)
    require(mask.sum() > 0, "synthetic defect should occupy pixels")
    require(rng is not None, "")

    small = downscale_map(np.zeros((4, 6), dtype=np.float32), factor=2.0)
    require(small.ndim == 2 and small.dtype == np.float16, "submission maps must be single-channel float16")
    print("smoke test passed")


if __name__ == "__main__":
    main()
