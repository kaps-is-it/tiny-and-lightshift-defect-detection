"""Estimate photometric reliability of context channels on paired normal training crops."""
import numpy as np
import torch

from multiscale_detector import MultiScaleDetector
from run_experiment import INPUT_SIZE, normalize, read_image


class LightingContext(MultiScaleDetector):
    def __init__(self, device):
        super().__init__(device=device)
        self.weights = None

    @torch.inference_mode()
    def learn_weights(self, paths, seed):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[-1:1:complex(INPUT_SIZE), -1:1:complex(INPUT_SIZE)]
        signal, nuisance = 0., 0.
        for path in paths:
            image = read_image(path)
            x0 = int(rng.integers(0, image.width-INPUT_SIZE+1))
            y0 = int(rng.integers(0, image.height-INPUT_SIZE+1))
            crop = np.asarray(image.crop((x0, y0, x0+INPUT_SIZE, y0+INPUT_SIZE)))
            slope = rng.uniform(-.30, .30, size=2)
            gain = rng.uniform(.7, 1.3) + slope[0]*x + slope[1]*y
            altered = (crop.astype(np.float32)/255)**rng.uniform(.75, 1.35) * gain[:, :, None]
            view = np.uint8(np.rint(255*np.clip(altered, 0, 1)))
            _, context = super()._feature_grids(normalize([crop, view]))
            clean, changed = context
            signal += clean.var(dim=(0, 1), unbiased=False)
            nuisance += (clean-changed).square().mean(dim=(0, 1))
        # Minimize nuisance*w^2 + signal*(w-1)^2 per channel, then restore RMS scale.
        weights = signal/(signal+nuisance).clamp_min(1e-8)
        self.weights = weights/weights.square().mean().sqrt().clamp_min(1e-8)

    def _feature_grids(self, batch):
        fine, context = super()._feature_grids(batch)
        if self.weights is None:
            raise RuntimeError("Estimate channel reliability before fitting the normal bank")
        return fine, context*self.weights
