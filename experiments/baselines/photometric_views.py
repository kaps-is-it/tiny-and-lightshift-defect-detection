"""Photometric preprocessing and simulated lighting views for RGB images."""

import math

import numpy as np
from PIL import Image, ImageFilter


PHOTOMETRIC_MODES = ("identity", "luma", "retinex")
THUMBNAIL_LONG_SIDE = 256


def _thumbnail_size(image: Image.Image) -> tuple[int, int]:
    ratio = min(1.0, THUMBNAIL_LONG_SIDE / max(image.size))
    return (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))


def _apply_gain(
    image: Image.Image, field: np.ndarray, rgb_gain: np.ndarray | None = None
) -> Image.Image:
    gain = Image.fromarray(field.astype(np.float32, copy=False))
    gain = np.asarray(gain.resize(image.size, Image.Resampling.BILINEAR))
    pixels = np.array(image, dtype=np.float32)
    pixels *= gain[..., None]
    if rgb_gain is not None:
        pixels *= rgb_gain.astype(np.float32, copy=False)
    np.rint(pixels, out=pixels)
    np.clip(pixels, 0, 255, out=pixels)
    return Image.fromarray(pixels.astype(np.uint8))


def transform_image(image: Image.Image, mode: str) -> Image.Image:
    """Return a same-size RGB view using statistics from just this image."""
    if mode not in PHOTOMETRIC_MODES:
        raise ValueError(f"Unknown photometric mode: {mode}")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    if mode == "identity":
        return rgb

    luma = rgb.resize(_thumbnail_size(rgb), Image.Resampling.BILINEAR).convert("L")
    if mode == "luma":
        # Limit the gamma adjustment so dark backgrounds and bright highlights
        # cannot drive the entire image to an extreme exposure.
        median = float(np.median(np.asarray(luma))) / 255.0
        median = min(0.95, max(0.05, median))
        gamma = min(1.2, max(0.8, math.log(0.45) / math.log(median)))
        levels = np.arange(256, dtype=np.float32) / 255
        table = np.rint(levels**gamma * 255).astype(np.uint8).tolist()
        return rgb.point(table * 3)

    # Estimate the broad illumination field on a thumbnail. Apply a mild
    # correction to the original pixels so the detail itself is never blurred.
    radius = max(2.0, min(luma.size) * 0.1)
    background = np.asarray(luma.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    center = float(np.median(background))
    gain = ((center + 12.0) / (background + 12.0)) ** 0.30
    return _apply_gain(rgb, np.clip(gain, 0.85, 1.18))


def domain_randomized(image: Image.Image, seed: int) -> Image.Image:
    """Simulate broad gradients, falloff, exposure, and color balance.

    This creates varied views of normal training images. The gain fields are
    simple image-space approximations of lighting changes.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rng = np.random.default_rng(seed % (1 << 64))
    width, height = _thumbnail_size(rgb)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    center_x, center_y = rng.uniform(-0.25, 0.25, size=2)
    slope_x, slope_y = rng.uniform(-0.12, 0.12, size=2)
    radius_squared = ((x - center_x) ** 2 + (y - center_y) ** 2) / 2
    vignette = rng.uniform(0.0, 0.18) * np.minimum(radius_squared, 1.0)
    exposure = rng.uniform(0.84, 1.18)
    gain = exposure * (1.0 + slope_x * x + slope_y * y - vignette)
    rgb_gain = rng.uniform(0.93, 1.07, size=3)
    return _apply_gain(rgb, np.clip(gain, 0.65, 1.4), rgb_gain)
