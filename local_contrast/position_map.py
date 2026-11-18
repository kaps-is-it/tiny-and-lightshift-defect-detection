"""Position-conditioned normal reference, fitted on good tiles.

The global tiled bank compares every test patch to good patches from anywhere
in the frame. Printed text and foil are at fixed pixel positions so most of
them are always "normal-looking" somewhere — a single global bank can't tell
whether a patch is a defect or just a bit of label it hasn't memorised. This
module fits a per-tile-position, per-cell model of normal instead:

    fit(good) -> for each fixed tile origin and each 32x32 cell, the mean and
    variance of standardised ResNet-18 layer2+layer3 features over the good
    training images. score = Mahalanobis-ish departure from that cell's normal
    distribution, stretched back to native resolution.

Specialisation pattern from PaDiM/PNI, simplified to a diagonal Gaussian.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights, resnet18

from benchmark import summarise_by_light
from evaluation import ImageResult
from run_experiment import load_splits, read_image, tile_origins, normalize, INPUT_SIZE

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

_BACKBONE = None
_BACKBONE_DEVICE = None


def backbone(device=None):
    global _BACKBONE, _BACKBONE_DEVICE
    if _BACKBONE is None or str(_BACKBONE_DEVICE) != str(device):
        _BACKBONE = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        if device is not None:
            _BACKBONE = _BACKBONE.to(device)
        _BACKBONE.requires_grad_(False)
        _BACKBONE_DEVICE = device
    return _BACKBONE


def feature_grid(batch: torch.Tensor) -> torch.Tensor:
    """Concatenated layer2+layer3 grid, same as detector.PatchDetector."""
    device = batch.device
    model = backbone(device)
    batch = batch.to(memory_format=torch.channels_last, non_blocking=True)
    with torch.inference_mode():
        stem = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
        l2 = model.layer2(model.layer1(stem))
        l3 = model.layer3(l2)
        fine = F.avg_pool2d(l2, kernel_size=3, stride=1, padding=1)
        coarse = F.avg_pool2d(l3, kernel_size=3, stride=1, padding=1)
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((fine, coarse), dim=1).permute(0, 2, 3, 1)  # (B, 32, 32, C)


def tile_tensor(arr: np.ndarray, tile_size: int):
    """Yield (tensor, placement) for every fixed-origin tile of this image."""
    for y in tile_origins(arr.shape[0], tile_size):
        for x in tile_origins(arr.shape[1], tile_size):
            ex, ey = min(x + tile_size, arr.shape[1]), min(y + tile_size, arr.shape[0])
            sub = arr[y:ey, x:ex]
            ratio = INPUT_SIZE / max(ex - x, ey - y)
            tw, th = max(1, round((ex - x) * ratio)), max(1, round((ey - y) * ratio))
            left, top = (INPUT_SIZE - tw) // 2, (INPUT_SIZE - th) // 2
            r = np.asarray(Image.fromarray((np.clip(sub, 0, 1) * 255).astype(np.uint8)).resize((tw, th), Image.Resampling.BILINEAR),
                           dtype=np.float32) / 255.0
            r = np.pad(r, ((top, INPUT_SIZE - top - th), (left, INPUT_SIZE - left - tw), (0, 0)), mode="reflect")
            batch = (r - MEAN) / STD
            yield torch.from_numpy(batch.transpose(2, 0, 1)), (y, x, ex, ey)


def _batch_grids(tensors, device, batch=16):
    pieces = []
    for b in range(0, len(tensors), batch):
        chunk = torch.stack(tensors[b:b + batch]).to(device=device, memory_format=torch.channels_last, non_blocking=True)
        pieces.append(feature_grid(chunk))
    return torch.cat(pieces, dim=0)


class PositionConditionedBank:
    """Per-tile-position, per-cell diagonal Gaussian fitted on good images."""

    def __init__(self, device, tile_size=256, shrinkage=0.05, eps=1e-6):
        self.device = torch.device(device)
        self.tile_size = tile_size
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self._stats: dict[tuple, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self._gmean: torch.Tensor | None = None
        self._gstd: torch.Tensor | None = None

    def _all_tile_grids(self, paths, passes):
        for i in range(0, len(paths), 8):
            for p in paths[i:i + 8]:
                tensors = [t for t, _ in tile_tensor(np.asarray(read_image(p), dtype=np.float32) / 255.0, self.tile_size)]
                if tensors:
                    yield _batch_grids(tensors, self.device)

    @torch.inference_mode()
    def fit(self, paths):
        # ---- Pass 1: global per-channel statistics.
        gsum = None
        gsq = None
        gn = 0
        dims = None
        for grids in self._all_tile_grids(paths, 1):
            flat = grids.reshape(grids.shape[0], -1, grids.shape[-1]).double()
            s = flat.sum(dim=(0, 1))
            q = (flat ** 2).sum(dim=(0, 1))
            gsum = s if gsum is None else gsum + s
            gsq = q if gsq is None else gsq + q
            gn += flat.shape[0] * flat.shape[1]
            dims = flat.shape[-1]
        gmean = (gsum / gn).float()
        gstd = ((gsq / gn) - gmean.double() ** 2 + 1e-12).sqrt().float().clamp_min(1e-6)
        self._gmean, self._gstd = gmean, gstd

        # ---- Per-cell statistics on standardised features.
        sums: dict[tuple, torch.Tensor] = {}
        sumsq: dict[tuple, torch.Tensor] = {}
        counts: dict[tuple, int] = {}
        dims = self._gmean.numel()
        for i in range(0, len(paths), 8):
            for p in paths[i:i + 8]:
                tensors, placements = [], []
                for t, pos in tile_tensor(np.asarray(read_image(p), dtype=np.float32) / 255.0, self.tile_size):
                    tensors.append(t)
                    placements.append(pos)
                if not tensors:
                    continue
                grids = _batch_grids(tensors, self.device)
                flat = grids.reshape(grids.shape[0], -1, grids.shape[-1]).double()
                flat = (flat - gmean.double()) / gstd.double()
                for off, pos in enumerate(placements):
                    key = (pos[0], pos[1])
                    vec = flat[off]
                    if key not in sums:
                        sums[key] = vec.sum(0)
                        sumsq[key] = vec.square().sum(0)
                        counts[key] = 1
                    else:
                        sums[key] += vec.sum(0)
                        sumsq[key] += vec.square().sum(0)
                        counts[key] += 1
        global_mean = torch.zeros(dims, dtype=torch.double)
        global_var = torch.ones(dims, dtype=torch.double)
        # global standardised var is ~1 by construction; use CUDA-matching fill-in
        global_var = global_var.to(self.device)
        for pos, s in sums.items():
            n = counts[pos]
            mean = s / n
            var = (sumsq[pos] / n - mean ** 2).clamp_min(self.eps)
            shrunk = var * (1 - self.shrinkage) + self.shrinkage * global_var
            self._stats[pos] = (n, mean.float(), shrunk.float())
        print(f"fit on {len(self._stats)} tile origins, grid {32}x{32}")
        return self

    @torch.inference_mode()
    def score(self, arr: np.ndarray) -> np.ndarray:
        tensors, placements = [], []
        for t, pos in tile_tensor(arr, self.tile_size):
            tensors.append(t)
            placements.append(pos)
        if not tensors:
            raise ValueError("empty image")
        raw_grids = _batch_grids(tensors, self.device)
        grids = raw_grids.reshape(len(tensors), -1, raw_grids.shape[-1]).cpu().numpy().astype(np.float32)
        dims = grids.shape[-1]
        gmean = self._gmean.cpu().numpy()
        gstd = self._gstd.cpu().numpy()
        taper = np.maximum(np.hanning(self.tile_size), 0.1).astype(np.float32)
        win = taper[:, None] * taper[None, :]
        out = np.zeros(arr.shape[:2], np.float32)
        wgt = np.zeros_like(out)
        for i, pos in enumerate(placements):
            key, stats = (pos[0], pos[1]), self._stats.get((pos[0], pos[1]))
            if stats is None:
                continue
            _, mean, var = stats
            mu = mean.cpu().numpy()
            sigma = var.cpu().numpy()
            # floor each channel's sigma by a fraction of the global std so that
            # near-constant channels do not dominate the distance
            sigma = np.maximum(sigma, 0.1 * self._gstd.cpu().numpy() ** 2)
            cell = grids[i].reshape(-1, dims)
            cell = (cell - gmean) / gstd
            d = np.sqrt(np.maximum(((cell - mu) ** 2 / sigma).sum(1), 0.0))
            dmap = d.reshape(32, 32)
            y, x, ex, ey = pos
            up = np.asarray(Image.fromarray(dmap).resize((ex - x, ey - y), Image.Resampling.BILINEAR))
            crop = win[:ey - y, :ex - x]
            out[y:ey, x:ex] += crop * up
            wgt[y:ey, x:ex] += crop
        result = out / np.maximum(wgt, 1e-8)
        return gaussian_filter(result, sigma=2.0)
