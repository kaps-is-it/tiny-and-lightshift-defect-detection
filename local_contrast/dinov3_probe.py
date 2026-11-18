"""Feature-swap experiment: DINOv3 ViT-S/16 vs the ResNet-18 reference bank.

Same tiling, the same bank budget, the same evaluation harness as the tiled
baseline — only the frozen feature extractor changes. This probe answers one
question honestly: how much of the baseline's head-room is backbone-limited?
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

from benchmark import summarise_by_light
from evaluation import ImageResult
from run_experiment import load_splits, read_image, tile_origins, INPUT_SIZE

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def as_tensor(arr: np.ndarray) -> torch.Tensor:
    """Resize a native crop to 256x256 with reflect padding, ImageNet-normalized."""
    h, w = arr.shape[:2]
    ratio = INPUT_SIZE / max(h, w)
    tw, th = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (INPUT_SIZE - tw) // 2, (INPUT_SIZE - th) // 2
    r = np.asarray(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((tw, th), Image.Resampling.BILINEAR),
                   dtype=np.float32) / 255.0
    r = np.pad(r, ((top, INPUT_SIZE - top - th), (left, INPUT_SIZE - left - tw), (0, 0)), mode="reflect")
    t = (r - MEAN[None, None, :]) / STD[None, None, :]
    return torch.from_numpy(t.transpose(2, 0, 1))


class Dinov3PatchCore:
    """PatchCore-style detector whose features come from a frozen DINOv3 ViT."""

    def __init__(self, device, blocks=(8, 10, 11), bank_size=4096, seed=42):
        self.device = torch.device(device)
        import timm

        self.model = timm.create_model(
            "vit_small_patch16_dinov3_qkvb.lvd1689m",
            pretrained=False,
            checkpoint_path="artifacts/superadd-assets/dinov3-vits16-timm.safetensors",
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        self.blocks = tuple(blocks)
        self.patch = 16
        self.bank_size = bank_size
        self.seed = seed
        self._reference = None

    def _batches(self, paths, batch_size):
        batch = []
        for path in paths:
            arr = np.asarray(read_image(path), dtype=np.float32) / 255.0
            for y in tile_origins(arr.shape[0], 256):
                for x in tile_origins(arr.shape[1], 256):
                    ex, ey = min(x + 256, arr.shape[1]), min(y + 256, arr.shape[0])
                    batch.append(as_tensor(arr[y:ey, x:ex]))
                    if len(batch) == batch_size:
                        yield torch.stack(batch)
                        batch = []
        if batch:
            yield torch.stack(batch)

    @torch.inference_mode()
    def _features(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        out = self.model.forward_intermediates(
            batch, indices=self.blocks, norm=False,
            output_fmt="NLC", intermediates_only=True,
        )
        feats = [o / o.norm(dim=-1, keepdim=True).clamp_min(1e-6) for o in out]
        return torch.cat(feats, dim=-1).contiguous()

    @torch.inference_mode()
    def fit(self, paths, batch_size=32):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        bank = None
        keys = None
        total = 0
        for batch in self._batches(paths, batch_size):
            feats = self._features(batch).flatten(0, 1)
            fresh = feats
            if bank is not None:
                fresh = torch.cat((bank, feats))
            keys = torch.rand(len(fresh), generator=generator, device=self.device, dtype=torch.float64) if keys is None or bank is None \
                else torch.cat((keys, torch.rand(len(feats), generator=generator, device=self.device, dtype=torch.float64)))
            priorities, keep = keys.topk(min(self.bank_size, len(keys)), sorted=False)
            bank = fresh[keep]
            keys = priorities
            total += len(feats)
        self._reference = bank, bank.square().sum(dim=1)
        print(f"bank built from {total} patch vectors")
        return self

    @torch.inference_mode()
    def predict_tiled(self, arr: np.ndarray, tile_size=256, batch_size=16) -> np.ndarray:
        placements, tiles = [], []
        for y in tile_origins(arr.shape[0], tile_size):
            for x in tile_origins(arr.shape[1], tile_size):
                placements.append((y, x, min(x + tile_size, arr.shape[1]), min(y + tile_size, arr.shape[0])))
                tiles.append(as_tensor(arr[y:min(y + tile_size, arr.shape[0]), x:min(x + tile_size, arr.shape[1])]))
        acc = np.zeros(arr.shape[:2], np.float32)
        wgt = np.zeros_like(acc)
        taper = np.maximum(np.hanning(tile_size), 0.1).astype(np.float32)
        win = taper[:, None] * taper[None, :]
        token_side = tile_size // self.patch
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(tiles[i:i + batch_size])
            grid = self._features(batch)                          # (b, 256, D)
            queries = grid.flatten(0, 1).contiguous()               # (b*256, D)
            bank, bank_norms = self._reference
            scores = []
            for chunk in queries.split(4096):
                dists = torch.addmm(bank_norms, chunk, bank.T, alpha=-2.0) + chunk.square().sum(dim=1, keepdim=True)
                scores.append(dists.amin(dim=1).clamp_min_(0).sqrt_())
            flat = torch.cat(scores).reshape(batch.shape[0], token_side, token_side)
            up = F.interpolate(flat.unsqueeze(1).float(), size=(INPUT_SIZE, INPUT_SIZE),
                               mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
            for j, (y, x, ex, ey) in enumerate(placements[i:i + batch_size]):
                tw, th = ex - x, ey - y
                sm = up[j]
                if tw != tile_size or th != tile_size:
                    # crop the reflect padding back to the tile rectangle
                    ratio = INPUT_SIZE / max(tw, th)
                    cw, ch = max(1, round(tw * ratio)), max(1, round(th * ratio))
                    left, top = (INPUT_SIZE - cw) // 2, (INPUT_SIZE - ch) // 2
                    sm = np.asarray(Image.fromarray(sm[top:top + ch, left:left + cw]).resize((tw, th)))
                acc[y:ey, x:ex] += win[:th, :tw] * sm
                wgt[y:ey, x:ex] += win[:th, :tw]
        return gaussian_filter(acc / np.maximum(wgt, 1e-8), sigma=2.0)

    @torch.inference_mode()
    def predict(self, paths_or_arr, batch_size=16):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mvtec_ad_2_dataset")
    parser.add_argument("--category", default="can")
    parser.add_argument("--blocks", default="8,10,11")
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/dinov3-probe")
    args = parser.parse_args()
    blocks = tuple(int(b) for b in args.blocks.split(","))

    training, validation, captures = load_splits(Path(args.dataset), args.category, 42)
    det = Dinov3PatchCore(args.device, blocks, args.bank_size, seed=42)

    print(f"fitting DINOv3 probe (blocks {blocks}, bank {args.bank_size})", flush=True)
    t0 = time.perf_counter()
    det.fit(training)
    print(f"fit in {time.perf_counter() - t0:.0f}s", flush=True)

    by_light = defaultdict(list)
    for capture in captures:
        by_light[capture.lighting].append(capture)

    for light in sorted(by_light, key=lambda n: (n != "regular", n)):
        group = by_light[light]
        results = []
        for capture in group:
            arr = np.asarray(read_image(capture.image), dtype=np.float32)
            am = det.predict_tiled(arr)
            mask = None
            if capture.mask:
                with Image.open(capture.mask) as t:
                    mask = np.asarray(t) != 0
            results.append(ImageResult(capture.image.name, capture.scene, light, mask is not None,
                                       float(am.max()), am, mask))
        summary = summarise_by_light({light: results})["per_light"][light]
        print(f"  {light:<16} AU-PRO {summary['au_pro_05']*100:.1f}%  tiny {summary['tiny_pro_at_05']*100:.1f}%  fn {summary['fn']}/{summary['defective']}", flush=True)
    au = []
    for light, results in by_light.items():
        pass


if __name__ == "__main__":
    main()
