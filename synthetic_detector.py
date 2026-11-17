"""Train a fine segmentation branch on simulated defects in normal tiles."""

from collections.abc import Iterable
import heapq

import torch
from torch import nn
import torch.nn.functional as F

from multiscale_detector import MultiScaleDetector, _coreset, _distance_map, _update_sample


class _Segmenter(nn.Module):
    """Combine native pixel features with a small lower-resolution context path."""

    def __init__(self) -> None:
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 16, 3, dilation=2, padding=2), nn.GELU(),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        detail = self.detail(images)
        context = F.interpolate(
            self.context(detail), size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.mask(torch.cat((detail, context), dim=1))[:, 0]


def _simulate_defects(
    clean: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate soft-edged scratches and spots, with some clean controls."""
    count, _, height, width = clean.shape
    random = torch.rand(count, 11, 1, 1, generator=generator, device=clean.device)
    center_x, center_y, angle, length, narrow, radius, aspect, shape, polarity, depth, active = (
        random.unbind(dim=1)
    )
    x = torch.arange(width, device=clean.device) - (0.15 + 0.70 * center_x) * (width - 1)
    y = torch.arange(height, device=clean.device).view(-1, 1) - (0.15 + 0.70 * center_y) * (height - 1)
    angle = angle * torch.pi
    along = x * angle.cos() + y * angle.sin()
    across = y * angle.cos() - x * angle.sin()

    # Fade the ends of each line over a few pixels and vary its width.
    stroke = torch.exp(-0.5 * (across / (0.65 + 1.6 * narrow)).square())
    stroke = stroke * torch.sigmoid((4 + min(height, width) * 0.20 * length - along.abs()) / 1.8)
    radius = 2 + 6 * radius
    spot = torch.exp(-0.5 * ((across / radius).square() + (along / (radius * (0.6 + aspect))).square()))
    opacity = torch.where(shape < 0.55, stroke, spot)
    active = active > 0.20
    active[0] = True
    opacity = opacity * active

    mean = clean.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = clean.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    rgb = clean * std + mean
    # Move toward white or black using a fraction of the available contrast.
    room = torch.where((polarity > 0.5)[:, None], 1 - rgb, -rgb)
    amount = opacity * (0.08 + 0.52 * depth)
    changed = clean + room * amount[:, None] / std
    return changed, (opacity >= 0.30).to(clean.dtype)


class SyntheticDetector(MultiScaleDetector):
    """Pair a synthetic-defect segmentation map with ResNet patch matching.

    Train the segmenter from crops of normal source tiles. The context branch
    uses the same frozen ResNet features and distance scores as MultiScaleDetector.
    The fine bank size is accepted for compatibility with that detector's interface.
    """

    def __init__(
        self,
        device: str,
        bank_size: int = 4096,
        fine_bank_size: int = 1024,
        seed: int = 42,
        *,
        steps: int = 180,
        crop_size: int = 128,
    ) -> None:
        if steps < 1 or crop_size < 32:
            raise ValueError("Training steps must be positive and crop size must be at least 32")
        super().__init__(device, bank_size, fine_bank_size, seed)
        self.steps = steps
        self.crop_size = crop_size
        self._segmenter: _Segmenter | None = None
        self._context_reference: tuple[torch.Tensor, torch.Tensor] | None = None

    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        context_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        sample_generator = torch.Generator().manual_seed(self.seed ^ 6)
        crop_generator = torch.Generator().manual_seed(self.seed ^ 9)
        samples = []
        seen = 0
        context_sample = None
        for batch in batches:
            with torch.no_grad():
                _, context = self._feature_grids(batch)
                context_sample = _update_sample(
                    context.flatten(0, 2), context_sample, context_generator,
                    self.bank_size * (4 if self.coreset else 1),
                )

            height, width = batch.shape[-2:]
            if min(height, width) < self.crop_size:
                raise ValueError("Source tiles are smaller than the synthetic training crops")
            priorities = torch.rand(len(batch), generator=sample_generator, dtype=torch.float64)
            for tile, priority in zip(batch, priorities.tolist(), strict=True):
                seen += 1
                if len(samples) == 512 and priority <= samples[0][0]:
                    continue
                top = int(torch.randint(height - self.crop_size + 1, (), generator=crop_generator))
                left = int(torch.randint(width - self.crop_size + 1, (), generator=crop_generator))
                crop = tile[:, top:top + self.crop_size, left:left + self.crop_size]
                entry = (priority, seen, crop.to(device="cpu", dtype=torch.float16, copy=True))
                if len(samples) == 512:
                    heapq.heapreplace(samples, entry)
                else:
                    heapq.heappush(samples, entry)

        if context_sample is None or not samples:
            raise ValueError("Cannot fit without normal source tiles")

        context_bank = context_sample[0]
        if self.coreset:
            context_bank = _coreset(context_bank, self.bank_size, self.seed)
        clean = torch.stack([crop for _, _, crop in samples]).to(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed ^ 12)
            segmenter = _Segmenter()
        segmenter = segmenter.to(device=self.device, memory_format=torch.channels_last).train()

        generator = torch.Generator(device=self.device).manual_seed(self.seed ^ 27)
        optimizer = torch.optim.AdamW(segmenter.parameters(), lr=0.002)
        pos_weight = torch.tensor(8.0, device=self.device)
        for _ in range(self.steps):
            indexes = torch.randint(len(clean), (min(12, len(clean)),), generator=generator, device=self.device)
            inputs, target = _simulate_defects(clean[indexes].float(), generator)
            optimizer.zero_grad(set_to_none=True)
            logits = segmenter(inputs.contiguous(memory_format=torch.channels_last))
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            probabilities = logits.sigmoid()
            overlap = (probabilities * target).sum()
            soft_dice = (2 * overlap + 1) / (probabilities.sum() + target.sum() + 1)
            (loss + 1 - soft_dice).backward()
            optimizer.step()

        self._segmenter = segmenter.eval()
        self._context_reference = context_bank, context_bank.square().sum(dim=1)

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._segmenter is None or self._context_reference is None:
            raise RuntimeError("Fit the detector before predicting")
        batch = batch.to(device=self.device, non_blocking=True, memory_format=torch.channels_last)
        _, context = self._feature_grids(batch)
        fine = self._segmenter(batch).sigmoid().cpu()
        return fine, _distance_map(context, self._context_reference)
