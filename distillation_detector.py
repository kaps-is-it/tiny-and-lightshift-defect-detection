"""A compact student–teacher fine branch trained on normal crops."""

from collections.abc import Iterable
from random import Random

import torch
import torch.nn.functional as F
from torch import nn

from multiscale_detector import MultiScaleDetector, _coreset, _distance_map, _update_sample


CROP_CACHE_LIMIT = 512
STUDENT_BATCH_SIZE = 12


def _student_model(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 24, 5, stride=2, padding=2),
        nn.GELU(),
        nn.Conv2d(24, 32, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(32, 48, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(48, channels, 1),
    )


def _augment_normal(crops: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Vary exposure and contrast, with a quarter of the crops kept as controls."""
    uniforms = torch.rand(len(crops), 3, 1, 1, generator=generator, device=crops.device)
    gain, contrast, active = [parameter[:, None] for parameter in uniforms.unbind(1)]
    mean = crops.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = crops.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    rgb = crops * std + mean
    changed = rgb * (0.85 + 0.30 * gain)
    changed = (changed - 0.5) * (0.85 + 0.30 * contrast) + 0.5
    return torch.where(active > 0.25, (changed.clamp(0, 1) - mean) / std, crops)


class DistillationDetector(MultiScaleDetector):
    """Use student prediction error for detail and ResNet patch matching for context.

    Train only on source-normal crops and their mild photometric variations.
    The fine score is the mean squared residual across standardized ResNet
    layer-1 channels. Both maps are returned on CPU, in fine/context order.
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
        self._fitted: tuple[nn.Module, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None = None

    def _teacher(self, batch: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        stem = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
        return model.layer1(stem)

    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        context_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        reservoir_rng = Random(self.seed ^ 6)
        crop_rng = Random(self.seed ^ 9)
        crops: list[torch.Tensor] = []
        context_sample = None
        seen = 0
        with torch.no_grad():
            for batch in batches:
                height, width = batch.shape[-2:]
                if min(height, width) < self.crop_size:
                    raise ValueError("Source tiles are smaller than the training crops")
                _, context = self._feature_grids(batch)
                context_sample = _update_sample(
                    context.flatten(0, 2), context_sample, context_generator,
                    self.bank_size * (4 if self.coreset else 1),
                )
                for tile in batch:
                    slot = seen if seen < CROP_CACHE_LIMIT else reservoir_rng.randrange(seen + 1)
                    seen += 1
                    if slot >= CROP_CACHE_LIMIT:
                        continue
                    top = crop_rng.randrange(height - self.crop_size + 1)
                    left = crop_rng.randrange(width - self.crop_size + 1)
                    crop = tile[:, top:top + self.crop_size, left:left + self.crop_size]
                    crop = crop.to(device="cpu", dtype=torch.float16, copy=True)
                    if slot == len(crops):
                        crops.append(crop)
                    else:
                        crops[slot] = crop

            if not crops or context_sample is None:
                raise ValueError("Cannot fit without normal source tiles")

            bank = context_sample[0]
            if self.coreset:
                bank = _coreset(bank, self.bank_size, self.seed)
            context_reference = bank, bank.square().sum(dim=1)
            clean = torch.stack(crops).to(self.device)
            total = squared_total = None
            samples = 0
            for batch in clean.split(STUDENT_BATCH_SIZE):
                target = self._teacher(batch.float().contiguous(memory_format=torch.channels_last))
                channel_sum = target.sum((0, 2, 3), keepdim=True)
                squared_sum = target.square().sum((0, 2, 3), keepdim=True)
                total = channel_sum if total is None else total + channel_sum
                squared_total = squared_sum if squared_total is None else squared_total + squared_sum
                samples += target.numel() // target.shape[1]
            mean = total / samples
            scale = (squared_total / samples - mean.square()).clamp_min(0.0025).sqrt()

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed ^ 12)
            student = _student_model(mean.shape[1])
        student = student.to(device=self.device, memory_format=torch.channels_last).train()
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.0015)
        generator = torch.Generator(device=self.device).manual_seed(self.seed ^ 27)
        for _ in range(self.steps):
            indexes = torch.randint(
                len(clean), (min(STUDENT_BATCH_SIZE, len(clean)),),
                device=self.device, generator=generator,
            )
            batch = _augment_normal(clean[indexes].float(), generator)
            batch = batch.contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                target = (self._teacher(batch) - mean) / scale
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(student(batch), target).backward()
            optimizer.step()

        self._fitted = student.eval(), mean, scale, context_reference

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fitted is None:
            raise RuntimeError("Fit the detector before predicting")
        student, mean, scale, context_reference = self._fitted
        batch = batch.to(device=self.device, non_blocking=True, memory_format=torch.channels_last)
        fine, context = self._feature_grids(batch)
        target = (fine.permute(0, 3, 1, 2) - mean) / scale
        residual = (student(batch) - target).square().mean(dim=1)
        return residual.cpu(), _distance_map(context, context_reference)
