"""PatchCore-style localization with a small normal reference bank."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class PatchDetector:
    """Score patches by their distance to the nearest normal reference.

    This baseline keeps a uniform sample of training patches and uses their
    raw nearest-neighbor distances. PatchCore uses a coreset selection step
    and an additional image-level score reweighting step.
    """

    def __init__(self, device: str, bank_size: int = 4096, seed: int = 42) -> None:
        if bank_size < 1:
            raise ValueError("bank_size must be positive")

        self.device = torch.device(device)
        self.bank_size = bank_size
        self.seed = seed
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT).to(
            device=self.device, memory_format=torch.channels_last
        )
        self.backbone.requires_grad_(False).eval()
        self._reference: tuple[torch.Tensor, torch.Tensor] | None = None

    def _feature_grid(self, batch: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        batch = batch.to(
            device=self.device, non_blocking=True, memory_format=torch.channels_last
        )
        stem = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
        layer2 = model.layer2(model.layer1(stem))
        layer3 = model.layer3(layer2)

        fine = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        coarse = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((fine, coarse), dim=1).permute(0, 2, 3, 1)

    @torch.inference_mode()
    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        """Build a bounded memory bank from normalized, defect-free images."""
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        bank = None
        priorities = None

        for batch in batches:
            features = self._feature_grid(batch).flatten(0, 2)
            keys = torch.rand(len(features), generator=generator, device=self.device, dtype=torch.float64)
            if bank is not None:
                features = torch.cat((bank, features))
                keys = torch.cat((priorities, keys))

            # Keeping the highest random keys yields a uniform sample of all
            # patches seen so far, even when batches contain different counts.
            priorities, keep = keys.topk(min(self.bank_size, len(keys)), sorted=False)
            bank = features[keep]

        if bank is None or not len(bank):
            raise ValueError("Cannot fit a memory bank without training patches")

        self._reference = bank, bank.square().sum(dim=1)

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Return one continuous distance map per input image on the CPU."""
        if self._reference is None:
            raise RuntimeError("Fit the detector before predicting")

        bank, bank_norms = self._reference
        grid = self._feature_grid(batch)
        queries = grid.flatten(0, 2)
        chunk_size = max(1, min(4096, (8 * 1024**2) // len(bank)))

        scores = []
        for chunk in queries.split(chunk_size):
            distances = torch.addmm(bank_norms, chunk, bank.T, alpha=-2.0)
            distances += chunk.square().sum(dim=1, keepdim=True)
            scores.append(distances.amin(dim=1).clamp_min_(0).sqrt_())

        return torch.cat(scores).reshape(grid.shape[:3]).cpu()
