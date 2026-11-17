"""Localize defects at two feature strides with one frozen ResNet-18."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _update_sample(
    features: torch.Tensor,
    previous: tuple[torch.Tensor, torch.Tensor] | None,
    generator: torch.Generator,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.rand(len(features), generator=generator, device=features.device, dtype=torch.float64)
    if previous is not None:
        bank, priorities = previous
        features = torch.cat((bank, features))
        keys = torch.cat((priorities, keys))

    # The largest random keys give a uniform sample over the complete stream.
    priorities, keep = keys.topk(min(size, len(keys)), sorted=False)
    return features[keep], priorities


def _distance_map(
    grid: torch.Tensor, reference: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    bank, squared_norms = reference
    queries = grid.flatten(0, 2)
    # Bound the FP32 distance matrix to 32 MiB for realistic bank sizes.
    rows_per_chunk = max(1, min(4096, (8 * 1024**2) // len(bank)))
    scores = []
    for chunk in queries.split(rows_per_chunk):
        distances = torch.addmm(squared_norms, chunk, bank.T, alpha=-2.0)
        distances += chunk.square().sum(dim=1, keepdim=True)
        scores.append(distances.amin(dim=1).clamp_min_(0).sqrt_())
    return torch.cat(scores).reshape(grid.shape[:3]).cpu()


def _coreset(bank: torch.Tensor, size: int, seed: int) -> torch.Tensor:
    """Select diverse points from a bounded reservoir in a random projection."""
    if len(bank) <= size:
        return bank
    rng = torch.Generator(device=bank.device).manual_seed(seed)
    projection = torch.randn((bank.shape[1], 16), generator=rng, device=bank.device)
    features = bank @ projection
    norms = features.square().sum(dim=1)
    closest = torch.full_like(norms, float("inf"))
    current = (features - features.mean(0)).square().sum(dim=1).argmax().reshape(1)
    selected = torch.empty(size, dtype=torch.long, device=bank.device)
    for index in range(size):
        selected[index:index + 1] = current
        point = features.index_select(0, current)
        distances = torch.addmv(norms, features, point.flatten(), alpha=-2.0)
        distances += point.square().sum()
        closest = torch.minimum(closest, distances)
        closest.scatter_(0, current, -1)
        current = closest.argmax().reshape(1)
    return bank[selected]


class MultiScaleDetector:
    """Fit separate nearest-neighbor references at strides four and eight.

    The fine branch uses unpooled 64-channel layer-1 features and a compact
    bank to keep the denser grid affordable. The context branch uses the
    layer-2 and layer-3 pooling, sampling, and scoring of PatchDetector.
    """

    def __init__(
        self, device: str, bank_size: int = 4096, fine_bank_size: int = 1024, seed: int = 42,
        *, backbone: torch.nn.Module | None = None,
    ) -> None:
        if min(bank_size, fine_bank_size) < 1:
            raise ValueError("Both bank sizes must be positive")

        self.device = torch.device(device)
        self.bank_size = bank_size
        self.fine_bank_size = fine_bank_size
        self.seed = seed
        self.coreset = False
        if backbone is None:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = backbone.to(
            device=self.device, memory_format=torch.channels_last
        )
        self.backbone.requires_grad_(False).eval()
        self._references: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None

    def _feature_grids(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.backbone
        batch = batch.to(device=self.device, non_blocking=True, memory_format=torch.channels_last)
        stem = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
        layer1 = model.layer1(stem)
        layer2 = model.layer2(layer1)
        layer3 = model.layer3(layer2)

        context_fine = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        context_coarse = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        context_coarse = F.interpolate(
            context_coarse, size=context_fine.shape[-2:], mode="bilinear", align_corners=False
        )
        context = torch.cat((context_fine, context_coarse), dim=1)
        return layer1.permute(0, 2, 3, 1), context.permute(0, 2, 3, 1)

    @torch.inference_mode()
    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        """Fit both references from normalized, defect-free image batches."""
        context_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        fine_generator = torch.Generator(device=self.device).manual_seed(self.seed ^ 1)
        fine_sample = context_sample = None
        multiplier = 4 if self.coreset else 1

        for batch in batches:
            fine, context = self._feature_grids(batch)
            fine_sample = _update_sample(
                fine.flatten(0, 2), fine_sample, fine_generator, multiplier * self.fine_bank_size
            )
            context_sample = _update_sample(
                context.flatten(0, 2), context_sample, context_generator, multiplier * self.bank_size
            )

        if fine_sample is None or context_sample is None or not len(fine_sample[0]):
            raise ValueError("Cannot fit reference banks without training patches")

        fine_bank, _ = fine_sample
        context_bank, _ = context_sample
        if self.coreset:
            fine_bank = _coreset(fine_bank, self.fine_bank_size, self.seed ^ 1)
            context_bank = _coreset(context_bank, self.bank_size, self.seed)
        self._references = (
            (fine_bank, fine_bank.square().sum(dim=1)),
            (context_bank, context_bank.square().sum(dim=1)),
        )

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU distance maps in fine, context order."""
        if self._references is None:
            raise RuntimeError("Fit the detector before predicting")

        fine, context = self._feature_grids(batch)
        fine_reference, context_reference = self._references
        return _distance_map(fine, fine_reference), _distance_map(context, context_reference)
