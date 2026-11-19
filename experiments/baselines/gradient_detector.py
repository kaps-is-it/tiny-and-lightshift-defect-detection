"""Test local edge evidence alongside the frozen ResNet-18 context branch."""

import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights

from experiments.baselines.multiscale_detector import MultiScaleDetector


class GradientDetector(MultiScaleDetector):
    """Search normal Sobel and Laplacian features on a stride-four grid.

    Local contrast normalization reduces sensitivity to exposure. The two
    scales and separate polarities retain evidence from thin light or dark
    lines. This branch is a surface-detail challenger for the context maps.
    """

    def __init__(
        self, device: str, bank_size: int = 4096, fine_bank_size: int = 1024,
        seed: int = 42, *, backbone: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(device, bank_size, fine_bank_size, seed, backbone=backbone)
        normalization = ResNet18_Weights.DEFAULT.transforms()
        luma = torch.tensor([0.299, 0.587, 0.114], device=self.device, dtype=torch.float32)
        std = luma.new_tensor(normalization.std)
        mean = luma.new_tensor(normalization.mean)
        self._luma_weights = (luma * std).reshape(1, 3, 1, 1)
        self._luma_bias = (luma * mean).sum().reshape(1)
        self._filters = luma.new_tensor([
            [[-0.125, 0, 0.125], [-0.25, 0, 0.25], [-0.125, 0, 0.125]],
            [[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]],
            [[0, 0.25, 0], [0.25, -1, 0.25], [0, 0.25, 0]],
        ]).unsqueeze(1)
        gaussian = luma.new_tensor([0.25, 0.5, 0.25])
        self._blur = (gaussian[:, None] * gaussian[None, :]).reshape(1, 1, 3, 3)

    def _gradient_grid(self, batch: torch.Tensor) -> torch.Tensor:
        luma = F.conv2d(batch, self._luma_weights, bias=self._luma_bias)

        # Separable box averages give local first and second moments.
        moments = torch.cat((luma, luma.square()), dim=1)
        moments = F.avg_pool2d(moments, (1, 9), 1, (0, 4), count_include_pad=False)
        moments = F.avg_pool2d(moments, (9, 1), 1, (4, 0), count_include_pad=False)
        average, squared_average = moments.chunk(2, dim=1)
        contrast = (squared_average - average.square()).clamp_min_(0).sqrt_().add_(0.04)

        # Replicate the border so constant tiles do not acquire artificial edges.
        blurred = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode="replicate"), self._blur)
        features = []
        for source, dilation in ((luma, 1), (blurred, 2)):
            padded = F.pad(source, (dilation,) * 4, mode="replicate")
            response = F.conv2d(padded, self._filters, dilation=dilation)
            response /= contrast
            for polarity in (response, -response):
                pooled = F.max_pool2d(polarity, kernel_size=4, stride=4, ceil_mode=True)
                features.append(pooled.clamp_min_(0))

        return torch.cat(features, dim=1).permute(0, 2, 3, 1)

    def _feature_grids(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(device=self.device, non_blocking=True, memory_format=torch.channels_last)
        context = super()._feature_grids(batch)[1]
        return self._gradient_grid(batch), context
