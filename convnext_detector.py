"""Frozen DINOv3 ConvNeXt features for the two-resolution patch detector."""

from pathlib import Path

import torch
import torch.nn.functional as F

from multiscale_detector import MultiScaleDetector


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "artifacts/backbones/convnext_tiny_dinov3.safetensors"
WEIGHTS_REVISION = "39c7bb32f83c11bbb13cb23058f8852a4dac88ed"
WEIGHTS_SHA256 = "a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f"


class ConvNeXtDetector(MultiScaleDetector):
    """Use stride-four detail and stride-eight context from ConvNeXt-Tiny.

    The reference sampling, distances, and score maps use the same code as
    the ResNet detector. Inputs use the same ImageNet RGB normalization.
    """

    def __init__(
        self,
        device: str,
        bank_size: int = 4096,
        fine_bank_size: int = 1024,
        seed: int = 42,
        weights: Path = DEFAULT_WEIGHTS,
    ) -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Missing DINOv3 ConvNeXt-Tiny weights: {weights}")
        try:
            from safetensors.torch import load_file
            from timm import create_model
        except ImportError as error:
            raise ImportError("ConvNeXt requires the optional timm and safetensors packages") from error

        backbone = create_model("convnext_tiny.dinov3_lvd1689m", pretrained=False, num_classes=0)
        backbone.load_state_dict(load_file(str(weights), device="cpu"))
        backbone.prune_intermediate_layers(indices=[0, 1, 2], prune_norm=True)
        super().__init__(device, bank_size, fine_bank_size, seed, backbone=backbone)

    def _feature_grids(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.backbone
        batch = batch.to(device=self.device, non_blocking=True, memory_format=torch.channels_last)
        fine = model.stages[0](model.stem(batch))
        stage1 = model.stages[1](fine)
        stage2 = model.stages[2](stage1)

        context_fine = F.avg_pool2d(stage1, kernel_size=3, stride=1, padding=1)
        context_coarse = F.avg_pool2d(stage2, kernel_size=3, stride=1, padding=1)
        context_coarse = F.interpolate(
            context_coarse, size=context_fine.shape[-2:], mode="bilinear", align_corners=False
        )
        context = torch.cat((context_fine, context_coarse), dim=1)
        return fine.permute(0, 2, 3, 1), context.permute(0, 2, 3, 1)
