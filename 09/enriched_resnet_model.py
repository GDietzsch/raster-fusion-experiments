"""Compact ResNet architecture used by algorithms 09 and 09A."""
from __future__ import annotations
from src.utils.source_paths import project_root, source_path
import torch
from torch import nn
from torch.nn import functional as F
RADIUS = 2
WIDE_STEP = 82

class ResidualBranch(nn.Module):
    """One spatial projection block followed by a pointwise residual block."""

    def __init__(self, channels: int, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, filters, 3)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3)
        self.bn2 = nn.BatchNorm2d(filters)
        self.shortcut = nn.Conv2d(channels, filters, 1)
        self.shortcut_bn = nn.BatchNorm2d(filters)
        self.refine1 = nn.Conv2d(filters, filters, 1)
        self.refine_bn1 = nn.BatchNorm2d(filters)
        self.refine2 = nn.Conv2d(filters, filters, 1)
        self.refine_bn2 = nn.BatchNorm2d(filters)

    @staticmethod
    def _conv(x: torch.Tensor, layer: nn.Conv2d, dilation: int) -> torch.Tensor:
        return layer(x) if dilation == 1 else F.conv2d(
            x, layer.weight, layer.bias, dilation=dilation)

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        radius = 2 * dilation
        height = x.shape[-2] - 2 * radius
        width = x.shape[-1] - 2 * radius
        if height < 1 or width < 1:
            raise ValueError('Residual branch needs a 5x5 support at its scale')
        center = x[:, :, radius:radius + height, radius:radius + width]
        residual = self.bn2(self._conv(
            F.relu(self.bn1(self._conv(x, self.conv1, dilation))),
            self.conv2, dilation))
        shortcut = self.shortcut_bn(self.shortcut(center))
        merged = F.relu(residual + shortcut)
        refinement = self.refine_bn2(self.refine2(
            F.relu(self.refine_bn1(self.refine1(merged)))))
        return F.relu(merged + refinement)


class ResidualCNN(nn.Module):
    """Small two-scale residual regressor for center-pixel raster targets.

    It uses the shortcut principle of ResNet but is not an ImageNet ResNet-18.
    """

    def __init__(self, channels: int, filters: int, wide_context: bool = True):
        super().__init__()
        self.architecture = 'resnet_compact'
        self.wide_context = wide_context
        self.local = ResidualBranch(channels, filters)
        if wide_context:
            self.wide = ResidualBranch(channels, filters)
        self.head = nn.Conv2d(2 * filters if wide_context else filters, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x[:, 0])
        if not self.wide_context:
            return self.head(local).squeeze(1)
        wide = self.wide(x[:, 1])
        return self.head(torch.cat((local, wide), dim=1)).squeeze(1)

    def forward_dense(self, x: torch.Tensor, height: int, width: int,
                      step: int = WIDE_STEP) -> torch.Tensor:
        radius = 2 * step if self.wide_context else RADIUS
        local_x = x[:, :, radius - RADIUS:radius + height + RADIUS,
                    radius - RADIUS:radius + width + RADIUS]
        local = self.local(local_x)
        if not self.wide_context:
            if local.shape[-2:] != (height, width):
                raise ValueError('Residual local inference changed tile dimensions')
            return self.head(local).squeeze(1)
        wide = self.wide(x, dilation=step)
        if local.shape[-2:] != (height, width) or wide.shape[-2:] != (height, width):
            raise ValueError('Residual dense inference changed tile dimensions')
        return self.head(torch.cat((local, wide), dim=1)).squeeze(1)
