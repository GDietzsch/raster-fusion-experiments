"""Compact U-Net with interpolation to preserve odd input dimensions."""
import torch
from torch import nn
from .cnn import output_activation


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, cfg: dict, residual: bool = False):
        super().__init__()
        modules = []
        for channels in [inputs, outputs]:
            modules.append(nn.Conv2d(channels, outputs, 3, padding=1))
            if cfg.get("batch_normalization", True):
                modules.append(nn.BatchNorm2d(outputs))
            modules.append(nn.ReLU())
        modules.append(nn.Dropout2d(cfg.get("dropout", .1)))
        self.layers = nn.Sequential(*modules)
        self.shortcut = nn.Conv2d(inputs, outputs, 1) if residual else None

    def forward(self, x):
        result = self.layers(x)
        return torch.relu(result+self.shortcut(x)) if self.shortcut is not None else result


class UNet(nn.Module):
    def __init__(self, channels: int, cfg: dict, residual: bool = False):
        super().__init__()
        depth, filters = int(cfg.get("depth", 2)), int(cfg.get("initial_filters", 16))
        self.upsampling = cfg.get('upsampling', 'bilinear')
        if self.upsampling not in ('nearest', 'bilinear'):
            raise ValueError('Unsupported upsampling mode')
        if not 1 <= depth <= 5 or filters < 1:
            raise ValueError("Require depth in [1,5] and positive initial_filters")
        widths = [filters*2**level for level in range(depth+1)]
        self.encoders = nn.ModuleList()
        for width in widths:
            self.encoders.append(ConvBlock(channels, width, cfg, residual))
            channels = width
        self.decoders = nn.ModuleList([ConvBlock(widths[i+1]+widths[i], widths[i], cfg, residual) for i in reversed(range(depth))])
        self.head = nn.Sequential(nn.Conv2d(widths[0], 1, 1), output_activation(cfg.get("output_activation", "sigmoid")))

    def forward(self, x):
        skips = []
        for index, encoder in enumerate(self.encoders):
            if index:
                x = nn.functional.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            options = {'align_corners': False} if self.upsampling == 'bilinear' else {}
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode=self.upsampling, **options)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)
