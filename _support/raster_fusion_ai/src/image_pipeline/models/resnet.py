"""Small residual regressor initialized from scratch; not ResNet-18/50."""
from torch import nn
from .unet import ConvBlock
from .cnn import output_activation


class ResNet(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        width = cfg.get('initial_filters', 16)
        self.layers = nn.Sequential(
            ConvBlock(channels, width, cfg, residual=True),
            ConvBlock(width, width, cfg, residual=True),
            nn.Conv2d(width, 1, 1),
            output_activation(cfg.get('output_activation', 'linear')))

    def forward(self, x):
        return self.layers(x)
