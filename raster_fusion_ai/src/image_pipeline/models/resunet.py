"""Residual U-Net, sharing the tested encoder/decoder implementation."""
from .unet import UNet


class ResUNet(UNet):
    def __init__(self, channels: int, cfg: dict):
        super().__init__(channels, cfg, residual=True)
