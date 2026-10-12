"""Neural architecture registry."""
def build_model(name: str, channels: int, cfg: dict):
    if name == "resnet":
        from .resnet import ResNet
        return ResNet(channels, cfg)
    if name == "cnn":
        from .cnn import CNN
        return CNN(channels, cfg)
    if name == "unet":
        from .unet import UNet
        return UNet(channels, cfg)
    if name == "resunet":
        from .resunet import ResUNet
        return ResUNet(channels, cfg)
    raise ValueError(f"Unknown image model: {name}")
