"""Compact fully convolutional regression baseline."""
from torch import nn


def output_activation(name: str) -> nn.Module:
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "linear":
        return nn.Identity()
    raise ValueError("Output activation must be sigmoid or linear")


class CNN(nn.Module):
    def __init__(self, channels: int, cfg: dict):
        super().__init__()
        filters = cfg.get("initial_filters", 16)
        modules = []
        for _ in range(3):
            modules.append(nn.Conv2d(channels, filters, 3, padding=1))
            if cfg.get("batch_normalization", True):
                modules.append(nn.BatchNorm2d(filters))
            modules.extend([nn.ReLU(), nn.Dropout2d(cfg.get("dropout", .1))])
            channels = filters
        modules.extend([nn.Conv2d(filters, 1, 1), output_activation(cfg.get("output_activation", "sigmoid"))])
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)
