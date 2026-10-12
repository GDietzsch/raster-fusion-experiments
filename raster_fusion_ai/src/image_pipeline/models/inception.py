"""Multi-scale blocks for dense regression; not canonical GoogLeNet/Inception-v3."""
import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, cfg: dict, residual: bool = False):
        super().__init__()
        if residual:
            raise ValueError('This Inception experiment does not combine residual blocks')
        width = max(1, outputs // 4)

        def conv(a, b, size):
            layers = [nn.Conv2d(a, b, size, padding=size // 2)]
            if cfg.get('batch_normalization', True):
                layers.append(nn.BatchNorm2d(b))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.branches = nn.ModuleList([
            conv(inputs, width, 1),
            nn.Sequential(conv(inputs, width, 1), conv(width, width, 3)),
            nn.Sequential(conv(inputs, width, 1), conv(width, width, 3), conv(width, width, 3)),
            nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), conv(inputs, width, 1)),
        ])
        self.fuse = conv(4 * width, outputs, 1)
        self.dropout = nn.Dropout2d(cfg.get('dropout', 0.0))

    def forward(self, x):
        return self.dropout(self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1)))
