from collections import namedtuple
import torch
import torch.nn

EfficientNetConfig = namedtuple(
    "EfficientNetConfig", ["name", "width_coef", "depth_coef", "resolution"]
)

BlockConfig = namedtuple(
    "BlockConfig",
    [
        "repetition",
        "kernel_size",
        "stride",
        "in_channels",
        "out_channels",
        "expand_ratio",
        "se_ratio",
    ],
)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
