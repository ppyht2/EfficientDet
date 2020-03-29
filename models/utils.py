from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F

ModelParams = namedtuple(
    "ModelParams",
    ["width_coefficient", "depth_coefficient", "resolution", "dropout_rate"],
)

BlockParams = namedtuple(
    "BlockParams",
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


def get_efficientnet_params(model_name):
    params_dict = {
        "efficientnet-b0": ModelParams(1.0, 1.0, 224, 0.2),
        "efficientnet-b1": ModelParams(1.0, 1.1, 240, 0.2),
        "efficientnet-b2": ModelParams(1.1, 1.2, 260, 0.3),
        "efficientnet-b3": ModelParams(1.2, 1.4, 300, 0.3),
        "efficientnet-b4": ModelParams(1.4, 1.8, 380, 0.4),
        "efficientnet-b5": ModelParams(1.6, 2.2, 456, 0.4),
        "efficientnet-b6": ModelParams(1.8, 2.6, 528, 0.5),
        "efficientnet-b7": ModelParams(2.0, 3.1, 600, 0.5),
        "efficientnet-b8": ModelParams(2.2, 3.6, 672, 0.5),
        "efficientnet-l2": ModelParams(4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
