import torch
from torch import nn
import torch.nn.functional as F
import math
from models.utils import Swish, ModelParams, BlockParams


def scale_repeats(depth: int, config: ModelParams):
    """ compound scaling for depth coefficinet """
    return math.ceil(depth * config.depth_coefficient)


def scale_filters(filter: int, config: ModelParams):
    """ compound scaling for width coefficient """
    divisor: int = 8  # ensure filter size is divisible by 8
    estimated_filter = filter * config.width_coefficient

    # round to the nearest multipler of divisor
    multiplier = round((estimated_filter / divisor))
    actual_filter = int(multiplier) * divisor
    return actual_filter


class StemBlock(nn.Module):
    IN_CHANNELS: int = 3
    STRIDE: int = 2
    KERNEL_SIZE: int = 3
    OUT_CHANNELS: int = 32
    PADDING: int = 1

    def __init__(self, model_params: ModelParams):
        super(StemBlock, self).__init__()
        self._model_params = model_params

        self._out_channels = scale_filters(
            StemBlock.OUT_CHANNELS, self._model_params
        )

        self._conv = nn.Conv2d(
            in_channels=StemBlock.IN_CHANNELS,
            out_channels=self.out_channels,
            kernel_size=StemBlock.KERNEL_SIZE,
            stride=StemBlock.STRIDE,
            padding=StemBlock.PADDING,
            bias=False,
        )

        self._bn = nn.BatchNorm2d(num_features=self.out_channels)

        self._act = Swish()

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = self._act(x)
        return x


class MBConvBlock(nn.Module):
    """ this module is made of several components

    # 1. expansion from residual bottle neck
    # 2. depthwise convolution
    # 3. squeeze and excitation
    # 4. projection
    """

    def __init__(self, block_params: BlockParams, model_params):
        self._block_params = block_params
        self._model_params = model_params

        # 1. Expansion
        exp_in_channels = self._block_params.in_channels
        exp_out_channels = exp_in_channels * self._block_params.expand_ratio
        self._expand_conv = nn.Conv2d(
            in_channels=exp_in_channels,
            out_channels=exp_out_channels,
            kernel_size=1,
            bias=False,
            # TODO: shall we consider padding here?
        )
        self._expand_bn = nn.BatchNorm2d(num_features=exp_out_channels)

        # 2. Depthwise Convolutions
        self._depthwise_conv = nn.Conv2d(
            in_channels=exp_out_channels,
            out_channels=exp_out_channels,
            groups=exp_out_channels,
            kernel_size=self._block_params.kernel_size,
            stride=self._block_params.stride,
            bias=False,
        )
        self._depthwise_bn = nn.BatchNorm2d(num_features=exp_out_channels)

        # 3. Squeeze and Excitation
        squeezed_channels = max(
            1,
            int(
                self._block_params.in_channels
                * self._block_params.squeeze_ratio
            ),
        )

        self._squeeze_conv = nn.Conv2d(
            in_channels=exp_out_channels,
            out_channels=squeezed_channels,
            kernel_size=1,
        )

        self._excitation_conv = nn.Conv2d(
            in_channels=squeezed_channels,
            out_channels=exp_out_channels,
            kernel_size=1,
        )

        # 4. Projection

    def forward(self, x):
        pass
