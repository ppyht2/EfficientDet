import math
from models.utils import EfficientNetConfig


def scale_depth(depth: int, config: EfficientNetConfig):
    scaled_depth = depth * config.depth_coef
    scaled_depth = math.ceil(scaled_depth)
    return scaled_depth
