import torch
from torch import Tensor
import src.shaders.lib.glsl_builtins as gl


def box(coord: Tensor, size: Tensor) -> Tensor:
    edge_smooth = 0.0001
    size = 0.5 - size * 0.5
    uv = gl.smoothstep(size, size + edge_smooth, coord)
    uv *= gl.smoothstep(size, size + edge_smooth, 1.0 - coord)
    return uv[0] * uv[1]
