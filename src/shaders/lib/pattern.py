import torch
from torch import Tensor

import src.shaders.lib.glsl_builtins as gl


def box(coord: Tensor, size: Tensor) -> Tensor:
    edge_smooth = 0.001
    size = 0.5 - size * 0.5
    uv = gl.smoothstep(size, size + edge_smooth, coord)
    uv = uv * gl.smoothstep(size, size + edge_smooth, 1.0 - coord)
    return (uv[:,:,0] * uv[:,:,1]).unsqueeze(-1)


def tile(coord: Tensor, scale: Tensor, shift: Tensor):
    tiled = coord * scale

    st = gl.step(1.0, torch.fmod(tiled[:, :, 0:2], 2.0))
    shifted_xy = tiled[:, :, 0:2] + shift * st.flip(-1)

    return gl.fract(torch.cat((shifted_xy, tiled[:,:,-1].unsqueeze(-1)), dim=2))
