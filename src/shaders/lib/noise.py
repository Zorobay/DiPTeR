import torch
from torch import Tensor

import src.shaders.lib.glsl_builtins as gl


def simpleSmoothstep(p: Tensor) -> Tensor:
    return p * p * (3. - 2. * p)


def random(p: Tensor) -> Tensor:
    return gl.fract(torch.sin(p[:, :, 0] * 888. + p[:, :, 1] * 5322.) * 3451.)


def smoothNoise2D(p: Tensor) -> Tensor:
    w = p.shape[0]
    h = p.shape[1]
    local_uv = simpleSmoothstep(gl.fract(p))
    local_id = torch.floor(p)

    # Find the noise value at the four corners of a local box
    bl = random(local_id)
    br = random(local_id + torch.tensor((1, 0)).repeat(w, h, 1))
    tl = random(local_id + torch.tensor((0, 1)).repeat(w, h, 1))
    tr = random(local_id + torch.tensor((1, 1)).repeat(w, h, 1))

    # Interpolate between bottom, top, and bottom -> top
    b = gl.mix(bl, br, local_uv[:,:,0])
    t = gl.mix(tl, tr, local_uv[:,:,0])
    return gl.mix(b, t, local_uv[:,:,1]).unsqueeze(-1)


def fractalBrownianMotion(p: Tensor, detail: Tensor) -> Tensor:
    """
    Adds octaves of smooth noise to create "fractal brownian motion" or "fractal noise".

    :param p: 2D Tensor of coordinates
    :param detail: Integer 2D tensor dictating the number of octaves to add
    :return: A pseudo-random 2D noise Tensor
    """
    w, h = p.shape[0], p.shape[1]

    # Initial values
    value = torch.tensor((0.0)).repeat(w, h, 1)
    amplitude = torch.tensor((0.5)).repeat(w, h, 1)

    # Loop and add octaves of more and more detailed noise
    for i in range(detail):
        value = value + (amplitude * smoothNoise2D(p))
        p = p * 2.0
        amplitude = amplitude * 0.5

    return value
