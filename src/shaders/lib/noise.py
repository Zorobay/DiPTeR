import torch
from torch import Tensor

import src.shaders.lib.glsl_builtins as gl

def simpleSmoothstep(p: Tensor) -> Tensor:
    return p*p*(3. - 2.*p)


def Noise2D(p: Tensor) -> Tensor:
    return gl.fract(torch.sin(p[0] * 888. + p[1] * 5322.) * 3451.)


def SmoothNoise2D(p: Tensor) -> Tensor:
    local_uv = simpleSmoothstep(gl.fract(p))
    local_id = torch.floor(p)

    # Find the noise value at the four corners of a local box
    bl = Noise2D(local_id)
    br = Noise2D(local_id + torch.tensor((1,0)))
    tl = Noise2D(local_id + torch.tensor((0,1)))
    tr = Noise2D(local_id + torch.tensor((1,1)))

    # Interpolate between bottom, top, and bottom -> top
    b = gl.mix(bl,br,local_uv[0])
    t = gl.mix(tl,tr,local_uv[0])

    return gl.mix(b,t,local_uv[1])
