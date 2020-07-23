import torch
from torch import Tensor

from dipter.shaders.shader_super import Shader


def cat(values: list) -> Tensor:
    return torch.cat(values, dim=2)


def x(t: Tensor) -> Tensor:
    return t[:, :, 0].unsqueeze(-1)


def y(t: Tensor) -> Tensor:
    return t[:, :, 1].unsqueeze(-1)


def z(t: Tensor) -> Tensor:
    return t[:, :, 2].unsqueeze(-1)


def w(t: Tensor) -> Tensor:
    return t[:, :, 3].unsqueeze(-1)


def xx(t: Tensor) -> Tensor:
    return cat([x(t), x(t)])


def xy(t: Tensor) -> Tensor:
    return t[:, :, 0:2]


def yy(t: Tensor) -> Tensor:
    return cat([y(t), y(t)])


def yz(t: Tensor) -> Tensor:
    return t[:, :, 1:3]


def zz(t: Tensor) -> Tensor:
    return cat([y(t), z(t)])


def zw(t: Tensor) -> Tensor:
    return t[:, :, 2:4]


def zzzz(t: Tensor) -> Tensor:
    return torch.cat([z(t), z(t), z(t), z(t)], dim=2)


def vec1(val: float) -> Tensor:
    size = [*Shader.frag_pos().shape[0:2], 1]
    return torch.full(size, val, dtype=torch.float32, device=Shader.frag_pos().device)


def vec3(val: float) -> Tensor:
    return torch.full_like(Shader.frag_pos(), val)


def vec4(val: float) -> Tensor:
    frag_pos = Shader.frag_pos()
    size = [*frag_pos.shape[0:2], 4]
    return torch.full(size, val, dtype=torch.float32, device=frag_pos.device)
