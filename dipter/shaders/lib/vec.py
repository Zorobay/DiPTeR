import numbers

import torch
from torch import Tensor

from dipter.shaders.shader_super import Shader


def cat(*args) -> Tensor:
    return torch.cat(args, dim=2)


# ---- Single element retrieval functions ----

def x(t: Tensor) -> Tensor:
    return t[:, :, 0].unsqueeze(-1)


def y(t: Tensor) -> Tensor:
    return t[:, :, 1].unsqueeze(-1)


def z(t: Tensor) -> Tensor:
    return t[:, :, 2].unsqueeze(-1)


def w(t: Tensor) -> Tensor:
    return t[:, :, 3].unsqueeze(-1)


# ---- Double element retrieval functions ----

def xx(t: Tensor) -> Tensor:
    return cat(x(t), x(t))


def xy(t: Tensor) -> Tensor:
    return t[:, :, 0:2]


def yy(t: Tensor) -> Tensor:
    return cat(y(t), y(t))


def yz(t: Tensor) -> Tensor:
    return t[:, :, 1:3]


def zz(t: Tensor) -> Tensor:
    return cat(y(t), z(t))


def zw(t: Tensor) -> Tensor:
    return t[:, :, 2:4]


# ---- Triple element retrieval functions ----

def xxx(t: Tensor) -> Tensor:
    return cat(x(t), x(t), x(t))


def xyz(t: Tensor) -> Tensor:
    return cat(x(t), y(t), z(t))


def www(t: Tensor) -> Tensor:
    return cat(w(t), w(t), w(t))


# ---- Quadruple element retrieval functions ----

def zzzz(t: Tensor) -> Tensor:
    return cat(z(t), z(t), z(t), z(t))


def rep(val) -> Tensor:
    """Repeats the input value to create a matrix Tensor of correct render size."""
    size = [*Shader.frag_pos().shape[0:2], 1]
    if isinstance(val, numbers.Number):
        return torch.full(size, val, dtype=torch.float32, device=Shader.frag_pos().device)
    elif isinstance(val, Tensor):
        return val.repeat(*Shader.render_size(), 1)

    raise ValueError("Incorrect Shape or type of arguments!")

def vec2(*args) -> Tensor:
    return _vecn(*args, n=2)

def vec3(*args) -> Tensor:
    return _vecn(*args, n=3)


def vec4(*args) -> Tensor:
    return _vecn(*args, n=4)


def _vecn(*args, n: int):
    n_args = len(args)
    frag_pos = Shader.frag_pos()
    if all([isinstance(el, numbers.Number) for el in args]):  # Only numbers are given, create Tensor and repeat
        reps = 1 if n_args == n else n
        return torch.tensor(args, dtype=torch.float32, device=frag_pos.device).repeat(*Shader.render_size(), reps)
    elif all([isinstance(el, Tensor)] for el in args):  # Only Tensors are given, concatenate them
        if n_args == 1:
            return args[0].repeat(1, 1, n)
        else:
            return cat(*args)
    else:
        raise ValueError("Incorrect Shape or type of arguments!")
