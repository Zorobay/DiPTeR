import typing as ty

import torch
from torch import Tensor

SMALL = 0.000001


def mix(x: Tensor, y: Tensor, a: ty.Union[Tensor, float]) -> Tensor:
    """performs a linear interpolation between input and y using a to weight between them."""
    return x * (1 - a) + y * a


def fract(x: Tensor) -> Tensor:
    """returns the fractional part of input."""
    return x - torch.floor(x)


def smoothstep(edge0: ty.Union[float, Tensor], edge1: ty.Union[float, Tensor], x: Tensor) -> Tensor:
    """
    performs smooth Hermite interpolation between 0 and 1 when edge0 < input < edge1. This is useful in cases where a threshold function with a smooth
    transition is desired.
    """
    t = torch.clamp((x - edge0) / ((edge1 - edge0) + SMALL), 0., 1.)
    return t * t * (3.0 - 2.0 * t)


def step(edge: ty.Union[float, Tensor], x: Tensor) -> Tensor:
    """
    `step` generates a step function by comparing x to edge.
    For element i of the return value, 0.0 is returned if x[i] < edge[i], and 1.0 is returned otherwise.
    """
    return (torch.sign(x-edge) + 1) / 2


def mod(x: Tensor, y: ty.Union[float, Tensor]) -> Tensor:
    """returns the value of input modulo y"""
    return torch.remainder(x, y)


def dot(x: Tensor, y: Tensor) -> Tensor:
    """dot returns the dot product of two vectors, x and y. i.e., x[0]⋅y[0]+x[1]⋅y[1]+..."""
    return torch.einsum("abc, abc -> ab", x, y).unsqueeze(-1)
