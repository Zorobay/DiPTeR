import typing as ty

import torch
from torch import Tensor

SMALL = 0.000001


def mix(x: Tensor, y: Tensor, a: ty.Union[Tensor, float]) -> Tensor:
    """performs a linear interpolation between x and y using a to weight between them."""
    return x * (1 - a) + y * a


def fract(x: Tensor) -> Tensor:
    """returns the fractional part of x."""
    return x - torch.floor(x)


def smoothstep(edge0: ty.Union[float, Tensor], edge1: ty.Union[float, Tensor], x: Tensor) -> Tensor:
    """
    performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1. This is useful in cases where a threshold function with a smooth
    transition is desired.
    """
    t = torch.clamp((x - edge0) / ((edge1 - edge0)+SMALL), 0., 1.)
    return t * t * (3.0 - 2.0 * t)


def step(edge: ty.Union[float, Tensor], x: Tensor) -> Tensor:
    return (x >= edge).float()
