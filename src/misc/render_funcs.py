import logging
import typing
import warnings

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
CHANNELS = 3

logger_ = logging.getLogger(__name__)


def render_iter(width: int, height: int, f: typing.Callable, *args):
    warnings.warn("Iterative rendering method is deprecated!", DeprecationWarning, stacklevel=2)
    img = np.zeros((height, width, CHANNELS))
    x_pos, y_pos = get_coordinates(width, height)

    for row in range(height):
        for col in range(width):
            frag_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(frag_pos, *args)
            img[height - 1 - row, col, :] = val

    return img


def get_coordinates(width: int, height: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = torch.from_numpy(np.linspace(0., 1.0, width, endpoint=False)) + (x_res / 2.0)
    y_pos = torch.from_numpy(np.linspace(0., 1.0, height, endpoint=False)) + (y_res / 2.0)
    return x_pos, y_pos


def generate_frag_pos(width: int, height: int) -> torch.Tensor:
    x_pos, y_pos = torch.meshgrid(*get_coordinates(width, height))
    frag_pos = torch.stack([x_pos, y_pos, torch.zeros_like(x_pos)], dim=2)
    return frag_pos.float()
