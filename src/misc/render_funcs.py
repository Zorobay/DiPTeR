import typing

import autograd.numpy as anp
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
CHANNELS = 3


# def input(width: int, height: int, f: typing.Callable, *call_dict):
#     img = np.zeros((width, height, 4))
#
#     x_res = 1.0 / width
#     y_res = 1.0 / height
#     x_pos = np.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0)
#     y_pos = np.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0)
#
#     for input in range(width):
#         for y in range(height):
#             frag_pos = anp.array((x_pos[input], y_pos[y], 0.))
#             img[y, input, :] = f(frag_pos, *call_dict)
#
#     return img


def get_coordinates(width: int, height: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = torch.from_numpy(np.linspace(0., 1.0, width, endpoint=False)) + (x_res / 2.0)
    y_pos = torch.from_numpy(np.linspace(0., 1.0, height, endpoint=False)) + (y_res / 2.0)
    return x_pos, y_pos


def render_torch(width: int, height: int, f: typing.Callable, *args):
    img = np.zeros((height, width, CHANNELS))
    x_pos, y_pos = get_coordinates(width, height)

    for row in range(height):
        for col in range(width):
            frag_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(frag_pos, *args)
            img[height - 1 - row, col, :] = val

    return img


def render_torch_loop(width: int, height: int, f: typing.Callable, *args):
    img = torch.zeros((width, height, CHANNELS), device=device)
    x_pos, y_pos = get_coordinates(width, height)

    for row in range(height):
        for col in range(width):
            frag_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(frag_pos, *args)
            img[col, row,:] = val

    return img


def generate_vert_pos(width: int, height: int) -> torch.Tensor:
    x_pos, y_pos = torch.meshgrid((get_coordinates(width, height)))
    frag_pos = torch.stack([x_pos, y_pos, torch.zeros_like(x_pos)], dim=2)
    return frag_pos

def render_torch_matrix(width: int, height: int, f, *args):
    x_pos, y_pos = torch.meshgrid((get_coordinates(width, height)))
    pix_pos = torch.stack([x_pos, y_pos, torch.zeros_like(x_pos)], dim=2)
    return f(pix_pos, *args)


def render_torch_with_callback(width: int, height: int, row_callback: typing.Callable, f: typing.Callable, *args):
    img = np.zeros((height, width, CHANNELS))
    x_pos, y_pos = get_coordinates(width, height)

    for row in range(height):
        row_callback(row)
        for col in range(width):
            frag_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(frag_pos, *args)
            img[height - 1 - row, col, :] = val

    return img
