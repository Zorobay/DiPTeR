import typing

import autograd.numpy as anp
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = 3


# def render(width: int, height: int, f: typing.Callable, *args):
#     img = np.zeros((width, height, 4))
#
#     x_res = 1.0 / width
#     y_res = 1.0 / height
#     x_pos = np.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0)
#     y_pos = np.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0)
#
#     for x in range(width):
#         for y in range(height):
#             vert_pos = anp.array((x_pos[x], y_pos[y], 0.))
#             img[y, x, :] = f(vert_pos, *args)
#
#     return img


def get_coordinates2(width: int, height: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = torch.from_numpy(np.linspace(0., 1.0, width, endpoint=False)) + (x_res / 2.0)
    y_pos = torch.from_numpy(np.linspace(0., 1.0, height, endpoint=False)).flip(0) + (y_res / 2.0)
    return x_pos, y_pos


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
            vert_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(vert_pos, *args)
            img[height - 1 - row, col, :] = val

    return img


def render_torch2(width: int, height: int, f: typing.Callable, *args):
    img = torch.zeros((CHANNELS, width, height), device=device)
    x_pos, y_pos = get_coordinates(width, height)

    for row in range(height):
        for col in range(width):
            vert_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(vert_pos, *args)
            img[:, height - 1 - row, col] = val

    return img


def render_torch3(width: int, height: int, f, *args):
    x_pos, y_pos = torch.meshgrid((get_coordinates2(width, height)))
    pix_pos = torch.stack([x_pos, y_pos, torch.zeros_like(x_pos)], dim=2)
    return f(pix_pos, *args)


def render_torch_with_callback(width: int, height: int, row_callback: typing.Callable, f: typing.Callable, *args):
    img = np.zeros((height, width, CHANNELS))
    x_pos, y_pos = get_coordinates(width, height)

    for row in range(height):
        row_callback(row)
        for col in range(width):
            vert_pos = torch.tensor((x_pos[col], y_pos[row], 0.), dtype=torch.float32)
            val = f(vert_pos, *args)
            img[height - 1 - row, col, :] = val

    return img


class Loss:

    def __init__(self, truth: np.ndarray, f: typing.Callable):
        self.truth = truth
        self.width = truth.shape[0]
        self.height = truth.shape[1]
        self.f = f

    def loss(self, *args) -> float:
        x_res = 1.0 / self.self.width
        y_res = 1.0 / self.height
        x_pos = anp.linspace(0, 1.0, self.width, endpoint=False) + (x_res / 2.0)
        y_pos = anp.linspace(0, 1.0, self.height, endpoint=False) + (y_res / 2.0)
        loss_sum = 0

        for x in range(self.width):
            for y in range(self.height):
                vert_pos = anp.array((x_pos[x], y_pos[y], 0.))
                val = self.f(vert_pos, *args)
                loss_sum += anp.sum(anp.abs(self.truth[y, x, :] - val))

        return loss_sum / (self.width * self.height * CHANNELS)
