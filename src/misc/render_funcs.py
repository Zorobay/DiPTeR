import typing

import autograd.numpy as anp
import numpy as np


def render(width: int, height: int, f: typing.Callable, *args):
    img = np.zeros((width, height, 4))

    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = np.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0)
    y_pos = np.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0)

    for x in range(width):
        for y in range(height):
            vert_pos = anp.array((x_pos[x], y_pos[y], 0.))
            img[y, x, :] = f(vert_pos, *args)

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

        return loss_sum / (self.width * self.height * 4)
