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
