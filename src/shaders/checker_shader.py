import numpy as onp
import autograd.numpy as np


class CheckerShader:

    def __init__(self, scale: float, color1: onp.ndarray, color2: onp.ndarray):
        self.color1 = None
        self.color2 = None
        self.scale = None
        self.scale_min = 0
        self.scale_max = 20
        self.color1 = color1
        self.color2 = color2
        self.scale = scale

    def checker(self, x: float, y: float) -> float:
        xi = np.abs(np.floor(x))
        yi = np.abs(np.floor(y))

        if np.mod(xi, 2) == np.mod(yi, 2):
            return 1.0

        return 0.0

    def shade(self, x: float, y: float):
        x = x * self.scale * 20
        y = y * self.scale * 20

        fac = self.checker(x, y)
        return self.color1 * fac + self.color2 * (1 - fac)
