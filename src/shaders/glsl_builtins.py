import autograd.numpy as np
import numpy as onp


def mix(x: onp.ndarray, y: onp.ndarray, a: float) -> onp.ndarray:
    """Performs a linear interpolation between x and y using a to weight between them.
    A weight of 1.0 returns y and a weight of 0.0 returns x.
    """
    return x * (1.0 - a) + y * a
