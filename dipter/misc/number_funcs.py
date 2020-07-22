import numpy as np


def round_significant(x, p:int) -> np.ndarray:
    """Rounds a number 'x' to 'p' number of significant figures."""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
