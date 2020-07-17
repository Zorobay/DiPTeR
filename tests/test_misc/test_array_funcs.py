import unittest

import numpy as np

from dipter.misc import array_funcs


def test_index_of():
    f = array_funcs.index_of

    a = [(6, 4, 2), (1, 2), (-5, 3, 4, 1)]
    assert f(a, lambda t: len(t) == 2) ==  1, "Failed to find index of tuple of length 2!"
    assert f(a, lambda t: t[0] < 0) ==  2, "Failed to find index of negative first element in tuple!"

    a = np.array([(4, 1), (-4, 1), (-5, 1)])
    assert f(a, lambda x: x[1] > x[0]) ==  1, "Failed to find index where second tuple element is greater than first element!"
    assert f(a, lambda x: x[0] > 10) ==  -1, "Did not return -1 for condition that is not met!"

    a = [1, 2, 3, 4, -5, 0, 0]
    assert f(a, lambda x: x < 0) ==  4, "Did not return correct index for simple list!"
