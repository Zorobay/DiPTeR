from autograd import numpy as np
from numpy.testing import assert_allclose

from src.shaders.lib import glsl_builtins as gl


def test_step():
    # Test for single float
    f = gl.step
    edge = 0.5
    assert f(edge, 0.) == 0.0, "Value of step for single float is wrong!"
    assert f(edge, edge - 0.001) == 0.0, "Value of step for float close to edge is wrong!"
    assert f(edge, edge) == 1.0, "Value of step with x=edge is wrong!"
    edge = 100000.0
    assert f(edge, 0) == 0.0, "Value of step for large edge is wrong!"
    assert f(edge, edge - 0.001) == 0.0, "Value of step for float close to large edge is wrong!"
    assert f(edge, edge) == 1.0, "Value of step with x=edge when edge is large is wrong!"
    edge = -10
    assert f(edge, -11) == 0.0, "Value of step when edge is negative is wrong!"
    assert f(edge, edge) == 1.0, "Value of step when edge is negative is wrong!"

    # Test for array
    edge = np.array((-5, -0.1, 0.0, 1.0, 10000))
    x = np.array((0, -1000, 0, 0, 0))
    assert_allclose(f(edge, x), np.array((1, 0, 1.0, 0, 0)), err_msg="Value of step for edge and x as list-like is wrong!")
    x = -0.1
    assert_allclose(f(edge, x), np.array((1., 1., 0., 0., 0.)), err_msg="Value of step when edge is list but not x is wrong!")
    edge = 1
    x = np.array((1, -1000, 10))
    assert_allclose(f(edge, x), np.array((1, 0, 1)), err_msg="Value of step when x is list but not edge is wrong!")


def test_mix():
    f = gl.mix
    x = np.array((1., 0.4, 0., 1.))
    y = np.array((0.4, 0., 0., 0.3))
    a = 0.
    assert_allclose(f(x, y, a), x, err_msg="mix with a=0 does not return x.")
    a = 1.0
    assert_allclose(f(x, y, a), y, err_msg="mix with a=1.0 does not return y.")
    a = 0.5
    assert_allclose(f(x, y, a), np.array((0.7, 0.2, 0, 0.65)), err_msg="mix with a=0.5 did not return the correct value!")
    a = -0.5
    assert_allclose(f(x, y, a), np.array((1.3, 0.6, 0, 1.35)), err_msg="mix with a=0.5 did not return the correct value!")
    a = np.array((0, 0.1, 0.2, 1.0))
    assert_allclose(f(x, y, a), np.array((x[0], 0.36, 0, y[3])))


def test_fract():
    f = gl.fract
    assert_allclose(f(1.4), 0.4)
    assert_allclose(f(1.5999), 0.5999)
    assert_allclose(f(-0.55), 0.45)
    assert_allclose(f(1.0), 0.0)
    assert_allclose(f(555.34), 0.34)
    x = np.array((1.4, 1.5999, -0.55, 1.0, 555.34))
    truth = np.array((0.4, 0.5999, 0.45, 0.0, 0.34))
    assert_allclose(f(x), truth, err_msg="fract returns wrong value for arrays")


def test_smoothstep():
    f = gl.smoothstep
    edge0 = 1.0
    edge1 = 3.0
    assert f(edge0, edge1, edge0 - 1) == 0, "smoothstep with x<edge0 did not return 0"
    assert f(edge0, edge1, edge0) == 0, "smoothstep with x=edge0 did not return 0"
    assert f(edge0, edge1, edge1) == 1, "smoothstep with x=edge1 did not return 1"
    assert f(edge0, edge1, edge1 + 1) == 1, "smoothstep with x>edge1 did not return 1"
    x = edge0 + (edge1 - edge0) / 2
    assert_allclose(f(edge0, edge1, x), 0.5, err_msg="smoothstep with x halfway between edges did not return a value halfway between edges.")
