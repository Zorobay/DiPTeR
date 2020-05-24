import torch
from torch.testing import assert_allclose
from torch import tensor
from src.shaders.lib import glsl_builtins as gl


def test_step():
    # Test for single float
    f = gl.step
    edge = tensor(0.5)
    assert f(edge, tensor(0.)) == tensor(0.0), "Value of step for single float is wrong!"
    assert f(edge, edge - tensor(0.001)) == tensor(0.0), "Value of step for float close to edge is wrong!"
    assert f(edge, edge) == tensor(1.0), "Value of step with input=edge is wrong!"
    edge = tensor(100000.0, dtype=torch.float64)
    assert f(edge, 0) == tensor(0.0), "Value of step for large edge is wrong!"
    assert f(edge, edge - 0.001) == tensor(0.0), "Value of step for float close to large edge is wrong!"
    assert f(edge, edge) == tensor(1.0), "Value of step with input=edge when edge is large is wrong!"
    edge = tensor(-10)
    assert f(edge, tensor(-11)) == tensor(0.0), "Value of step when edge is negative is wrong!"
    assert f(edge, edge) == tensor(1.0), "Value of step when edge is negative is wrong!"

    # Test for array
    edge = tensor((-5, -0.1, 0.0, 1.0, 10000), dtype=torch.float32)
    x = tensor((0, -1000., 0, 0, 0), dtype=torch.float32)
    assert_allclose(f(edge, x), tensor((1., 0., 1.0, 0., 0.)))
    x = tensor(-0.1)
    assert_allclose(f(edge, x), tensor((1., 1., 0., 0., 0.)))
    edge = tensor(1)
    x = tensor((1, -1000, 10))
    assert_allclose(f(edge, x), tensor((1., 0., 1.)))


def test_mix():
    f = gl.mix
    x = tensor((1., 0.4, 0., 1.))
    y = tensor((0.4, 0., 0., 0.3))
    a = tensor(0.)
    assert_allclose(f(x, y, a), x)
    a = tensor(1.0)
    assert_allclose(f(x, y, a), y)
    a = tensor(0.5)
    assert_allclose(f(x, y, a), tensor((0.7, 0.2, 0, 0.65)))
    a = tensor(-0.5)
    assert_allclose(f(x, y, a), tensor((1.3, 0.6, 0, 1.35)))
    a = tensor((0, 0.1, 0.2, 1.0))
    assert_allclose(f(x, y, a), tensor((x[0], 0.36, 0, y[3])))


def test_fract():
    f = gl.fract
    assert_allclose(f(tensor(0.)), tensor(0.))
    assert_allclose(f(tensor(1.4)), tensor(0.4))
    assert_allclose(f(tensor(1.5999)), tensor(0.5999))
    assert_allclose(f(tensor(-0.55)), tensor(0.45))
    assert_allclose(f(tensor(1.0)), tensor(0.0))
    assert_allclose(f(tensor(555.34)), tensor(0.34))
    x = tensor((1.4, 1.5999, -0.55, 1.0, 555.34))
    truth = tensor((0.4, 0.5999, 0.45, 0.0, 0.34))
    assert_allclose(f(x), truth)


def test_smoothstep():
    f = gl.smoothstep
    edge0 = tensor(1.0)
    edge1 = tensor(3.0)

    assert f(edge0, edge1, tensor(0)) == tensor(0), "smoothstep with input=0 returned wrong value"
    assert f(edge0, edge1, edge0 - tensor(1)) == tensor(0), "smoothstep with input<edge0 did not return 0"
    assert f(edge0, edge1, edge0) == tensor(0), "smoothstep with input=edge0 did not return 0"
    assert f(edge0, edge1, edge1) == tensor(1), "smoothstep with input=edge1 did not return 1"
    assert f(edge0, edge1, edge1 + tensor(1)) == tensor(1), "smoothstep with input>edge1 did not return 1"
    x = edge0 + (edge1 - edge0) / tensor(2)
    assert_allclose(f(edge0, edge1, x), tensor(0.5))

    assert f(tensor(0),tensor(0),tensor(1)) == tensor(1.0), "smoothstep with edge0=edge1=0 did not return 1.0"
