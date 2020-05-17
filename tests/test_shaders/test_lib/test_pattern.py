from torch import tensor
from torch.testing import assert_allclose

from src.shaders.lib import pattern as pattern


def test_box():
    coord = tensor((0.5, 0.))
    size = tensor((1., 1.))
    # Should not belong to box because of the minimum edge smooth value added in box primary_function
    assert_allclose(pattern.box(coord, size), tensor(0.0))

    coord = tensor((0.5, 0.001))
    assert_allclose(pattern.box(coord, size), tensor(1.))

    size = tensor((0., 0.))
    assert_allclose(pattern.box(coord, size), tensor(0.))

    size = tensor((0.1, 1.))
    assert_allclose(pattern.box(coord, size), tensor(1.0))

    coord = tensor((0.5, 0.5))
    size = tensor((0.001, 0.3))
    assert_allclose(pattern.box(coord, size), tensor(1.0))
