import math

import torch
from torch.testing import assert_allclose

from src.shaders.lib import trig


def test_deg_to_rad():
    rad = trig.deg_to_rad(torch.tensor(0.))
    assert_allclose(rad, torch.tensor(0.))

    rad = trig.deg_to_rad(torch.tensor(360.))
    assert_allclose(rad, torch.tensor(math.pi * 2))

    rad = trig.deg_to_rad(torch.tensor(445.))
    assert_allclose(rad, torch.tensor(7.76672))

    rad = trig.deg_to_rad(torch.tensor(90.))
    assert_allclose(rad, torch.tensor(math.pi / 2))
