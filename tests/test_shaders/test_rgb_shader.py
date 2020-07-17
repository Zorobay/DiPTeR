import logging

import torch

from dipter.shaders.rgb_shader import RGBShader
from tests.stuff_for_testing import funcs
from tests.stuff_for_testing.funcs import assert_abs_mean_diff, render_opengl_callback_loop, assert_abs_max_diff
from tests.stuff_for_testing.shader import ShaderTest

_logger = logging.getLogger(__name__)


class TestRGBShader(ShaderTest):

    def __init__(self):
        super().__init__(RGBShader)

    def test_color_clipping(self):
        self.set_arg('red', -1.)
        self.set_arg('green', 3.)
        self.set_arg('blue', 0.5)

        py, op = self.render_both()
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when clipping colors!", tol=self.PIXEL_TOLERANCE)

    def test_rgb_random(self):
        self.W = 50
        self.H = 100
        self.args = self.shader.get_parameters_list(False)
        self.args = funcs.randomize_inputs_torch(self.args, self.shader)
        self.shader.set_inputs(self.args)

        gls = []
        pys = []
        params = []

        def callback(gl):
            params.append(self.args)
            py = self.render_py_torch()
            pys.append(py)
            gls.append(gl)

            self.args = funcs.randomize_inputs_torch(self.args, self.shader)
            self.shader.set_inputs(self.args)

        render_opengl_callback_loop(self.W, self.H, self.program, callback, 5)

        assert_abs_mean_diff(pys, gls, "Average pixel difference of {} is too large when randomizing input!", test_name="test_brick_random",
                             tol=self.PIXEL_TOLERANCE)
        assert_abs_max_diff(pys, gls, tol=self.PIXEL_TOLERANCE)
