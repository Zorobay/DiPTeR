import logging

import torch
import numpy as np
from src.misc.render_funcs import render_torch
from src.shaders.color_shader import ColorShader
from tests.stuff_for_testing import funcs
from tests.stuff_for_testing.funcs import render_opengl, assert_abs_mean_diff

_logger = logging.getLogger(__name__)


class TestColorShader:

    @classmethod
    def setup_class(cls):
        cls.AVG_PIXEL_TOLERANCE = 0.005
        cls.W = 100
        cls.H = 100

    def setup(self):
        self.shader = ColorShader()
        self.program = self.shader.get_program()
        self.default_args = [torch.tensor(t[-1]) for t in self.shader.get_inputs()]
        self.args = []

    def render_both(self):
        python_render = render_torch(self.W, self.H, self.shader.shade, *self.default_args)
        opengl_render = render_opengl(self.W, self.H, self.program)
        return python_render, opengl_render

    def test_default_parameters(self):
        python_render, opengl_render = self.render_both()
        assert_abs_mean_diff(python_render, opengl_render, "The average absolute pixel difference of {} is too large!")

    def test_random_parameters(self):
        self.W = 100
        self.H = 100
        self.args = self.shader.get_parameters_list_torch(False)
        gls = []
        pys = []
        params = []

        self.args = funcs.randomize_inputs_torch(self.args, self.shader)
        self.shader.set_inputs(self.args)

        def cb(gl):
            params.append(self.args)
            py = np.flip(render_torch(self.W, self.H, self.shader.shade, *self.args), axis=0)
            pys.append(py)
            gls.append(gl)

            self.args = funcs.randomize_inputs_torch(self.args, self.shader)
            self.shader.set_inputs(self.args)

        funcs.render_opengl_callback_loop(self.W, self.H, self.program, cb, iter=5, freq=4)

        for py,gl in zip(pys, gls):
            assert_abs_mean_diff(py, gl, "The average absolute pixel difference of {} is too large when randomizing parameters!")
            funcs.assert_abs_max_diff(py, gl, tol=0.01)
