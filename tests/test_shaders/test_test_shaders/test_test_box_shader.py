import torch

from tests.stuff_for_testing import funcs
from tests.stuff_for_testing.shader import ShaderTest
from tests.stuff_for_testing.shaders.test_box_shader import TestBoxShader


class TestTestBoxShader(ShaderTest):

    def __init__(self):
        super().__init__(TestBoxShader)

    def test_default(self):
        py, gl = self.render_both()
        funcs.assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE)
        funcs.assert_abs_max_diff(py, gl, tol=self.PIXEL_TOLERANCE)

    def test_size_min(self):
        self.set_arg('size', torch.tensor((0., 0.)))
        py, gl = self.render_both()
        funcs.assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE)
        funcs.assert_abs_max_diff(py, gl, tol=self.PIXEL_TOLERANCE)

    def test_size_max(self):
        self.set_arg('size', torch.tensor((1., 1.)))
        py, gl = self.render_both()
        funcs.assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE)
        funcs.assert_abs_max_diff(py, gl, tol=self.PIXEL_TOLERANCE)

    def test_size_random(self):
        pys = []
        gls = []

        self.set_arg('size', torch.rand((2,)))

        def cb(gl):
            py = self.render_py_torch()
            pys.append(py)
            gls.append(gl)
            self.set_arg('size', torch.rand((2,)))

        funcs.render_opengl_callback_loop(self.W, self.H, self.program, cb, 5)

        funcs.assert_abs_mean_diff(pys, gls, tol=self.PIXEL_TOLERANCE)
        funcs.assert_abs_max_diff(pys, gls, tol=self.PIXEL_TOLERANCE)
