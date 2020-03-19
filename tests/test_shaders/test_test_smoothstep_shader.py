from datetime import datetime

import numpy as np
from PIL import Image

from src.opengl.object_vertices import get_2d_plane
from src.shaders.test_shader import TestShader
from tests.misc_required_for_testing import save_images, render_opengl, assert_abs_mean_diff
from src.misc.render_funcs import render


class TestTestShader:

    def setup(self):
        self.V, self.I = get_2d_plane()
        self.shader = TestShader()
        self.program = self.shader.get_program(len(self.V), set_defaults=True)
        self.program.bind(self.V)

        self.default_args = [t[-1] for t in self.shader.get_inputs()]
        self.W = 100
        self.H = 100

        self.AVG_PIXEL_TOLERANCE = 0.001

    def render_both(self):
        python_render = render(self.W, self.H, self.shader.shade, *self.default_args)
        opengl_render = render_opengl(self.W, self.H, self.program)
        return python_render, opengl_render

    def test_alignment_for_low_res(self):
        self.W = 50
        self.H = 50
        py, op = self.render_both()

        save_images(["python_alignment_LowRes", "opengl_alignment_LowRes"], [py, op])
        assert_abs_mean_diff(py,op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_alignment_for_low_res")

    def test_alignment_for_high_res(self):
        self.W = 200
        self.H = 200
        py, op = self.render_both()

        #save_images(["python_alignment_HiRes", "opengl_alignment_HiRes"], [py, op])
        assert_abs_mean_diff(py,op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_alignment_for_high_res")
