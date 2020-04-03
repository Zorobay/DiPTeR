import logging

import numpy as np
import torch

from src.misc.render_funcs import render_torch
from src.mytensor import Teensor
from src.opengl.object_vertices import get_2d_plane
from src.shaders.brick_shader import BrickShader
from tests.stuff_for_testing import funcs
from tests.stuff_for_testing.funcs import save_images, render_opengl, assert_abs_mean_diff, render_opengl_callback_loop
from tests.stuff_for_testing.shader import ShaderTest

_logger = logging.getLogger(__name__)


class TestBrickShader(ShaderTest):

    def __init__(self):
        super().__init__(BrickShader)

    def test_default_parameters(self):
        self.args = self.default_args
        py, gl = self.render_py_torch(), self.render_gl()

        assert_abs_mean_diff(py, gl, "The average absolute pixel difference of {} is too large!")

    def test_color_clipping(self):
        val = torch.tensor((2.0, -1, 0.0, 1.0))
        self.program['color_brick'] = val
        self.args = self.default_args
        self.args[4] = val

        py, op = self.render_both()
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when clipping colors!")

    def test_brick_elongate_zero(self):
        key = "brick_elongate"
        self.set_arg(key, torch.tensor(0.0))
        self.set_arg("brick_scale", torch.tensor(10.))

        py, gl = self.render_both()
        assert_abs_mean_diff(py, gl, "Average pixel difference of {} is too large for zero 'brick_elongate'!")

    def test_brick_elongate_small(self):
        key = "brick_elongate"
        key_i = 2
        val = 1.0
        self.program[key] = val
        self.args = self.default_args
        self.args[key_i] = val

        py, op = self.render_both()
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when 'brick_elongate' is low.")

    def test_brick_elongate_max(self):
        key = "brick_elongate"
        key_i = 2
        val = 100.

        self.program[key] = val
        self.args = self.default_args
        self.args[key_i] = val

        py, op = self.render_both()
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when 'brick_elongate' is at 100 (max)!")

    def test_brick_random(self):
        self.W = 50
        self.H = 100
        self.args = self.shader.get_parameters_list_torch(False)
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

        assert_abs_mean_diff(pys, gls, "Average pixel difference of {} is too large when randomizing input!", test_name="test_brick_random")
