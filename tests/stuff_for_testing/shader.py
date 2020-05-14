import typing

import torch

from src.misc import render_funcs
from src.shaders.shader_super import Shader
from tests.stuff_for_testing import funcs
from tests.stuff_for_testing.funcs import assert_abs_mean_diff, assert_abs_max_diff


class ShaderTest:

    def __init__(self, shader: typing.Type[Shader]):
        self.shader_class = shader
        self.shader = None
        self.program = None
        self.args = []
        self.default_args = []
        self.W = 100
        self.H = 100
        self.PIXEL_TOLERANCE = 1. / 255.

    def setup(self):
        self.shader = self.shader_class()
        self.program = self.shader.get_program()
        self.default_args = [torch.tensor(t[-1]) for t in self.shader.get_inputs()]
        self.args = self.default_args

    def set_arg(self, name: str, val: typing.Any):
        """Sets a new value to the uniform 'name' in both the program and the self.args parameters."""
        assert self.args
        for i, info in enumerate(self.shader.get_inputs()):
            uniform = info[1]
            if uniform == name:
                self.args[i] = val
                self.program[uniform] = val
                return

        raise AttributeError("The uniform {} does not exist in shader {}!".format(name, self.shader_class))

    def render_py_torch(self):
        return render_funcs.render_torch(self.W, self.H, self.shader.shade, *self.args)

    def render_gl(self):
        return funcs.render_opengl(self.W, self.H, self.program)

    def render_both(self):
        return self.render_py_torch(), self.render_gl()

    def test_default_parameters(self):
        py, gl = self.render_py_torch(), self.render_gl()

        assert_abs_mean_diff(py, gl, "The average absolute pixel difference of {} is too large!", tol=self.PIXEL_TOLERANCE)
        assert_abs_max_diff(py, gl, tol=self.PIXEL_TOLERANCE)

    def test_all_params_max(self):
        for i, info in enumerate(self.shader.get_inputs()):
            max_ = info[3][1]
            uniform = info[1]
            in_type = info[2]
            if "array" in in_type:
                shape = info[4].shape
                self.set_arg(uniform, torch.ones(shape) * max_)
            else:
                self.set_arg(uniform, torch.ones(1) * max_)

        py, gl = self.render_both()
        assert_abs_max_diff(py, gl, tol=self.PIXEL_TOLERANCE)
        assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE)

    def test_all_params_min(self):
        for i, info in enumerate(self.shader.get_inputs()):
            min_ = info[3][0]
            uniform = info[1]
            in_type = info[2]
            if "array" in in_type:
                shape = info[4].shape
                self.set_arg(uniform, torch.ones(shape) * min_)
            else:
                self.set_arg(uniform, torch.ones(1) * min_)

        py, gl = self.render_both()
        assert_abs_max_diff(py, gl, tol=self.PIXEL_TOLERANCE)
        assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE)
