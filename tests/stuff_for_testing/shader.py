import typing

import torch

from src.misc import render_funcs
from src.shaders.shader_super import Shader
from tests.stuff_for_testing import funcs


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


    def render_py_torch(self):
        return render_funcs.render_torch(self.W, self.H, self.shader.shade_torch, *self.args)


    def render_gl(self):
        return funcs.render_opengl(self.W, self.H, self.program)


    def render_both(self):
        return self.render_py_torch(), self.render_gl()
