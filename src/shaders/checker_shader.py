import typing

import torch
from torch import Tensor

import src.shaders.lib.glsl_builtins as gl
from src.opengl.internal_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGB
from src.shaders.shader_super import FunctionShader


class CheckerShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "checker_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Color1", "color1", INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.ones(3)),
            ("Color2", "color2", INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.zeros(3)),
            ("Scale", "scale", INTERNAL_TYPE_FLOAT, (0, 100), 10.0)
        ]

    def shade_mat(self, vert_pos: Tensor, color1: Tensor, color2: Tensor, scale: Tensor) -> Tensor:
        p = vert_pos * scale

        p_int = torch.abs(torch.floor(p))

        check = torch.eq(torch.eq(gl.mod(p_int[0], 2.0), gl.mod(p_int[1], 2.0)), gl.mod(p_int[2], 2.0))
        if check:
            return color1
        else:
            return color2

    def shade(self, vert_pos: Tensor, color1: Tensor, color2: Tensor, scale: Tensor) -> Tensor:
        p = vert_pos * scale

        p_int = torch.abs(torch.floor(p))

        check = torch.eq(torch.eq(gl.mod(p_int[0], 2.0), gl.mod(p_int[1], 2.0)), gl.mod(p_int[2], 2.0))
        if check:
            return color1
        else:
            return color2
        #return gl.mix(color2, color1, check)
