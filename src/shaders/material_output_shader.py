import typing

import torch
from torch import Tensor

from src.opengl.internal_types import INTERNAL_TYPE_SHADER
from src.shaders.shader_super import CompilableShader


class MaterialOutputShader(CompilableShader):
    FRAGMENT_SHADER_FILENAME = "material_output_shader_frag.glsl"

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Shader", "shader", INTERNAL_TYPE_SHADER, (0, 1), torch.tensor((1.0, 1.0, 1.0)))
        ]

    def get_outputs(self) -> typing.List[typing.Tuple[str, str]]:
        return []

    def shade(self, vert_pos: Tensor, shader: Tensor) -> Tensor:
        return shader

    def shade_mat(self, shader: Tensor) -> Tensor:
        return shader
