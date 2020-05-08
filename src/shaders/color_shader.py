from numpy.core.multiarray import ndarray

import src.opengl.internal_types
from src.shaders.shader_super import *


class ColorShader(Shader):
    VERTEX_SHADER_FILENAME = "vertex_shader.glsl"
    FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Color", "color", src.opengl.internal_types.INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((1., 1., 1.)))
        ]

    def shade_torch(self, vert_pos: Tensor, color: Tensor) -> Tensor:
        return color
