import typing

import numpy as np

import src.opengl.shader_types as st
from src.shaders.shader_super import Shader


class ColorShader(Shader):

    def __init__(self):
        super().__init__()
        self.VERTEX_SHADER_FILENAME = "color_shader_vert.glsl"
        self.FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Color", "color", st.INTERNAL_TYPE_RGB, (0, 0), (0., 0., 0.))
        ]

    def shade(self, color: np.ndarray):
        return np.array((1.0, 1.0, 1.0))
