import src.opengl.shader_types as st
from src.shaders.shader_super import *


class ColorShader(Shader):
    VERTEX_SHADER_FILENAME = "vertex_shader.glsl"
    FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Color", "color", st.INTERNAL_TYPE_ARRAY_RGBA, (0, 1), np.array((1., 0., 0., 1.)))
        ]

    def shade(self, vert_pos: ndarray, color: ndarray) -> ndarray:
        return color

    def shade_torch(self, vert_pos: Tensor, color: Tensor) -> Tensor:
        return color
