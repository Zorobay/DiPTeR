from src.shaders.shader_super import *


class TestFragInterpolationShader(Shader):
    FRAGMENT_SHADER_FILENAME = "test/test_frag_interpolation_frag.glsl"
    VERTEX_SHADER_FILENAME = "test/vertex_shader.glsl"

    def __init__(self):
        super().__init__()


    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return []

    def shade(self, vert_pos: ndarray, *args) -> ndarray:
        frag_color = None

        frag_color = anp.append(vert_pos, 1.0)
        return frag_color
