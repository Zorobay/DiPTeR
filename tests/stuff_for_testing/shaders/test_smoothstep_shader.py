from src.shaders.shader_super import *


class TestSmoothstepShader(Shader):
    FRAGMENT_SHADER_FILENAME = "test/test_smoothstep_frag.glsl"
    VERTEX_SHADER_FILENAME = "test/vertex_shader.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return []

    def shade(self, vert_pos: ndarray, *args) -> ndarray:
        frag_color = None

        edge0 = 0.5
        edge1 = 1.0
        red = smoothstep(edge0, edge1, vert_pos[0])
        frag_color = anp.array((red, 0.5, 0., 1.0))
        return frag_color
