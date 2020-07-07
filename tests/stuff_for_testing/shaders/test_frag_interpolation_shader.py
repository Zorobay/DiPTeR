from numpy.core.multiarray import ndarray

from src.shaders.shader_super import *


class TestFragInterpolationShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "test/test_frag_interpolation_frag.glsl"
    SHADER_FILENAME = "vertex_shader.glsl"

    def __init__(self):
        super().__init__()


    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return []

    def shade_iter(self, frag_pos: ndarray, *args) -> ndarray:
        frag_color = None

        return frag_color

    def shade_iter(self, frag_pos, *args):
        return frag_pos
