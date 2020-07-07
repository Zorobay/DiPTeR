from numpy.core.multiarray import ndarray

from src.shaders.shader_super import *


class TestLinesShader(FunctionShader):

    FRAGMENT_SHADER_FILENAME = "test/test_lines_frag.glsl"
    SHADER_FILENAME = "vertex_shader.glsl"
    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return []

    def shade_iter(self, frag_pos: ndarray, *args) -> ndarray:
        frag_color = None
        return None

    def shade_iter(self, frag_pos, *args):
        frag_color = None

        if frag_pos[0] > 0.05 and frag_pos[0] < 0.06:
            frag_color = torch.tensor((0., 1., 0.))
        elif frag_pos[0] >= 0.1 and frag_pos[0] <= 0.2:
            frag_color = torch.tensor((1.0, 1., 0.))
        else:
            frag_color = torch.tensor((1., 1., 1.))

        if frag_pos[1] >= 0.7 and frag_pos[1] <= 0.9:
            frag_color = torch.tensor((0., 0., 1.))

        return frag_color

