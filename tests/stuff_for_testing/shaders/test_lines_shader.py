from src.shaders.shader_super import *


class TestLinesShader(Shader):

    FRAGMENT_SHADER_FILENAME = "test/test_lines_frag.glsl"
    VERTEX_SHADER_FILENAME = "vertex_shader.glsl"
    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return []

    def shade(self, vert_pos: ndarray, *args) -> ndarray:
        frag_color = None
        if vert_pos[0] > 0.05 and vert_pos[0] < 0.06:
            frag_color = anp.array((0., 1., 0., 1.))
        elif vert_pos[0] >= 0.1 and vert_pos[0] <= 0.2:
            frag_color = anp.array((1.0, 1., 0., 1.0))
        else:
            frag_color = anp.array((1., 1., 1., 1.))

        if vert_pos[1] >= 0.7 and vert_pos[1] <= 0.9:
            frag_color = anp.array((0., 0., 1., 1.))

        return frag_color

    def shade_torch(self, vert_pos, *args):
        frag_color = None

        if vert_pos[0] > 0.05 and vert_pos[0] < 0.06:
            frag_color = torch.tensor((0., 1., 0., 1.))
        elif vert_pos[0] >= 0.1 and vert_pos[0] <= 0.2:
            frag_color = torch.tensor((1.0, 1., 0., 1.0))
        else:
            frag_color = torch.tensor((1., 1., 1., 1.))

        if vert_pos[1] >= 0.7 and vert_pos[1] <= 0.9:
            frag_color = torch.tensor((0., 0., 1., 1.))

        return frag_color

