from numpy.core.multiarray import ndarray

from dipter.shaders.lib import glsl_builtins as gl
from dipter.shaders.shader_super import *


class TestSmoothstepShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "test/test_smoothstep_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Edge0", "edge0", INTERNAL_TYPE_ARRAY_FLOAT, (0, 1), torch.tensor((0.5, 0.5))),
            ("Edge1", "edge1", INTERNAL_TYPE_ARRAY_FLOAT, (0, 1), torch.tensor((1., 1.)))
        ]

    def shade_iter(self, frag_pos: ndarray, edge0, edge1) -> ndarray:
        frag_color = None

        return np.array((1., 1., 1.))

    def shade_iter(self, frag_pos: torch.Tensor, edge0: torch.Tensor, edge1: torch.Tensor) -> torch.Tensor:
        frag_color = None

        red = gl.smoothstep(edge0, edge1, frag_pos[:2])
        frag_color = torch.cat((red, torch.tensor([0.])))
        return frag_color
