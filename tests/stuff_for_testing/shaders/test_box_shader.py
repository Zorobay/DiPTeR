from numpy.core.multiarray import ndarray

from src.shaders.lib import glgl_builtins_torch as gl
from src.shaders.lib.pattern_torch import *
from src.shaders.shader_super import *


class TestBoxShader(Shader):
    FRAGMENT_SHADER_FILENAME = "test/test_box_frag.glsl"

    def __init__(self):
        super().__init__()

    def shade(self, vert_pos: ndarray, *args) -> ndarray:
        return np.zeros((4))

    def shade_torch(self, vert_pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        bg_color = torch.tensor((1., 1., 1.))
        box_color = torch.tensor((0., 0., 0.))
        frag_color = torch.cat((gl.mix(bg_color, box_color, box(vert_pos[:2], size, torch.tensor(0.0))), torch.tensor([1.])))
        return frag_color

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Size", "size", INTERNAL_TYPE_ARRAY_FLOAT, (0, 1), (0.5, 0.5))
        ]
