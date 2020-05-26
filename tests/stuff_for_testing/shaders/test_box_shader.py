from numpy.core.multiarray import ndarray

from src.shaders.lib.pattern import *
from src.shaders.shader_super import *


class TestBoxShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "test/test_box_frag.glsl"

    def __init__(self):
        super().__init__()

    def shade(self, frag_pos: ndarray, *args) -> ndarray:
        return np.zeros((4))

    def shade(self, frag_pos: torch.Tensor, width: Tensor, height: Tensor) -> torch.Tensor:
        bg_color = torch.tensor((1., 1., 1.))
        box_color = torch.tensor((0., 0., 0.))
        frag_color = gl.mix(bg_color, box_color, box(frag_pos[:2], torch.stack([width, height])))
        return frag_color

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            # ("Size", "size", INTERNAL_TYPE_ARRAY_FLOAT, (0, 1), torch.tensor((0.5, 0.5)))
            ("Width", "width", INTERNAL_TYPE_FLOAT, (0, 1), 0.5),
            ("Height", "height", INTERNAL_TYPE_FLOAT, (0, 1), 0.5)
        ]
