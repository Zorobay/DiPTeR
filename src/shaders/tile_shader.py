from src.shaders.lib import pattern
from src.shaders.shader_super import *


class TileShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "tile_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Scale", "scale", INTERNAL_TYPE_ARRAY_FLOAT, (0, 100), torch.tensor((1., 1., 1.))),
            ("Shift", "shift", INTERNAL_TYPE_ARRAY_FLOAT, (0, 1), torch.tensor((0., 0.)))
        ]

    # def shade(self, frag_pos: Tensor) -> Tensor:
    #     input = frag_pos[0]
    #     color = torch.stack((input, input, input))
    #     return color

    def shade_mat(self, frag_pos: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return pattern.tile(frag_pos, scale, shift)
