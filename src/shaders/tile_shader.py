from src.shaders.lib import pattern
from src.shaders.shader_super import *


class TileShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "tile_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "tile_shade"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Scale", "scale", DataType.Vec3_Float, (0, 100), torch.tensor((1., 1., 1.))),
            ShaderInputParameter("Shift", "shift", DataType.Vec3_Float, (0, 1), torch.tensor((0., 0.)))
        ]

    def shade_iter(self, frag_pos: Tensor) -> Tensor:
        input = frag_pos[0]
        color = torch.stack((input, input, input))
        return color

    def shade_mat(self, frag_pos: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return pattern.tile(frag_pos, scale, shift)
