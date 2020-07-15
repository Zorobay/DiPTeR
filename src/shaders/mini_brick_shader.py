import src.shaders.lib.glsl_builtins as gl
from src.shaders.lib import pattern
from src.shaders.shader_super import *


class MiniBrickShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "test/mini_brick.glsl"
    FRAGMENT_SHADER_FUNCTION = "mini_brick_shade"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Mortar Scale", "mortar_scale", DataType.Float, (0.0, 1.0), 0.85),
            ShaderInputParameter("Brick Scale", "brick_scale", DataType.Float, (0.0, 100.0), 10.0)
        ]

    def shade_mat(self, mortar_scale: Tensor, brick_scale: Tensor) -> Tensor:
        brick_elongate = self.tensor(2.0)
        brick_shift = self.tensor(0.5)
        color_brick = self.tensor((0.69, 0.25, 0.255))
        color_mortar = self.tensor((0.9, 0.9, 0.9))

        scale = torch.cat([torch.div(brick_scale, brick_elongate + TINY_FLOAT), brick_scale, brick_scale], dim=2)
        uv3 = pattern.tile(Shader.frag_pos(), scale, torch.cat((brick_shift, torch.zeros_like(brick_shift)), dim=2))
        b = pattern.box(uv3[:, :, 0:2], torch.cat((mortar_scale, mortar_scale), dim=2))
        frag_color = gl.mix(color_mortar, color_brick, b)
        return frag_color
