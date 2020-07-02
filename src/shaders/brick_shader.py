import src.shaders.lib.glsl_builtins as gl
from src.shaders.lib import pattern
from src.shaders.shader_super import *


class BrickShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "brick_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "brick_shade"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInput]:
        return [
            ShaderInput("Mortar Scale", "mortar_scale", DataType.Float, (0.0, 1.0), 0.85),
            ShaderInput("Brick Scale", "brick_scale", DataType.Float, (0.0, 100.0), 10.0),
            ShaderInput("Brick Elongate", "brick_elongate", DataType.Float, (0.0, 100.0), 2.0),
            ShaderInput("Brick Shift", "brick_shift", DataType.Float, (0., 1.), 0.5),
            ShaderInput("Brick Color", "color_brick", DataType.Vec3_RGB, (0, 1), torch.tensor((0.69, 0.25, 0.255))),
            ShaderInput("Mortar Color", "color_mortar", DataType.Vec3_RGB, (0, 1), torch.tensor((0.9, 0.9, 0.9)))
        ]

    def shade_mat(self, mortar_scale: Tensor, brick_scale: Tensor, brick_elongate: Tensor, brick_shift: Tensor,
                  color_brick: Tensor, color_mortar: Tensor) -> Tensor:
        scale = torch.cat([torch.div(brick_scale, brick_elongate + TINY_FLOAT), brick_scale, brick_scale], dim=2)
        uv3 = pattern.tile(Shader.frag_pos, scale, torch.cat((brick_shift, torch.zeros_like(brick_shift)), dim=2))
        b = pattern.box(uv3[:, :, 0:2], torch.cat((mortar_scale, mortar_scale), dim=2))
        frag_color = gl.mix(color_mortar, color_brick, b)
        return frag_color

