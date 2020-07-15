from src.shaders.lib import blender
from src.shaders.lib import glsl_builtins as gl
from src.shaders.shader_super import *


class BlenderBrickShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "blender_brick_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "blender_brick_shade"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Brick Color 1", "brick_color1", dtype=DataType.Vec3_RGB, limits=(0, 1), default=torch.tensor((0.6, 0.1, 0.1))),
            ShaderInputParameter("Brick Color 2", "brick_color2", dtype=DataType.Vec3_RGB, limits=(0, 1), default=torch.tensor((0.6, 0.5, 0.5))),
            ShaderInputParameter("Mortar Color", "mortar_color", dtype=DataType.Vec3_RGB, limits=(0, 1), default=torch.tensor((0.5, 0.5, 0.5))),
            ShaderInputParameter("Scale", "scale", dtype=DataType.Float, limits=(0, 100), default=2),
            ShaderInputParameter("Mortar Size", "mortar_size", dtype=DataType.Float, limits=(0, 0.125), default=0.02),
            ShaderInputParameter("Mortar Smooth", "mortar_smooth", dtype=DataType.Float, limits=(0, 1), default=0.1),
            ShaderInputParameter("Color Bias", "bias", dtype=DataType.Float, limits=(-1, 1), default=0.0),
            ShaderInputParameter("Brick Width", "brick_width", dtype=DataType.Float, limits=(0.01, 100), default=0.5),
            ShaderInputParameter("Row Height", "row_height", dtype=DataType.Float, limits=(0.01, 100), default=0.25),
            ShaderInputParameter("Horizontal Shift", "offset_amount", dtype=DataType.Float, limits=(0, 1), default=0.5),
            ShaderInputParameter("Shift Frequency", "offset_frequency", dtype=DataType.Int, limits=(1, 99), default=2),
            ShaderInputParameter("Brick Elongate", "squash_amount", dtype=DataType.Float, limits=(0, 100), default=1),
            ShaderInputParameter("Elongate Frequency", "squash_frequency", dtype=DataType.Int, limits=(1, 99), default=2)
        ]

    def get_outputs(self) -> typing.List[ShaderOutputParameter]:
        return [
            ShaderOutputParameter("Color", argument="color", dtype=DataType.Vec3_RGB),
            ShaderOutputParameter("Factor", argument="fac", dtype=DataType.Float)
        ]

    def shade_mat(self, brick_color1, brick_color2, mortar_color, scale, mortar_size, mortar_smooth, bias, brick_width, row_height,
                  offset_amount, offset_frequency, squash_amount, squash_frequency) -> typing.Tuple[Tensor, Tensor]:
        f2 = blender.calc_brick_texture(Shader.frag_pos() * scale,
                                        mortar_size,
                                        mortar_smooth,
                                        bias,
                                        brick_width,
                                        row_height,
                                        offset_amount,
                                        offset_frequency,
                                        squash_amount,
                                        squash_frequency)
        tint = f2[:, :, 0].unsqueeze(-1)
        f = f2[:, :, 1].unsqueeze(-1)

        brick_color1 = torch.where(f != 1.0, (1.0-tint)*brick_color1 + tint*brick_color2, brick_color1)

        color = gl.mix(brick_color1, mortar_color, f)
        fac = f
        return (color, fac)
