from src.shaders.shader_super import *


class ColorShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "shade_color"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInput]:
        return [
            ShaderInput("Color", "color", DataType.Vec3_RGB, (0, 1), torch.tensor((1., 1., 1.)))
        ]

    def shade_mat(self, color: Tensor) -> Tensor:
        return color
