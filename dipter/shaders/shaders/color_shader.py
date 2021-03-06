from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter


class ColorShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "shade_color"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Color", "color", DataType.Vec3_RGB, (0, 1), torch.tensor((1., 1., 1.)))
        ]

    def shade_mat(self, color: Tensor) -> Tensor:
        return color
