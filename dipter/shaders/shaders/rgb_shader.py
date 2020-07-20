from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter


class RGBShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "rgb_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "rgb"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Red", "red", DataType.Float, (0, 1), 0.0),
            ShaderInputParameter("Green", "green", DataType.Float, (0, 1), 0.0),
            ShaderInputParameter("Blue", "blue", DataType.Float, (0, 1), 0.0)
        ]

    def shade_mat(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.cat((red, green, blue), dim=2)

    def shade_iter(self, frag_pos: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.stack((red, green, blue))
