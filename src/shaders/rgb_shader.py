from src.shaders.shader_super import *


class RGBShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "rgb_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "rgb"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, DataType, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Red", "red", DataType.INTERNAL_TYPE_FLOAT, (0, 1), 0.0),
            ("Green", "green", DataType.INTERNAL_TYPE_FLOAT, (0, 1), 0.0),
            ("Blue", "blue", DataType.INTERNAL_TYPE_FLOAT, (0, 1), 0.0)
        ]

    def shade_mat(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.cat((red, green, blue), dim=2)

    def shade(self, frag_pos: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.stack((red, green, blue))
