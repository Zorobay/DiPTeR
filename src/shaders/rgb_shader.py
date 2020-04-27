from src.shaders.shader_super import *


class RGBShader(Shader):
    FRAGMENT_SHADER_FILENAME = "rgb_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Red", "red", INTERNAL_TYPE_FLOAT, (0, 1), 0.0),
            ("Green", "green", INTERNAL_TYPE_FLOAT, (0, 1), 0.0),
            ("Blue", "blue", INTERNAL_TYPE_FLOAT, (0, 1), 0.0)
        ]

    def shade_torch(self, vert_pos: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.tensor((red, green, blue))
