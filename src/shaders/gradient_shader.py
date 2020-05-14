from src.shaders.shader_super import *


class GradientShader(Shader):
    FRAGMENT_SHADER_FILENAME = "gradient_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
        ]

    # def shade(self, vert_pos: Tensor) -> Tensor:
    #     x = vert_pos[0]
    #     color = torch.stack((x, x, x))
    #     return color

    def shade(self, vert_pos: Tensor) -> Tensor:
        x = vert_pos[:, :, 1]
        color = x.unsqueeze(-1).repeat(1, 1, 3)
        return color
