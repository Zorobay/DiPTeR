from src.shaders.shader_super import *


class GradientShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "gradient_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "gradient"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
        ]

    # def shade(self, vert_pos: Tensor) -> Tensor:
    #     input = vert_pos[0]
    #     color = torch.stack((input, input, input))
    #     return color

    def shade_mat(self) -> Tensor:
        vert_pos = Shader.vert_pos
        return vert_pos[:, :, 0].unsqueeze(-1).repeat(1, 1, 3)

    def shade(self, vert_pos: Tensor) -> Tensor:
        x = vert_pos[:, :, 1]
        color = x.unsqueeze(-1).repeat(1, 1, 3)
        return color
