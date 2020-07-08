from src.shaders.shader_super import *


class GradientShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "gradient_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "gradient"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
        ]

    def shade_iter(self, frag_pos: Tensor) -> Tensor:
        input = frag_pos[0]
        color = torch.stack((input, input, input))
        return color

    def shade_mat(self) -> Tensor:
        frag_pos = Shader.frag_pos
        return frag_pos[:, :, 0].unsqueeze(-1).repeat(1, 1, 3)
