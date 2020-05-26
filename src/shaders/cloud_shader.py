from src.shaders.lib import noise
from src.shaders.shader_super import *


class CloudShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "cloud_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "cloud"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Scale", "scale", INTERNAL_TYPE_FLOAT, (0, 100), 1.0),
            ("Detail", "detail", INTERNAL_TYPE_INT, (0, 10), 4.0)
        ]

    def shade_mat(self, scale: Tensor, detail: Tensor) -> Tensor:
        w, h = Shader.width, Shader.height
        uv = Shader.frag_pos[:, :, :2]
        color = torch.tensor((0., 0., 0.)).repeat(w, h, 1)
        color = color + noise.fractalBrownianMotion(uv * scale, detail)

        return color

    # def shade(self, frag_pos: Tensor) -> Tensor:
    #     uv = frag_pos[:2]
    #     noise = noise.noise2D(uv)
    #     return torch.stack((noise, noise, noise))
