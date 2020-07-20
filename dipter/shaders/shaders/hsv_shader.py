from dipter.shaders.lib import glsl_builtins as gl
from dipter.shaders.shader_super import *


class HSVShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "hsv_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "hsv"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Hue", "h", DataType.Float, (0, 1), 1.0),
            ShaderInputParameter("Saturation", "s", DataType.Float, (0, 1), 1.0),
            ShaderInputParameter("Value", "v", DataType.Float, (0, 1), 1.0)
        ]

    def shade_mat(self, h: Tensor, s: Tensor, v: Tensor) -> Tensor:
        Wi, He = Shader.render_size()
        c = torch.tensor(6.0).repeat(Wi, He, 1)
        K = torch.tensor((1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0)).repeat(Wi, He, 1)
        K0 = K[:, :, 0].unsqueeze(-1).repeat(1, 1, 3)

        p = torch.abs(gl.fract(h.repeat(1, 1, 3) + K[:, :, 0:3]) * c - K[:, :, 3].unsqueeze(-1).repeat(1, 1, 3))
        color = v * gl.mix(K0, torch.clamp(p - K0, 0., 1.), s)
        return color

    def shade_iter(self, frag_pos: Tensor, h: Tensor, s: Tensor, v: Tensor) -> Tensor:
        K = torch.tensor((1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0))
        p = torch.abs(gl.fract(torch.stack([h, h, h]) + K[0:3]) * torch.tensor(6.0) - torch.stack([K[3], K[3], K[3]]))
        color = v * gl.mix(torch.stack([K[0], K[0], K[0]]), torch.clamp(p - torch.stack([K[0], K[0], K[0]]), 0, 1.0), s)
        return color
