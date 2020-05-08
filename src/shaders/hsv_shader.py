from src.shaders.lib import glsl_builtins as gl
from src.shaders.shader_super import *


class HSVShader(Shader):
    FRAGMENT_SHADER_FILENAME = "hsv_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Hue", "h", INTERNAL_TYPE_FLOAT, (0, 1), 1.0),
            ("Saturation", "s", INTERNAL_TYPE_FLOAT, (0, 1), 1.0),
            ("Value", "v", INTERNAL_TYPE_FLOAT, (0, 1), 1.0)
        ]

    def shade_torch(self, vert_pos: Tensor, h: Tensor, s: Tensor, v: Tensor) -> Tensor:
        K = torch.tensor((1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0))
        p = torch.abs(gl.fract(torch.stack([h, h, h]) + K[0:3]) * torch.tensor(6.0) - torch.stack([K[3], K[3], K[3]]))
        color = v * gl.mix(torch.stack([K[0], K[0], K[0]]), torch.clamp(p - torch.stack([K[0], K[0], K[0]]), 0, 1.0), s)
        return color

