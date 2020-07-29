from dipter.shaders.lib import glsl_builtins as gl, vec
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
        c = vec.vec3(h,s,v)
        K = vec.vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0)
        p = torch.abs(gl.fract(vec.xxx(c) + vec.xyz(K)) * 6.0 - vec.www(K))
        return vec.z(c) * gl.mix(vec.xxx(K), torch.clamp(p - vec.xxx(K), 0.0, 1.0), vec.y(c))

    def shade_iter(self, frag_pos: Tensor, h: Tensor, s: Tensor, v: Tensor) -> Tensor:
        K = torch.tensor((1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0))
        p = torch.abs(gl.fract(torch.stack([h, h, h]) + K[0:3]) * torch.tensor(6.0) - torch.stack([K[3], K[3], K[3]]))
        color = v * gl.mix(torch.stack([K[0], K[0], K[0]]), torch.clamp(p - torch.stack([K[0], K[0], K[0]]), 0, 1.0), s)
        return color
