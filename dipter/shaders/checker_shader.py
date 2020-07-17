import dipter.shaders.lib.glsl_builtins as gl
from dipter.shaders.shader_super import *


class CheckerShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "checker_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Color1", "color1", DataType.Vec3_RGB, (0, 1), torch.ones(3)),
            ShaderInputParameter("Color2", "color2", DataType.Vec3_RGB, (0, 1), torch.zeros(3)),
            ShaderInputParameter("Scale", "scale", DataType.Float, (0, 100), 10.0)
        ]

    def shade_mat(self, frag_pos: Tensor, color1: Tensor, color2: Tensor, scale: Tensor) -> Tensor:
        p = frag_pos * scale

        p_int = torch.abs(torch.floor(p))

        check = torch.eq(torch.eq(gl.mod(p_int[0], 2.0), gl.mod(p_int[1], 2.0)), gl.mod(p_int[2], 2.0))
        if check:
            return color1
        else:
            return color2

    def shade_iter(self, frag_pos: Tensor, color1: Tensor, color2: Tensor, scale: Tensor) -> Tensor:
        p = frag_pos * scale

        p_int = torch.abs(torch.floor(p))

        check = torch.eq(torch.eq(gl.mod(p_int[0], 2.0), gl.mod(p_int[1], 2.0)), gl.mod(p_int[2], 2.0))
        if check:
            return color1
        else:
            return color2

        return gl.mix(color2, color1, check)
