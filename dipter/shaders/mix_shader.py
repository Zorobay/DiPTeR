from dipter.shaders.lib import glsl_builtins as gl
from dipter.shaders.shader_super import *


class MixShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "mix_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "mix"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("X", "x", DataType.Vec3_RGB, (0, 1), torch.tensor((1.0, 1.0, 1.0))),
            ShaderInputParameter("Y", "y", DataType.Vec3_RGB, (0, 1), torch.tensor((0.0, 0.0, 0.0))),
            ShaderInputParameter("Factor", "a", DataType.Float, (0, 1), 0.5)
        ]

    def shade_mat(self, x: Tensor, y: Tensor, a: Tensor) -> Tensor:
        return gl.mix(x, y, a)

    def shade_iter(self, frag_pos: Tensor, x: Tensor, y: Tensor, a: Tensor) -> Tensor:
        return gl.mix(x, y, a)
