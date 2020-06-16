from src.shaders.lib import glsl_builtins as gl
from src.shaders.shader_super import *


class MixShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "mix_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "mix"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, DataType, typing.Tuple[float, float], typing.Any]]:
        return [
            ("X", "x", DataType.INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((1.0, 1.0, 1.0))),
            ("Y", "y", DataType.INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((0.0, 0.0, 0.0))),
            ("Factor", "a", DataType.INTERNAL_TYPE_FLOAT, (0, 1), 0.5)
        ]

    def shade_mat(self, x: Tensor, y: Tensor, a: Tensor) -> Tensor:
        return gl.mix(x, y, a)

    def shade(self, frag_pos: Tensor, x: Tensor, y: Tensor, a: Tensor) -> Tensor:
        return gl.mix(x, y, a)
