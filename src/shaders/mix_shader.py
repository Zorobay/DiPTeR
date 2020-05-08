from src.shaders.lib import glsl_builtins as gl
from src.shaders.shader_super import *


class MixShader(Shader):
    FRAGMENT_SHADER_FILENAME = "mix_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [
            ("X", "x", INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((1.0, 1.0, 1.0))),
            ("Y", "y", INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((0.0, 0.0, 0.0))),
            ("Factor", "a", INTERNAL_TYPE_FLOAT, (0, 1), 0.5)
        ]

    def shade_torch(self, vert_pos: Tensor, x: Tensor, y: Tensor, a: Tensor) -> Tensor:
        return gl.mix(x, y, a)
