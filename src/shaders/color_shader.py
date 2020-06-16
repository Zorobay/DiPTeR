from src.shaders.shader_super import *


class ColorShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, DataType, typing.Tuple[float, float], typing.Any]]:
        return [
            ("Color", "color", DataType.INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((1., 1., 1.)))
        ]

    def shade(self, frag_pos: Tensor, color: Tensor) -> Tensor:
        return color
