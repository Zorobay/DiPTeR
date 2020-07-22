from dipter.shaders.shader_super import *


class TestColorClippingShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "test/test_color_clipping.glsl"
    FRAGMENT_SHADER_FUNCTION = "test_color_clipping"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Mul", "mul", dtype=DataType.Float, limits=(-10, 10), default=10.),
            ShaderInputParameter("Shift", "shift", dtype=DataType.Float, limits=(-10, 10), default=5),
        ]

    def shade_mat(self, mul, shift):
        color = Shader.frag_pos() * mul - shift
        return color
