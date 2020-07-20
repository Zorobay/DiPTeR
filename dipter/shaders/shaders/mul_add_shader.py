from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter, ShaderOutputParameter


class MulAddShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "mul_add_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "mul_add"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Scalar", "scalar", DataType.Float, FLOAT_INTERVAL, 0.0),
            ShaderInputParameter("Multiply", "mul", DataType.Float, FLOAT_INTERVAL, 1.0),
            ShaderInputParameter("Add", "add", DataType.Float, FLOAT_INTERVAL, 0.0)
        ]

    def get_outputs(self) -> typing.List[ShaderOutputParameter]:
        return [
            ShaderOutputParameter("Scalar", DataType.Float)
        ]

    def shade_mat(self, scalar: Tensor, mul: Tensor, add: Tensor) -> Tensor:
        return scalar*mul + add
