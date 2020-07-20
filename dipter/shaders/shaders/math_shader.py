from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter, ShaderOutputParameter


class ScalarMathShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "math_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "scalar_math"

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Operation", "operation", DataType.Int_Choice, (0, 3), 0, force_scalar=True, names=["Add", "Subtract", "Multiply", "Divide"]),
            ShaderInputParameter("Scalar", "scalar", DataType.Float, (-10000,10000), torch.tensor((0.))),
            ShaderInputParameter("Value", "value", DataType.Float, (-10000, 10000), torch.tensor((1.0)))
        ]

    def get_outputs(self) -> typing.List[ShaderOutputParameter]:
        return [
            ShaderOutputParameter("Scalar", dtype=DataType.Float)
        ]

    def shade_mat(self, operation: Tensor, scalar: Tensor, value: Tensor) -> Tensor:
        if operation == 0:
            return scalar + value
        elif operation == 1:
            return scalar - value
        elif operation == 2:
            return scalar * value
        elif operation == 3:
            return scalar / value

        return scalar
