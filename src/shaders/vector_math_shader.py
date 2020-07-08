from src.shaders.shader_super import *


class VectorMathShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "vector_math_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "vector_math"

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Operation", "operation", DataType.Int_Choice, (0, 3), 0, force_scalar=True, names=["Add", "Subtract", "Multiply", "Divide"]),
            ShaderInputParameter("Vector", "vector", DataType.Vec3_Float, (0, 1), torch.tensor((0., 0., 0.))),
            ShaderInputParameter("Value", "value", DataType.Vec3_Float, (-10000, 10000), torch.tensor((1.0, 1.0, 1.0))),
        ]

    def shade_mat(self, operation: Tensor, vector: Tensor, value: Tensor) -> Tensor:
        if operation == 0:
            return vector + value
        elif operation == 1:
            return vector - value
        elif operation == 2:
            return vector * value
        elif operation == 3:
            return vector / value

        return vector
