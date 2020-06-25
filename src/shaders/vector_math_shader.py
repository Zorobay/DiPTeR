from src.shaders.shader_super import *


class VectorMathShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "vector_math_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "vectorMath"

    def get_inputs(self) -> typing.List[ShaderInput]:
        return [
            ShaderInput("Vector", "vector", DataType.Vec3_Float, (0, 1), torch.tensor((0., 0., 0.))),
            ShaderInput("Multiplier", "mult", DataType.Vec3_Float, (-1000, 1000), torch.tensor((1.0, 1.0, 1.0)))
        ]

    def shade_mat(self, vector: Tensor, mult: Tensor) -> Tensor:
        return vector * mult
