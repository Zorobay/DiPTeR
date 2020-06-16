from src.shaders.shader_super import *


class VectorMathShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "vector_math_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "vectorMath"

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, DataType, typing.Tuple[float, float], Tensor]]:
        return [
            ("Vector", "vector", DataType.INTERNAL_TYPE_ARRAY_FLOAT, (0,1), torch.tensor((0.,0.,0.))),
            ("Multiplier", "mult", DataType.INTERNAL_TYPE_ARRAY_FLOAT, (-1000,1000), torch.tensor((1.0,1.0,1.0)))
        ]

    def shade_mat(self, vector: Tensor, mult: Tensor) -> Tensor:
        return vector * mult
