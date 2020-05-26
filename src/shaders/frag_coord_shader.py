import src.opengl.internal_types
from src.shaders.shader_super import *


class FragmentCoordinatesShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "frag_coord_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "fragCoord"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return [

        ]

    def shade_mat(self) -> Tensor:
        return Shader.frag_pos