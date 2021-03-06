from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter


class FragmentCoordinatesShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "frag_coord_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "fragCoord"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [

        ]

    def shade_mat(self) -> Tensor:
        return Shader.frag_pos()
