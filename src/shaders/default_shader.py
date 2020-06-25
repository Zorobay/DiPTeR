from src.shaders.shader_super import *


class DefaultShader(CompilableShader):

    FRAGMENT_SHADER_FILENAME = "default_shader_frag.glsl"

    def get_inputs(self) -> typing.List[ShaderInput]:
        return []

    def shade_mat(self, frag_pos: Tensor) -> Tensor:
        return frag_pos

    def shade(self, frag_pos: Tensor) -> Tensor:
        return frag_pos
