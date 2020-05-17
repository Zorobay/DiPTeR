import typing

from torch import Tensor

from src.shaders.shader_super import CompilableShader


class DefaultShader(CompilableShader):

    FRAGMENT_SHADER_FILENAME = "default_shader_frag.glsl"

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        return []

    def shade_mat(self, vert_pos: Tensor) -> Tensor:
        return vert_pos

    def shade(self, vert_pos: Tensor) -> Tensor:
        return vert_pos
