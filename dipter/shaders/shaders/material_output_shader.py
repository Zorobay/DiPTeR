from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter


class MaterialOutputShader(CompilableShader):
    FRAGMENT_SHADER_FILENAME = "material_output_shader_frag.glsl"

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Shader", "shader", DataType.Shader, (0, 1), torch.tensor((1.0, 1.0, 1.0)))
        ]

    def get_outputs(self) -> typing.List[typing.Tuple[str, str]]:
        return []

    def shade_iter(self, frag_pos: Tensor, shader: Tensor) -> Tensor:
        return shader

    def shade_mat(self, shader: Tensor) -> Tensor:
        return shader
