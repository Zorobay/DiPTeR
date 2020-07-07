from src.shaders.shader_super import *


class MaterialOutputShader(CompilableShader):
    FRAGMENT_SHADER_FILENAME = "material_output_shader_frag.glsl"

    def get_inputs(self) -> typing.List[ShaderInput]:
        return [
            ShaderInput("Shader", "shader", DataType.Shader, (0, 1), torch.tensor((1.0, 1.0, 1.0)))
        ]

    def get_outputs(self) -> typing.List[typing.Tuple[str, str]]:
        return []

    def shade_iter(self, frag_pos: Tensor, shader: Tensor) -> Tensor:
        return shader

    def shade_mat(self, shader: Tensor) -> Tensor:
        return shader
