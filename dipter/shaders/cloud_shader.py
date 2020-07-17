from dipter.shaders.lib import noise
from dipter.shaders.shader_super import *


class CloudShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "cloud_shader_frag.glsl"
    FRAGMENT_SHADER_FUNCTION = "cloud"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Scale", "scale", DataType.Float, (0, 100), 1.0),
            ShaderInputParameter("Detail", "detail", DataType.Int, (0, 10), 4.0, force_scalar=True)
        ]

    def get_outputs(self) -> typing.List[ShaderOutputParameter]:
        return [
            ShaderOutputParameter("Factor", DataType.Float, argument="out_factor"),
            ShaderOutputParameter("Color", DataType.Vec3_RGB, argument="out_color")
        ]

    def shade_mat(self, scale: Tensor, detail: Tensor) -> Tensor:
        w, h = Shader.render_size()
        uv = Shader.frag_pos()[:, :, :2]
        color = torch.tensor((0., 0., 0.)).repeat(w, h, 1)

        fBM = noise.fractalBrownianMotion(uv*scale, detail)
        color = color + fBM

        return (fBM, color)
