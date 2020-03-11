from src.shaders.shader_super import Shader


class ColorShader(Shader):

    def __init__(self):
        self.VERTEX_SHADER_FILENAME = "color_shader_vert.glsl"
        self.FRAGMENT_SHADER_FILENAME = "color_shader_frag.glsl"

