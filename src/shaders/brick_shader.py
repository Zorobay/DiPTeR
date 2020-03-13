from src.shaders.shader_super import Shader


class BrickShader(Shader):

    def __init__(self):
        super().__init__()
        self.VERTEX_SHADER_FILENAME = "brick_shader_vert.glsl"
        self.FRAGMENT_SHADER_FILENAME = "brick_shader_frag.glsl"
