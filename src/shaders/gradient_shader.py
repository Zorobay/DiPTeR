import autograd.numpy as np

from src.shaders.glsl_builtins import mix
from src.shaders.shader_super import Shader


class GradientShader(Shader):

    def __init__(self):
        super().__init__()
        # Define names of linked GLSL shaders
        self.VERTEX_SHADER_FILENAME = "gradient_shader_vert.glsl"
        self.FRAGMENT_SHADER_FILENAME = "gradient_shader_frag.glsl"

    def shade(self, x, y, z):
        color_x = mix(self.color1, self.color2, x)
        color_y = mix(self.color1, self.color2, y)
        return self.fac * color_x + (1.0 - self.fac) * color_y


