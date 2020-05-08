import typing
import torch
from torch import Tensor, tensor
from src.shaders import lib
from src.shaders.lib.noise import Noise2D
from src.shaders.shader_super import *


class CloudShader(Shader):
    FRAGMENT_SHADER_FILENAME = "cloud_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Scale", "scale", INTERNAL_TYPE_FLOAT, (0,100), 1.0)
        ]

    def shade_torch(self, vert_pos: Tensor) -> Tensor:
        uv = vert_pos[:2]
        noise = Noise2D(uv)
        return torch.stack((noise, noise, noise))


