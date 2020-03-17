import typing

import autograd.numpy as anp
from numpy import ndarray

from src.opengl.shader_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_RGBA
from src.shaders.shader_super import Shader
import src.shaders.lib.glsl_builtins as gl


class BrickShader(Shader):

    def __init__(self):
        super().__init__()
        self.VERTEX_SHADER_FILENAME = "brick_shader_vert.glsl"
        self.FRAGMENT_SHADER_FILENAME = "brick_shader_frag.glsl"

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Mortar Scale", "mortar_scale", INTERNAL_TYPE_FLOAT, (0.0, 1.0), 0.9),
            ("Brick Scale", "brick_scale", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 10.0),
            ("Brick Elongate", "brick_elongate", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 2.0),
            ("Brick Color", "color_brick", INTERNAL_TYPE_RGBA, (0, 0), (175, 63, 65, 255)),
            ("Mortar Color", "color_mortar", INTERNAL_TYPE_RGBA, (0, 0), (230, 230, 230, 255))
        ]

    def _brickTile(self, tile: ndarray, scale: ndarray, shift: float):
        tile[0] *= scale[0]
        tile[1] *= scale[1]
        tile[2] *= scale[2]

        st: float = gl.step(1.0, anp.mod(tile[1], 2.0))
        tile[0] += shift * st

        return gl.fract(tile)

    def shade(self, vert_pos: ndarray, mortar_scale: float, brick_scale: float, brick_elongate: float, color_brick: anp.ndarray,
              color_mortar: anp.ndarray) -> ndarray:
        uv3 = vert_pos[0:3]
        uv3 = self._brickTile(uv3, anp.array((4.0, 4.0, 1.0)), 0.0)
        return anp.append(uv3, 1.0)
