import typing

import autograd.numpy as anp
from numpy import ndarray

import src.shaders.lib.glsl_builtins as gl
from src.opengl.shader_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_RGBA
from src.shaders.lib.pattern import box
from src.shaders.shader_super import Shader


class BrickShader(Shader):

    def __init__(self):
        super().__init__()
        self.VERTEX_SHADER_FILENAME = "brick_shader_vert.glsl"
        self.FRAGMENT_SHADER_FILENAME = "brick_shader_frag.glsl"

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Mortar Scale", "mortar_scale", INTERNAL_TYPE_FLOAT, (0.0, 1.0), 0.5),
            ("Brick Scale", "brick_scale", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 10.0),
            ("Brick Elongate", "brick_elongate", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 2.0),
            ("Brick Shift", "brick_shift", INTERNAL_TYPE_FLOAT, (0., 1.), 0.5),
            ("Brick Color", "color_brick", INTERNAL_TYPE_RGBA, (0, 0), anp.array((0.69, 0.25, 0.255, 1.))),
            ("Mortar Color", "color_mortar", INTERNAL_TYPE_RGBA, (0, 0), anp.array((0.9, 0.9, 0.9, 1.)))
        ]

    def _brickTile(self, tile: ndarray, scale: ndarray, shift: float):
        tile[0] *= scale[0]
        tile[1] *= scale[1]
        tile[2] *= scale[2]

        st: float = gl.step(1.0, anp.mod(tile[1], 2.0))
        tile[0] += shift * st

        return gl.fract(tile)

    def shade(self, vert_pos: ndarray, mortar_scale: float, brick_scale: float, brick_elongate: float, brick_shift: float,
              color_brick: ndarray, color_mortar: ndarray) -> ndarray:
        uv3 = vert_pos[0:3]

        brick_shift = 0.5
        uv3 = self._brickTile(uv3, anp.array((brick_scale / brick_elongate, brick_scale, brick_scale)), brick_shift)
        b = box(uv3[:2], anp.array((mortar_scale, mortar_scale)), 0.0)
        frag_color = gl.mix(color_mortar, color_brick, b)
        return frag_color
        # uv3 = self._brickTile(uv3, anp.array((brick_scale/brick_elongate, brick_scale, brick_scale)), 0.5)
        # frag_color = gl.mix(color_mortar, color_brick, gl.box())
        # return anp.append(uv3, 1.0)
