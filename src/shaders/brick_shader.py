import typing
import torch
import autograd.numpy as anp
from numpy import ndarray
from torch import Tensor, tensor

import src.shaders.lib.glsl_builtins as gl
import src.shaders.lib.glgl_builtins_torch as to
from src.opengl.shader_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGBA
from src.shaders.lib.pattern_torch import box
from src.shaders.shader_super import Shader, TINY_FLOAT


class BrickShader(Shader):
    VERTEX_SHADER_FILENAME = "vertex_shader.glsl"
    FRAGMENT_SHADER_FILENAME = "brick_shader_frag.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Mortar Scale", "mortar_scale", INTERNAL_TYPE_FLOAT, (0.0, 1.0), 0.85),
            ("Brick Scale", "brick_scale", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 10.0),
            ("Brick Elongate", "brick_elongate", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 2.0),
            ("Brick Shift", "brick_shift", INTERNAL_TYPE_FLOAT, (0., 1.), 0.5),
            ("Brick Color", "color_brick", INTERNAL_TYPE_ARRAY_RGBA, (0, 0), anp.array((0.69, 0.25, 0.255, 1.))),
            ("Mortar Color", "color_mortar", INTERNAL_TYPE_ARRAY_RGBA, (0, 0), anp.array((0.9, 0.9, 0.9, 1.)))
        ]

    def _brickTile(self, tile: ndarray, scale: ndarray, shift: float):
        tile[0] *= scale[0]
        tile[1] *= scale[1]
        tile[2] *= scale[2]

        st: float = gl.step(1.0, anp.mod(tile[1], 2.0))
        tile[0] += shift * st

        return gl.fract(tile)

    def _brickTileTorch(self, tile: Tensor, scale: Tensor, shift: Tensor):
        tile[0] *= scale[0]
        tile[1] *= scale[1]
        tile[2] *= scale[2]

        st: Tensor = to.step(1.0, torch.fmod(tile[1], 2.0))
        tile[0] += shift * st

        return to.fract(tile)

    def shade(self, vert_pos: ndarray, mortar_scale: float, brick_scale: float, brick_elongate: float, brick_shift: float,
              color_brick: ndarray, color_mortar: ndarray) -> ndarray:
        # Increment each divisor by TINY_FLOAT to prevent DivisionByZeroErrors
        brick_elongate += TINY_FLOAT

        uv3 = vert_pos
        uv3 = self._brickTile(uv3, anp.array((brick_scale / brick_elongate, brick_scale, brick_scale)), brick_shift)
        b = box(uv3[:2], anp.array((mortar_scale, mortar_scale)), 0.0)
        frag_color = gl.mix(color_mortar, color_brick, b)
        return frag_color

    def shade_torch(self, vert_pos: Tensor, mortar_scale: Tensor, brick_scale: Tensor, brick_elongate: Tensor, brick_shift: Tensor,
                    color_brick: Tensor, color_mortar: Tensor) -> Tensor:
        brick_elongate_ = brick_elongate + TINY_FLOAT

        uv3 = vert_pos
        uv3 = self._brickTileTorch(uv3, Tensor((brick_scale / brick_elongate_, brick_scale, brick_scale)), brick_shift)
        b = box(uv3[0:2], tensor((mortar_scale, mortar_scale)), tensor(0.0))
        frag_color = to.mix(color_mortar, color_brick, b)
        return frag_color
