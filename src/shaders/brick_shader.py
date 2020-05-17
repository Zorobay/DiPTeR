import typing
import torch
from torch import Tensor

import src.shaders.lib.glsl_builtins as gl
from src.opengl.internal_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGB
from src.shaders.lib.pattern import box
from src.shaders.shader_super import FunctionShader, TINY_FLOAT


class BrickShader(FunctionShader):
    SHADER_FILENAME = "vertex_shader.glsl"
    FRAGMENT_SHADER_FILENAME = "brick_shader.glsl"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], float]]:
        return [
            ("Mortar Scale", "mortar_scale", INTERNAL_TYPE_FLOAT, (0.0, 1.0), 0.85),
            ("Brick Scale", "brick_scale", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 10.0),
            ("Brick Elongate", "brick_elongate", INTERNAL_TYPE_FLOAT, (0.0, 100.0), 2.0),
            ("Brick Shift", "brick_shift", INTERNAL_TYPE_FLOAT, (0., 1.), 0.5),
            ("Brick Color", "color_brick", INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((0.69, 0.25, 0.255))),
            ("Mortar Color", "color_mortar", INTERNAL_TYPE_ARRAY_RGB, (0, 1), torch.tensor((0.9, 0.9, 0.9)))
        ]

    def _brickTileTorch(self, tile: Tensor, scale: Tensor, shift: Tensor):
        tx = tile[0] * scale[0]
        ty = tile[1] * scale[1]
        tz = tile[2] * scale[2]

        st: Tensor = gl.step(1.0, torch.fmod(ty, 2.0))
        tx_shifted = tx + shift * st

        return gl.fract(torch.stack([tx_shifted, ty, tz]))

    def shade_mat(self, vert_pos: Tensor, mortar_scale: Tensor, brick_scale: Tensor, brick_elongate: Tensor, brick_shift: Tensor,
              color_brick: Tensor, color_mortar: Tensor) -> Tensor:

        scale = torch.stack([torch.div(brick_scale, brick_elongate + TINY_FLOAT), brick_scale, brick_scale])
        uv3 = self._brickTileTorch(vert_pos, scale, brick_shift)
        b = box(uv3[0:2], torch.stack((mortar_scale, mortar_scale)))
        frag_color = gl.mix(color_mortar, color_brick, b)
        return frag_color

    def shade(self, vert_pos: Tensor, mortar_scale: Tensor, brick_scale: Tensor, brick_elongate: Tensor, brick_shift: Tensor,
              color_brick: Tensor, color_mortar: Tensor) -> Tensor:

        scale = torch.stack([torch.div(brick_scale, brick_elongate + TINY_FLOAT), brick_scale, brick_scale])
        uv3 = self._brickTileTorch(vert_pos, scale, brick_shift)
        b = box(uv3[0:2], torch.stack((mortar_scale, mortar_scale)))
        frag_color = gl.mix(color_mortar, color_brick, b)
        return frag_color
