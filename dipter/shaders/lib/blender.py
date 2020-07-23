import torch

from dipter.shaders.lib import glsl_builtins as gl, noise, vec
from dipter.shaders.shader_super import TINY_FLOAT


def calc_brick_texture(p, mortar_size, mortar_smooth, bias, brick_width, row_height, offset_amount, offset_frequency, squash_amount,
                       squash_frequency):
    p_x = p[:, :, 0].unsqueeze(-1)
    p_y = p[:, :, 1].unsqueeze(-1)

    rownum = torch.floor(p_y / (row_height + TINY_FLOAT))

    brick_width_mult = torch.where(torch.fmod(rownum, squash_frequency) < 1.0, torch.tensor(1.0), squash_amount)
    brick_width = brick_width * brick_width_mult
    offset_mult = torch.where(torch.fmod(rownum, offset_frequency) < 1.0, torch.tensor(0.0), offset_amount)
    offset = brick_width * offset_mult

    bricknum = torch.floor((p_x + offset) / (brick_width + TINY_FLOAT))

    x = (p_x + offset) - brick_width * bricknum
    y = p_y - row_height * rownum

    row_brick_bias = rownum + bricknum
    tint = noise.random_float(row_brick_bias) + bias

    min_dist = torch.min(torch.min(x, y), torch.min(brick_width - x, row_height - y))

    res = torch.where(
        min_dist >= mortar_size, vec.cat([tint, vec.vec1(0.0)]),  #if
        torch.where(mortar_smooth == 0.0, vec.cat([tint, vec.vec1(0.0)]),  #else if
        vec.cat([tint, gl.smoothstep(0.0, mortar_smooth, 1.0-min_dist/(mortar_size+TINY_FLOAT))])) #else
    )
    return res
