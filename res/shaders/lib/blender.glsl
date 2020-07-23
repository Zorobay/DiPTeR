// NEEDS noise.glsl

/*
Code adapted from Blender source: https://github.com/sobotka/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_tex_brick.glsl
*/
vec2 calc_brick_texture(vec3 p, float mortar_size, float mortar_smooth, float bias, float brick_width, float row_height, float offset_amount, float offset_frequency, float
squash_amount, float squash_frequency)
{
    float offset = 0.0;
    float x, y;

    float rownum = floor(p.y / row_height);

    brick_width *= (mod(rownum, squash_frequency) < 1.0) ? 1.0 : squash_amount;           /* squash */
    offset = (mod(rownum, offset_frequency) < 1.0) ? 0.0 : (brick_width * offset_amount); /* offset */

    float bricknum = floor((p.x + offset) / brick_width);

    x = (p.x + offset) - brick_width * bricknum;
    y = p.y - row_height * rownum;

    float row_brick_bias = rownum + bricknum;
    float tint = random_float(row_brick_bias) + bias;

    float min_dist = min(min(x, y), min(brick_width - x, row_height - y));
    if (min_dist >= mortar_size) {
        return vec2(tint, 0.0);
    }
    else if (mortar_smooth == 0.0) {
        return vec2(tint, 1.0);
    }
    else {
        min_dist = 1.0 - min_dist / mortar_size;
        return vec2(tint, smoothstep(0.0, mortar_smooth, min_dist));
    }

    //min_dist = 1.0 - min_dist / mortar_size;
    //return vec2(tint, smoothstep(0.0, mortar_smooth, min_dist));
}
