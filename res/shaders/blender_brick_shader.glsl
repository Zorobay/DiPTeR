#version 430

/*
Code adapted from Blender source: https://github.com/sobotka/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_tex_brick.glsl
*/
#import "noise.glsl"
#import "blender.glsl"

void blender_brick_shade(vec3 frag_pos,
vec3 brick_color1,
vec3 brick_color2,
vec3 mortar_color,
float scale,
float mortar_size,
float mortar_smooth,
float bias,
float brick_width,
float row_height,
float offset_amount,
float offset_frequency,
float squash_amount,
float squash_frequency,
out vec3 color,
out float fac)
{
    vec2 f2 = calc_brick_texture(frag_pos * scale,
    mortar_size,
    mortar_smooth,
    bias,
    brick_width,
    row_height,
    offset_amount,
    offset_frequency,
    squash_amount,
    squash_frequency);
    float tint = f2.x;
    float f = f2.y;

    if (f != 1.0) {
        float facm = 1.0 - tint;
        brick_color1 = facm * brick_color1 + tint * brick_color2;
    }

    color = mix(brick_color1, mortar_color, f);
    fac = f;
}
