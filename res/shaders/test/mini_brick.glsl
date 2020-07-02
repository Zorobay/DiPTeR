#version 430

#import "trig.glsl"
#import "pattern.glsl"

vec3 mini_brick_shade(vec3 frag_pos, float mortar_scale, float brick_scale)
{
    vec2 uv = frag_pos.xy;
    vec3 uv3 = frag_pos;

    float brick_elongate = 2.0;
    float brick_shift = 0.5;
    vec3 color_brick = vec3(0.69, 0.25, 0.255);
    vec3 color_mortar = vec3(0.9,0.9,0.9);

    uv3 = tile(uv3, vec3(brick_scale/brick_elongate, brick_scale, brick_scale), vec2(brick_shift,0.0));
    return mix(color_mortar, color_brick, box(uv3.xy, vec2(mortar_scale, mortar_scale)));
}
