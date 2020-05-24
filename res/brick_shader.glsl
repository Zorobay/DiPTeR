#version 430

/*in vec3 vert_pos;

uniform float mortar_scale;
uniform float brick_scale;
uniform float brick_elongate;
uniform float brick_shift;
uniform vec3 color_brick;
uniform vec3 color_mortar;

out vec4 frag_color;*/

#import "trig.glsl"
#import "pattern.glsl"

vec3 brick_shade(vec3 vert_pos, float mortar_scale, float brick_scale, float brick_elongate, float brick_shift, vec3 color_brick, vec3 color_mortar)
{
    vec2 uv = vert_pos.xy;
    vec3 uv3 = vert_pos;

    uv3 = tile(uv3, vec3(brick_scale/brick_elongate, brick_scale, brick_scale), vec2(brick_shift,0.0));
    return mix(color_mortar, color_brick, box(uv3.xy, vec2(mortar_scale, mortar_scale)));
}

/*
void main()
{
    vec2 uv = vert_pos.xy;
    vec3 uv3 = vert_pos;

    uv3 = tile(uv3, vec3(brick_scale/brick_elongate, brick_scale, brick_scale), vec3(brick_shift,0.0,0.0));
    frag_color = vec4(mix(color_mortar, color_brick, box(uv3.xy, vec2(mortar_scale, mortar_scale))), 1.0);
}
*/
