#version 430

#import "pattern.glsl"

vec3 tile_shade(vec3 frag_pos, vec3 scale, vec2 shift)
{
    vec3 tiled = tile(frag_pos, scale, shift);
    return tiled;
}
