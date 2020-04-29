#version 430

in vec3 vert_pos;

uniform float mortar_scale;
uniform float brick_scale;
uniform float brick_elongate;
uniform float brick_shift;
uniform vec3 color_brick;
uniform vec3 color_mortar;

out vec4 frag_color;

#import "trig.glsl"
#import "pattern.glsl"


vec3 brickTile(vec3 tile, vec3 scale, float shift){
    // tiles (repeats) the coordinates in 'tile' and scales them down by a factor in 'scale'
    tile *= scale;

    float st = step(1.0, mod(tile.y,2.0));
    tile.x += shift*st;

    return fract(tile);
}

void main()
{
    vec2 uv = vert_pos.xy;
    vec3 uv3 = vert_pos;

    uv3 = brickTile(uv3, vec3(brick_scale/brick_elongate, brick_scale, brick_scale), brick_shift);
    frag_color = vec4(mix(color_mortar, color_brick, box(uv3.xy, vec2(mortar_scale, mortar_scale))), 1.0);
}
