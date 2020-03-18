#version 430

in vec3 vert_pos;

uniform float mortar_scale;
uniform float brick_scale;
uniform float brick_elongate;
uniform float brick_shift;
uniform vec4 color_brick;
uniform vec4 color_mortar;

out vec4 frag_color;

#import "trig.glsl"
#import "pattern.glsl"

/*
vec3 rotateZ(vec3 color, float deg){
    float rad = deg_to_rad(deg);

    color.x -= 0.5;
    color.y -= 0.5;
    vec4 homo_color = vec4(color, 1.0);
    mat4 rotMat = mat4(cos(rad), -sin(rad), 0., 0.,
    sin(rad), cos(rad), 0., 0.,
    0., 0., 1., 0.,
    0., 0., 0., 1.);
    vec4 color_rot = rotMat * homo_color;
    color_rot.x += 0.5;
    color_rot.y += 0.5;
    return color_rot.xyz;
}
*/

vec3 brickTile(vec3 tile, vec3 scale, float shift){
    // tiles (repeats) the coordinates in 'tile' and scales them down by a factor in 'scale'
    tile.x *= scale.x;
    tile.y *= scale.y;
    tile.z *= scale.z;

    float st = step(1.0, mod(tile.y,2.0));
    tile.x += shift*st;

    return fract(tile);
}

void main()
{
    vec2 uv = vert_pos.xy;
    vec3 uv3 = vert_pos;

    uv3 = brickTile(uv3, vec3(brick_scale/brick_elongate, brick_scale, brick_scale), brick_shift);
    frag_color = mix(color_mortar, color_brick, box(uv3.xy, vec2(mortar_scale, mortar_scale), 0.0));
}
