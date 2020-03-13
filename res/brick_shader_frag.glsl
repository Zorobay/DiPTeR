#version 430

varying vec3 vert_pos;

uniform float mortar_scale;
uniform float brick_scale;
uniform float brick_elongate;

#define PI 3.1415926535897932384626433832795

float deg_to_rad(float deg){
    return (PI / 180.) * deg;
}

float box(vec2 coord, vec2 size, float edge_smooth){
    // Returns 1.0 if this coordinate belongs to the box, 0.0 otherwise (with some interpolation based on edge_smooth)

    // Results are undefined for smoothstep if edge0 == edge1, therefore we add a small term to edge_smooth
    edge_smooth = clamp(edge_smooth, 0.00001, 1.0);

    // Invert and normalize scaling so that (1.0,1.0) corresponds to largest size and (0.,0.) to smallest
    size = vec2(.5) - size*.5;
    // Create the upper corner
    vec2 bx = smoothstep(size, size+edge_smooth, vec2(1.0)- coord);
    // Create the lower corner
    bx *= smoothstep(size, size+edge_smooth, coord);
    return bx.x * bx.y;
}

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

vec3 brickTile(vec3 tile, vec3 scale, float shift){
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
    vec3 uv3 = vert_pos.xyz;

    uv3 = brickTile(uv3, vec3(brick_scale/brick_elongate, brick_scale, brick_scale), 0.5);// Tile the color into 4

    vec3 fg = vec3(0.5, 0.1, 0.0);
    vec3 bg = vec3(0.1, 0.1, 0.1);
    gl_FragColor = vec4(mix(bg, fg, box(uv3.xy, vec2(mortar_scale, mortar_scale), 0.0)), 1.0);
}
