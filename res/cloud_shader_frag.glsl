#version 430

in vec3 vert_pos;

uniform float scale;

out vec4 frag_color;

#import "noise.glsl"

void main()
{
    vec2 uv = vert_pos.xy;
    float noise = SmoothNoise2D(uv*scale);
    vec3 color = vec3(noise);
    frag_color = vec4(color, 1.0);
}
