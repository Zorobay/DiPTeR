#version 430

in vec3 frag_pos;

uniform vec3 scale;
uniform vec2 shift;

out vec4 frag_color;

#import "pattern.glsl"

void main() {
    vec3 tiled = tile(frag_pos, scale, shift);
    frag_color = vec4(tiled, 1.0);
}
