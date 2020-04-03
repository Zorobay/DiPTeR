#version 430

uniform vec2 edge0;
uniform vec2 edge1;

in vec3 vert_pos;
out vec4 frag_color;

void main() {
    vec2 red = smoothstep(edge0, edge1, vert_pos.xy); // interpolate red for 0.5<x<1.0
    frag_color = vec4(red, 0.,1.0);
}
