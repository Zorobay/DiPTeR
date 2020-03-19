#version 430

in vec3 vert_pos;
out vec4 frag_color;

void main() {
    float edge0 = 0.5;
    float edge1 = 1.0;
    float red = smoothstep(edge0, edge1, vert_pos.x); // interpolate red for 0.5<x<1.0
    frag_color = vec4(red, 0.5,0.,1.0);
}
