#version 430

in vec3 vert_pos;

uniform vec3 x;
uniform vec3 y;
uniform float a;

out vec4 frag_color;

void main() {
    frag_color = vec4(mix(x,y,a), 1.0);
}
