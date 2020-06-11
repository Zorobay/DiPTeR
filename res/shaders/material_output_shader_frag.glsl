#version 430

in vec3 frag_pos;

uniform vec3 shader;

out vec4 frag_color;


void main() {
    frag_color = vec4(shader, 1.0);
}
