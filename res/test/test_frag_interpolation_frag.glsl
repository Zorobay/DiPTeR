#version 430

in vec3 frag_pos;
out vec4 frag_color;

void main() {
    frag_color = vec4(frag_pos, 1.0);
}
