#version 430

in vec3 vert_pos;

out vec4 frag_color;

void main() {
    frag_color = vec4(vert_pos, 1.0);
}
