#version 430

in vec3 vert_pos;

out vec4 frag_color;

void main() {
    float x = vert_pos.x;
    vec3 color = vec3(x);
    gl_FragColor = vec4(color, 1.0);
}
