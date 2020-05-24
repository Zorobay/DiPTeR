#version 430


vec3 gradient(vec3 vert_pos) {
    return vec3(1.-vert_pos.x);
}
/*
in vec3 vert_pos;

out vec4 frag_color;

void main() {
    float x = vert_pos.x;
    vec3 color = vec3(x);
    gl_FragColor = vec4(color, 1.0);
}
*/
