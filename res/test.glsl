#version 430

in vec3 vert_pos;

uniform vec3 shader;

out vec4 frag_color;



//---- gradient_shader_frag.gradient ----
vec3 gradient_shader_frag_gradient(vec3 vert_pos) {
    return vec3(vert_pos.x);
}

void main() {
    vec3 shader = gradient_shader_frag_gradient(vert_pos);
    frag_color = vec4(shader, 1.0);
}
