#version 430

in vec3 frag_pos;

uniform vec3 shader;

out vec4 frag_color;



//---- gradient_shader_frag.gradient ----
vec3 gradient_shader_frag_gradient(vec3 frag_pos) {
    return vec3(frag_pos.x);
}

void main() {
    vec3 shader = gradient_shader_frag_gradient(frag_pos);
    frag_color = vec4(shader, 1.0);
}
