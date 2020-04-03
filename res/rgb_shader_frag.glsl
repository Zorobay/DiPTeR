#version 430

in vec3 vert_pos;

uniform float red;
uniform float green;
uniform float blue;

out vec4 frag_color;

void main() {
    frag_color = vec4(red, green, blue, 1.0);    
}
