#version 430

in vec3 vert_pos;

uniform vec3 color1;
uniform vec3 color2;
uniform float scale;

out vec4 frag_color;

void main() {
    vec3 p = vert_pos * scale;
    vec3 p_int = abs(floor(p));

    bool check = ((mod(p_int.x, 2) == mod(p_int.y, 2)) == bool(mod(p_int.z, 2)));
    vec3 color = mix(color2,color1,float(check));
    frag_color = vec4(color, 1.0);
}
