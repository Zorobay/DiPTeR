#version 430

in vec3 vert_pos;

uniform vec3 color1;
uniform vec3 color2;
uniform float scale;

out vec4 frag_color;

void main() {
    vec3 p = vert_pos * scale;

    int xi = int(abs(floor(p.x)));
    int yi = int(abs(floor(p.y)));
    int zi = int(abs(floor(p.z)));

    bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));
    vec3 color = check ? color1 : color2;
    frag_color = vec4(color, 1.0);
}
