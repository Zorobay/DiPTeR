#version 430

in vec3 vert_pos;

uniform float h;
uniform float s;
uniform float v;

out vec4 frag_color;

void main() {
    vec3 c = vec3(h,s,v);
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    frag_color = vec4(c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y), 1.0);
}
