#version 430


uniform vec3 color;
varying vec3 frag_pos;

void main() {
    gl_FragColor = vec4(color, 1.0);
}
