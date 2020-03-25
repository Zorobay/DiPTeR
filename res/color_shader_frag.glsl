#version 430


uniform vec4 color;
varying vec3 vert_pos;

void main() {
    gl_FragColor = color;
}
