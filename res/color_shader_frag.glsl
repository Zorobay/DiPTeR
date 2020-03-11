#version 430


uniform vec3 color;
varying vec3 vert_color;

void main() {
    gl_FragColor = vec4(vert_color, 1.0);
}
