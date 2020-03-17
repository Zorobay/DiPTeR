#version 430


uniform vec3 color;
varying vec3 vert_pos;

void main() {
    gl_FragColor = vec4(vec3(-1.,0.,0.), 1.0);
}
