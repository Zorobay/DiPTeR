#version 430

varying vec3 v_color;
uniform float rot;
uniform vec3 color1;
uniform vec3 color2;

void main() {
    float x = v_color.x;
    float y = (v_color.y + 1.0)/2.0;
    vec3 x_color = mix(color1, color2, x) * rot;
    vec3 y_color = mix(color1, color2, y) * (1.0-rot);

    gl_FragColor = vec4(x_color + y_color, 1.0);
}
