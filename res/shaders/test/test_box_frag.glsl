#version 120

uniform float width;
uniform float height;
//uniform vec2 size;

in vec3 frag_pos;
out vec4 frag_color;

#import "pattern.glsl"

void main() {
    vec3 bg_color = vec3(1.0,1.0,1.0);
    vec3 box_color = vec3(0.,0.,0.);
    frag_color = vec4(mix(bg_color, box_color, box(frag_pos.xy, vec2(width, height))), 1.0);
}
