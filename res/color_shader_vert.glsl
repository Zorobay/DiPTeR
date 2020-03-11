#version 430

uniform mat4 object_to_world;
uniform mat4 world_to_view;
uniform mat4 view_to_projection;

attribute vec3 in_vert_pos;
varying vec3 vert_color;

void main() {
    vert_color = in_vert_pos;
    gl_Position = view_to_projection * world_to_view * object_to_world * vec4(in_vert_pos, 1.0);
}
