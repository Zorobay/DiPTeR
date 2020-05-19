#version 430

uniform mat4 object_to_world;
uniform mat4 world_to_view;
uniform mat4 view_to_projection;
uniform vec3 maxes;
uniform vec3 mins;

in vec3 in_vert_pos;
out vec3 vert_pos;

void main() {
    // Normalize vertex positions to [0,1]
    vert_pos = (in_vert_pos - mins)/(maxes-mins);
    gl_Position = view_to_projection * world_to_view * object_to_world * vec4(in_vert_pos, 1.0);
}
