#version 430

uniform mat4 object_to_world;
uniform mat4 world_to_view;
uniform mat4 view_to_projection;

in vec3 in_vert_pos;
out vec3 frag_pos;

void main() {
    frag_pos = (in_vert_pos + 1.0)/2.;
    gl_Position = view_to_projection * world_to_view * object_to_world * vec4(in_vert_pos, 1.0);
}
