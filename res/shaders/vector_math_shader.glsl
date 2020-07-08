#version 120

vec3 vector_math(vec3 frag_pos, int operation, vec3 vector, vec3 value)
{
    if (operation == 0) {
        return vector + value;
    } else if (operation == 1) {
        return vector - value;
    } else if (operation == 2) {
        return vector * value;
    } else if (operation == 3) {
        return vector / value;
    }
    return vector;
}
