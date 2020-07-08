#version 120

float scalar_math(vec3 frag_pos, int operation, float scalar, float value)
{
    if (operation == 0) {
        return scalar + value;
    } else if (operation == 1) {
        return scalar - value;
    } else if (operation == 2) {
        return scalar * value;
    } else if (operation == 3) {
        return scalar / value;
    }
    return scalar;
}
