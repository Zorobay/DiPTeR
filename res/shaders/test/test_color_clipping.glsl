
vec3 test_color_clipping(vec3 frag_pos, float mul, float shift) {
    vec3 color = vec3(frag_pos)*mul - shift;
    return color;
}
