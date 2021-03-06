//Source: http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 hsv(vec3 frag_pos, float h, float s, float v) {
    vec3 c = vec3(h,s,v);
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
