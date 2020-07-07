float box(vec2 coord, vec2 size) {
    vec2 edge_smooth = vec2(0.001);
    // Normalize
    size = vec2(0.5) - size * 0.5;
    vec2 uv = smoothstep(size, size + edge_smooth, coord);
    uv *= smoothstep(size, size + edge_smooth, vec2(1.0) - coord);
    return uv.x * uv.y;
}

vec3 tile(vec3 coord, vec3 scale, vec2 shift){
    // tiles (repeats) the coordinates in 'coord' and scales them down by a factor in 'scale'
    coord *= scale;

    vec2 st = step(1.0, mod(coord.xy,2.0));  // Find every second row/col
    coord.xy += shift*st.yx;

    return fract(coord);
}
