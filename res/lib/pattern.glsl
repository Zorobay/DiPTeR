#version 430

float box(vec2 coord, vec2 size) {
    edge_smooth = vec2(0.0001);
    // Normalize
    size = vec2(0.5) - size * 0.5;
    vec2 uv = smoothstep(size, size + edge_smooth, coord);
    uv *= smoothstep(size, size + edge_smooth, vec2(1.0) - coord);
    return uv.x * uv.y;
}
