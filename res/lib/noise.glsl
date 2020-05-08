#version 430

vec2 simpleSmoothstep(vec2 p) {
    return p*p*(3. -2.*p);
}

float Noise2D(vec2 p) {
    return fract(sin(p.x * 888. + p.y * 5322.) * 3451.);
}

float SmoothNoise2D(vec2 p) {
    vec2 local_uv = simpleSmoothstep(fract(p));
    vec2 local_id = floor(p);

    // Find the noise value at the four corners of a local box
    float bl = Noise2D(local_id);
    float br = Noise2D(local_id + vec2(1,0));
    float tl = Noise2D(local_id + vec2(0,1));
    float tr = Noise2D(local_id + vec2(1,1));

    // Interpolate between bottom, top, and bottom -> top
    float b = mix(bl,br,local_uv.x);
    float t = mix(tl,tr,local_uv.x);

    return mix(b,t,local_uv.y);
}
