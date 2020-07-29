vec2 simpleSmoothstep(vec2 p) {
    return p*p*(3. -2.*p);
}

float random_float(float p) {
    return fract(sin(p * 98.81) * 12355.68);
}

float random(vec2 p) {
    return random_float(p.x * 103. + p.y * 85.62);
}

float smoothNoise2D(vec2 p) {
    vec2 local_uv = simpleSmoothstep(fract(p));
    vec2 local_id = floor(p);

    // Find the noise value at the four corners of a local box
    float bl = random(local_id);
    float br = random(local_id + vec2(1,0));
    float tl = random(local_id + vec2(0,1));
    float tr = random(local_id + vec2(1,1));

    // Interpolate between bottom, top, and bottom -> top
    float b = mix(bl,br,local_uv.x);
    float t = mix(tl,tr,local_uv.x);
    return mix(b,t,local_uv.y);
}

// Implementation: https://thebookofshaders.com/13/
float fractalBrownianMotion(vec2 p, int detail)
{
    // Initial values
    float value = 0.0;
    float amplitude = 0.5;

    // Loop and add octaves of more and more detailed noise
    for (int i = 0; i < detail; i++) {
        value = value + (amplitude * smoothNoise2D(p));
        p = p * 2.0;
        amplitude = amplitude * 0.5;
    }

    return value;
}
