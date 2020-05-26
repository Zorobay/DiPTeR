#version 430
#import "noise.glsl"


vec3 cloud(vec3 frag_pos, float scale, int detail)
{
    vec2 uv = frag_pos.xy;
    vec3 color = vec3(0.);
    color += fractalBrownianMotion(uv * scale, detail);

    return color;
}
