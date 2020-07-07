#version 430
#import "noise.glsl"


void cloud(vec3 frag_pos, float scale, int detail, out float out_factor, out vec3 out_color)
{
    vec2 uv = frag_pos.xy;
    vec3 color = vec3(0.);

    float fBM = fractalBrownianMotion(uv * scale, detail);
    out_factor = fBM;
    out_color += fBM;
}
