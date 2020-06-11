#version 430

in vec3 frag_pos;
out vec4 frag_color;

#import "trig.glsl"
#import "pattern.glsl"


void main()
{
    if (frag_pos.x > 0.05 && frag_pos.x < 0.06){  //Test horizontal thin alignment
        frag_color = vec4(0.,1.,0.,1.);
    } else if (frag_pos.x >= 0.1 && frag_pos.x <= 0.2){  // Test horizontal thick alignment
        frag_color = vec4(1.0, 1., 0., 1.0);
    } else {
        frag_color = vec4(1., 1., 1., 1.);
    }

    if (frag_pos.y >= 0.7 && frag_pos.y <= 0.9){  // Test vertical thick alignment
        frag_color = vec4(0.,0.,1.,1.);
    }
}
