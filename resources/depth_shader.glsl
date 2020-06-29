#version 430

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 uv;

void main() {
    uv = in_texcoord_0;
    gl_Position = vec4(in_position, 1.0);
}


#elif defined FRAGMENT_SHADER

#define TMIN 0.
#define TMAX 65536.

uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float z = texture(tex, uv).r;
    fragColor = vec4(hsv2rgb(vec3(z/7.5, .6, .81)), 1.);
}
    #endif