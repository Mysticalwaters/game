#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float time;

const float waveHeight = 0.09;
const float waveSpeed = 1.0;
const float waveScale = 1.0;  // Adjusts wave frequency

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);

    // Wave calculations (unchanged)
    float u1 = time * waveSpeed + (worldPos.x + worldPos.z) * waveScale;
    float u2 = time * waveSpeed - (worldPos.x - worldPos.z) * waveScale;
    float u3 = time * waveSpeed * 0.7 + worldPos.x * waveScale * 1.3;
    float u4 = time * waveSpeed * 1.2 - worldPos.z * waveScale * 0.8;
    float u5 = time * waveSpeed * 0.5 + (worldPos.x * 0.7 + worldPos.z * 1.4) * waveScale * 1.1;

    float wave1 = sin(u1) * waveHeight;
    float wave2 = sin(u2) * waveHeight * 0.8;
    float wave3 = cos(u3) * waveHeight * 0.6;
    float wave4 = sin(u4) * waveHeight * 0.5;
    float wave5 = cos(u5) * waveHeight * 0.4;

    vec3 finalPos = vec3(aPos.x, aPos.y + wave1 + wave2 + wave3 + wave4 + wave5, aPos.z);
    gl_Position = projection * view * model * vec4(aPos.x, finalPos.y, aPos.z, 1.0);
}