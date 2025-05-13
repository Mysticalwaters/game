#version 330 core
layout (location = 0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

const float baseWaveHeight = 0.2;
const float baseWaveSpeed = 1.0;
const float baseWaveLength = 1.0;
const int waveCount = 3;

out vec2 TexCoord;

float createWaves(float x) {
    float total = 0.0;
    float amplitude = baseWaveHeight;
    float waveLength = baseWaveLength;
    for(int i = 0; i < waveCount; i++) {
        float frequency = 2.0/waveLength;
        float phase = baseWaveSpeed * frequency;
        float wave = amplitude * sin(x * waveFrequency + time*phase);
        amplitude *= 0.5;
        waveLength *= 0.8;
        total += wave;
    }
    return wave / waveCount;
}

void main() {
    // Get the world position by transforming the vertex
    vec4 worldPos = model * vec4(aPos, 1.0);
    
    // Calculate waves using world coordinates
    float wave1 = createWaves(worldPos.x);
    
    // Apply waves to y coordinate
    vec3 finalPos = vec3(aPos.x, wave1, aPos.z);
    gl_Position = projection * view * model * vec4(finalPos, 1.0);
    TexCoord = aTexCoord;
}