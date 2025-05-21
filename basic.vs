#version 330 core
layout (location = 0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
layout(location=2) in vec3 aNormal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

const float waveHeight = 0.12;
const float waveSpeed = 1.5;
const float waveScale = 1.0;  // Adjusts wave frequency

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

// [Previous declarations remain the same...]

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    
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

    // Calculate proper partial derivatives for X and Z
    // Wave 1: affects both X and Z equally
    float dx1 = cos(u1) * waveHeight * waveScale;
    float dz1 = dx1;  // Same as dx1 because of (x+z)

    // Wave 2: affects X and Z with opposite signs
    float dx2 = -cos(u2) * waveHeight * 0.8 * waveScale;
    float dz2 = cos(u2) * waveHeight * 0.8 * waveScale;

    // Wave 3: only affects X
    float dx3 = -sin(u3) * waveHeight * 0.6 * waveScale * 1.3;
    float dz3 = 0.0;

    // Wave 4: only affects Z
    float dx4 = 0.0;
    float dz4 = -cos(u4) * waveHeight * 0.5 * waveScale * 0.8;

    // Wave 5: affects X and Z with different weights
    float dx5 = -sin(u5) * waveHeight * 0.4 * waveScale * 1.1 * 0.7;
    float dz5 = -sin(u5) * waveHeight * 0.4 * waveScale * 1.1 * 1.4;

    // Combine all derivatives
    float dxTotal = dx1 + dx2 + dx3 + dx4 + dx5;
    float dzTotal = dz1 + dz2 + dz3 + dz4 + dz5;

    // Create normal vector (tangent plane normal)
    vec3 normal = normalize(vec3(-dxTotal, 1.0, -dzTotal));

    // Transform normal to world space
    Normal = mat3(transpose(inverse(model))) * normal;

    // Apply waves to y coordinate
    vec3 finalPos = vec3(aPos.x, aPos.y + wave1 + wave2 + wave3 + wave4 + wave5, aPos.z);
    gl_Position = projection * view * model * vec4(finalPos, 1.0);
    TexCoord = aTexCoord;
}