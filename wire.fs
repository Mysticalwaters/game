#version 330 core
out vec4 FragColor;
uniform vec3 wireColor = vec3(0.9, 0.9, 0.9);  // Red wire

void main() {
    FragColor = vec4(wireColor, 1.0);
}