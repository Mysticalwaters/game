#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform vec4 colour = vec4(1.0);  // Default white
uniform sampler2D texture1;
uniform bool useTexture = false;

void main() {
    if(useTexture) {
        FragColor = texture(texture1, TexCoord) * colour;
    } else {
        FragColor = colour;
    }
}