#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light {
    //vec3 position;
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 viewPos;
uniform Light light;
uniform Material mat;

uniform float time;
void main() {
    vec4 textureColour = texture(mat.diffuse, TexCoord)
    vec3 ambient = light.ambient * textureColour.rgb;
    FragColor = vec4(ambient, 1.0);
}