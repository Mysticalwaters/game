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
uniform Material material;
uniform vec3 ambientLighting;

const float speed = 1.0;

uniform float time;
void main() {
    vec3 norm = normalize(Normal);

    vec4 textureColour = texture(material.diffuse, TexCoord + (time * speed));
    vec3 ambient = light.ambient;// + textureColour.rgb
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff; // * textureColour.rgb
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir);  // Halfway vector
    float spec = pow(max(dot(norm, halfDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec;
    vec3 result = ambient + diffuse*vec3(0.3, 0.3, 0.8) + specular;

    FragColor = vec4(result  ,1.0);
}