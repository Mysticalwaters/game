#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light {
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

// Fog constants (replace these values as needed)
const vec3 fogColor = vec3(0.53, 0.81, 0.92); // Light blue color matching your clear color
const float fogStart = 8.0;                   // Distance where fog starts
const float fogEnd = 50.0;                     // Distance where fog is fully opaque
const float fogDensity = 0.019;                // For exponential fog

const float speed = 0.5;
uniform float time;

void main() {
    // Calculate lighting as before
    float fogCoord = length(viewPos - FragPos);
    float fogFactor = (fogEnd - fogCoord) / (fogEnd - fogStart);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    if(fogFactor == 0.0) {
        discard;
    }
    vec3 norm = normalize(Normal);
    vec4 textureColour = texture(material.diffuse, TexCoord);
    vec3 ambient = light.ambient*textureColour.rgb;
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff*textureColour.rgb;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 32.0);
    vec3 specular = light.specular * spec;
    vec3 result = ambient + diffuse + specular;

    result = mix(result, vec3(0.3, 0.3, 0.8), FragPos.y);
    vec4 finalColor = mix(vec4(fogColor, 1.0), vec4(result, 1.0), fogFactor);
    FragColor = vec4(finalColor.rgb, 1.0);
}