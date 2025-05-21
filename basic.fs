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
const float fogEnd = 80.0;                     // Distance where fog is fully opaque
const float fogDensity = 0.019;                // For exponential fog

const float speed = 1.0;
uniform float time;

void main() {
    // Calculate lighting as before
    vec3 norm = normalize(Normal);
    
    vec4 textureColour = texture(material.diffuse, TexCoord + (time * speed));
    vec3 ambient = light.ambient;
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec;
    vec3 result = ambient + diffuse*vec3(0.3, 0.3, 0.8) + specular;
    
    float fogCoord = length(viewPos - FragPos); 
    float fogFactor = (fogEnd - fogCoord) / (fogEnd - fogStart);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    vec4 finalColor = mix(vec4(fogColor, 1.0), vec4(result, 1.0), fogFactor);
    
    FragColor = finalColor;
}