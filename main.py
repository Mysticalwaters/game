import glfw
from OpenGL.GL import *
import numpy as np
import glm
import string
import camera
from mesh import *
import time
from texture import *

view = 0
proj = 0

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

const float waveHeight = 0.2;
const float waveSpeed = 1.0;
const float waveScale = 1.0;  // Adjusts wave frequency

out vec2 TexCoord;

void main() {
    // Get the world position by transforming the vertex
    vec4 worldPos = model * vec4(aPos, 1.0);
    
    // Calculate waves using world coordinates
    float wave1 = sin(time * waveSpeed + (worldPos.x + worldPos.z) * waveScale) * waveHeight;
    float wave2 = sin(time * waveSpeed * 0.7 + (worldPos.x - worldPos.z) * waveScale * 1.3) * waveHeight * 0.5;
    
    // Apply waves to y coordinate
    vec3 finalPos = vec3(aPos.x, wave1 + wave2, aPos.z);
    gl_Position = projection * view * model * vec4(finalPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D uTexture;
void main() {
    vec3 baseColour = vec3(0.3, 0.3, 0.6);
    FragColor = vec4(baseColour, 1.0);
}
"""

activeShaders = []

class Shader:
    def __init__(self, vsSource : string, fsSource : string):
        self.__program = self.__createShaderProgram(vsSource, fsSource)
        activeShaders.append(self)
    
    #Function quite simply takes shader source code and uploads it to the GPU
    @staticmethod
    def __compileShader(shaderType, source):
        shader = glCreateShader(shaderType)
        glShaderSource(shader, source)
        glCompileShader(shader)
        #If compilation fails we raise an error with related information!
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader))
        #else we return the shader so that we can make a shader program
        return shader

    def __createShaderProgram(self, vsSource : string, fsSource : string):
        vertexShader = self.__compileShader(GL_VERTEX_SHADER, vsSource)
        fragShader = self.__compileShader(GL_FRAGMENT_SHADER, fsSource)
        program = glCreateProgram()

        glAttachShader(program, vertexShader)
        glAttachShader(program, fragShader)
        glLinkProgram(program)

        glDeleteShader(vertexShader)
        glDeleteShader(fragShader)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            glDeleteProgram(program)
            raise RuntimeError(glGetProgramInfoLog(program))
        return program

    def setMat4Uniform(self, location : string, value : glm.mat4):
        loc = glGetUniformLocation(self.__program, location)
        if loc == -1:
            print(f"Warning: Uniform '{location}' not found in shader!")
            return
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm.value_ptr(value))

    def setVec3Uniform(self, location : string, value : glm.vec3):
        loc = glGetUniformLocation(self.__program, location)
        if loc == -1:
            print(f"Warning: Uniform '{location}' not found in shader!")
            return
        glUniform3fv(loc, 1, GL_FALSE, glm.value_ptr(value))

    def setFloat(self, location: string, value: float):
        loc = glGetUniformLocation(self.__program, location)
        if loc == -1:
            print(f"Warning: Uniform '{location}' not found in shader!")
            return
        glUniform1f(loc, value)

    def setInt(self, location: string, value: int):
        loc = glGetUniformLocation(self.__program, location)
        if loc == -1:
            print(f"Warning: Uniform '{location}' not found in shader!")
            return
        glUniform1i(loc, value)
    
    @property
    def getProgram(self):
        return self.__program
    
    def use(self):
        glUseProgram(self.getProgram)

class Mesh:
    def __init__(self):
        self.VAO = 0
        self.VBO = 0
        self.EBO = 0
        self.indicies = 0
    def uploadToMesh(self):
        pass
    

class Model:
    def __init__(self, x, y, z, shader):
        self.pos = glm.vec3(x, y, z)
        self.Mesh = 0
        self.shader = shader
        self.texture = None
    def draw(self):
        self.shader.use()
        self.shader.setMat4Uniform("model", glm.translate(glm.mat4(1.0), self.pos))
        self.shader.setMat4Uniform("view", view)
        self.shader.setMat4Uniform("projection", proj)
        if self.texture != None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            self.shader.setInt("uTexture", 0)
        glBindVertexArray(self.Mesh.VAO)
        glDrawElements(GL_TRIANGLES, self.Mesh.indicies, GL_UNSIGNED_INT, None)  

def keycallback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)



def createWindow(name, resolution, cam):
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    glfw.window_hint(glfw.SAMPLES, 8)
    # Create a window
    window = glfw.create_window(resolution[0], resolution[1], name, None, None) #Creates window, first two values is window resolution whereas the string is the title.
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_key_callback(window, keycallback)
    glfw.set_cursor_pos_callback(window, cam.mouse_callback)
    glfw.swap_interval(0)
    return window

def create_plane(resX=10, resZ=10, size=5.0) -> Mesh:
    """Creates a subdivided plane mesh centered at origin."""
    half_size = size / 2.0
    stepX = size / (resX - 1)
    stepZ = size / (resZ - 1)

    vertices = []
    for z in range(resZ):
        for x in range(resX):
            xpos = -half_size + x * stepX
            zpos = -half_size + z * stepZ
            u = ((xpos + half_size)/size) * 16
            v = ((zpos + half_size)/size) * 16
            vertices.extend([xpos, 0.0, zpos, u, v])  # Flat plane on y=0

    # Generate indices for triangle list
    indices = []
    for z in range(resZ - 1):
        for x in range(resX - 1):
            topLeft     = z * resX + x
            topRight    = topLeft + 1
            bottomLeft  = topLeft + resX
            bottomRight = bottomLeft + 1

            # First triangle
            indices.extend([topLeft, bottomLeft, topRight])
            # Second triangle
            indices.extend([topRight, bottomLeft, bottomRight])

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    # OpenGL buffers
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)
    mesh = Mesh()
    mesh.indicies = indices.size
    mesh.VAO = VAO
    mesh.VBO = VBO
    mesh.EBO = EBO
    return mesh

def main():
    cam = camera.Camera(glm.vec3(0.0, 1.0, 3.0), 45.0)
    window = createWindow("My fishing game!", (1960, 1080), cam)
    basicShader = Shader(vertex_shader_source, fragment_shader_source)
    model = glm.mat4(1.0)  # Identity matrix
    
    
    
    deltatime = 0.0
    lastframe = 0.0

    glEnable(GL_DEPTH_TEST)
    frame_count = 0
    last_fps_time = 0.0
    fps = 0.0
    listWaterTiles = []
    
    for x in range(1):
        for z in range(1):
            water = Model(x * 10, 0, z * 10, basicShader)
            water.Mesh = create_plane(40, 40, 10)
            listWaterTiles.append(water)
            #
            # water.texture = waterTexture
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    # Main loop - Can be seen as the rendering loop, this is where most shader and draw calls will be made!
    while not glfw.window_should_close(window):
        #Set fps to 75
        #time.sleep(1 / frameRate)
        glClearColor(0.0, 0.0, 0.0, 1.0)  # RGBA (Red)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        currentTime = glfw.get_time()
        time = currentTime - lastframe
        deltatime += time
        lastframe = currentTime

        frame_count += 1
        if currentTime - last_fps_time >= 1.0:  # Update every second
            fps = frame_count / (currentTime - last_fps_time)
            frame_count = 0
            last_fps_time = currentTime
            glfw.set_window_title(window, f"My fishing game! - FPS: {fps:.1f}")

        cam.processInput(window,time)
        cam.updateCamera()

        basicShader.use()
        global view, proj
        view = cam.view
        proj = cam.projection
        basicShader.setFloat("time", currentTime)
        for i in listWaterTiles:
            i.draw()

              
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
        

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()