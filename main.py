import glfw
from OpenGL.GL import *
import numpy as np
import glm
import string
import camera
from mesh import *

# Vertex Shader (transforms vertices)
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# Fragment Shader (colors pixels)
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);  // Orange color
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
    
    @property
    def getProgram(self):
        return self.__program
    
    def use(self):
        glUseProgram(self.getProgram)
    

def generateVertexBuffers(data):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    
    # Specify vertex attributes (how to interpret VBO data)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)  # 3 floats per vertex
    glEnableVertexAttribArray(0)
    
    # Unbind VBO and VAO (optional)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return VAO, VBO

def keycallback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)



def createWindow(name, resolution, cam):
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    # Create a window
    window = glfw.create_window(resolution[0], resolution[1], name, None, None) #Creates window, first two values is window resolution whereas the string is the title.
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_key_callback(window, keycallback)
    glfw.set_cursor_pos_callback(window, cam.mouse_callback)
    return window

def main():
    cam = camera.Camera(glm.vec3(0.0, 0.0, 1.0), 45.0)
    window = createWindow("My fishing game!", (800, 600), cam)

    # Define triangle vertices (x, y, z) as 32 bit floating point number
    vertices = np.array([
        -0.5, -0.5, 0.0,  # Left
         0.5, -0.5, 0.0,  # Right
         0.0,  0.5, 0.0   # Top
    ], dtype=np.float32)

    basicShader = Shader(vertex_shader_source, fragment_shader_source)
    VAO, VBO = generateVertexBuffers(vertices) #fine right now, but should move to mesh creation at some point!

    model = glm.mat4(1.0)  # Identity matrix
    print(hasattr(glm, 'translate'))  # Should return True if available
    
    
    deltatime = 0.0
    lastframe = 0.0
    plane = createPlaneMesh(1, 1, 10, 10)
    plane.shader = basicShader
    for i in plane.vertices:
        print(i)

    # Main loop - Can be seen as the rendering loop, this is where most shader and draw calls will be made!
    while not glfw.window_should_close(window):
        # Clear the screen with red
        glClearColor(0.0, 0.0, 0.0, 1.0)  # RGBA (Red)
        glClear(GL_COLOR_BUFFER_BIT)

        currentTime = glfw.get_time()
        deltatime = currentTime - lastframe
        lastframe = currentTime
        
        # Rotate model over time
        model = glm.rotate(
            glm.mat4(1.0),  # Start with identity
            currentTime,            # Rotation angle (radians)
            glm.vec3(0.0, 0.0, 1.0)  # Rotate around Y-axis
        )

        
        basicShader.use()
        basicShader.setMat4Uniform("model", model)
        cam.processInput(window,currentTime)
        cam.updateCamera(activeShaders)
        

        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        plane.draw(glm.vec3(0.0, 0.0, 0.0), 1.0)
        

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()