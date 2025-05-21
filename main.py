import string
import time

import glfw
import glm
import numpy as np
from OpenGL.GL import *
import pygame

import camera
from mesh import *
from texture import *

view = 0
proj = 0

activeShaders = []

class Shader:
    def __init__(self, vsPath="basic.vs", fsPath="basic.fs"):
        self.__program = self.__createShaderProgram(vsPath, fsPath)
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
    @staticmethod
    def __loadShader(fp : string):
        with open(fp, "rb") as fp:
            content = fp.read()
            return content

    def __createShaderProgram(self, vsPath="basic.vs", fsPath="basic.fs"):
        vertexShader = self.__compileShader(GL_VERTEX_SHADER, self.__loadShader(vsPath))
        fragShader = self.__compileShader(GL_FRAGMENT_SHADER, self.__loadShader(fsPath))
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
        glUniform3fv(loc, 1, glm.value_ptr(value))

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
        self.shader.setVec3Uniform("light.ambient", glm.vec3(0.2, 0.2, 0.2))
        self.shader.setVec3Uniform("light.direction", glm.vec3(-0.2, -1.0, -0.3))
        self.shader.setVec3Uniform("light.diffuse", glm.vec3(0.5, 0.5, 0.5))
        if self.texture != None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
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
            u = ((xpos + half_size)/size) * 8
            v = ((zpos + half_size)/size) * 8
            vertices.extend([xpos, 0.0, zpos, u, v, 0.0, 1.0, 0.0])  # Flat plane on y=0

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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(5 * 4))
    glEnableVertexAttribArray(2)

    glBindVertexArray(0)
    mesh = Mesh()
    mesh.indicies = indices.size
    mesh.VAO = VAO
    mesh.VBO = VBO
    mesh.EBO = EBO
    return mesh

def main():
    cam = camera.Camera(glm.vec3(45.0, 1.0, 45.0), 45.0)
    window = createWindow("My fishing game!", (1960, 1080), cam)
    basicShader = Shader()
    model = glm.mat4(1.0)  # Identity matrix
    
    pygame.mixer.init()
    pygame.mixer.set_num_channels(8)

    sound = pygame.mixer.Sound("ambientAudio.mp3")
    sound.set_volume(1.0)
    channel = pygame.mixer.find_channel()
    channel.play(sound, loops=-1)
    
    deltatime = 0.0
    lastframe = 0.0

    glEnable(GL_DEPTH_TEST)
    frame_count = 0
    last_fps_time = 0.0
    fps = 0.0
    listWaterTiles = []
    waterTexture = loadTexture("waterTexture.png")
    for x in range(9):
        for z in range(9):
            water = Model(x * 10, 0, z * 10, basicShader)
            water.Mesh = create_plane(40, 40, 10)
            listWaterTiles.append(water)
            #
            water.texture = waterTexture
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    # Main loop - Can be seen as the rendering loop, this is where most shader and draw calls will be made!
    while not glfw.window_should_close(window):
        #Set fps to 75
        #time.sleep(1 / frameRate)
        glClearColor(135/255, 206/255, 235/255, 1.0)  # RGBA (Red)
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
        basicShader.setVec3Uniform("viewPos", cam.pos)
        for i in listWaterTiles:
            i.draw()

              
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
        

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()