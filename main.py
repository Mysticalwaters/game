import string
import time
import math

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

    def setVec4Uniform(self, location : string, value : glm.vec4):
        loc = glGetUniformLocation(self.__program, location)
        if loc == -1:
            print(f"Warning: Uniform '{location}' not found in shader!")
            return
        glUniform4fv(loc, 1, glm.value_ptr(value))

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
    def draw(self, wire=False):
        self.shader.use()
        self.shader.setMat4Uniform("view", view)
        self.shader.setMat4Uniform("projection", proj)
        if self.texture != None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindVertexArray(self.Mesh.VAO)
        if not wire:
            self.shader.setVec3Uniform("light.ambient", glm.vec3(0.2, 0.2, 0.2))
            self.shader.setVec3Uniform("light.direction", glm.vec3(-0.2, -1.0, -0.3))
            self.shader.setVec3Uniform("light.diffuse", glm.vec3(0.5, 0.5, 0.5))
            self.shader.setMat4Uniform("model", glm.translate(glm.mat4(1.0), self.pos))
            glDrawElements(GL_TRIANGLES, self.Mesh.indicies, GL_UNSIGNED_INT, None)  
        else:
            self.shader.setMat4Uniform("model", glm.mat4(1.0))
            glDrawElements(GL_LINES, self.Mesh.indicies, GL_UNSIGNED_INT, None)  
        

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

def create_sphere(rings, stacks, radius) -> Mesh:
    vertices = []
    for i in range(rings+1):
        theta = (i / rings) * math.pi
        for j in range(stacks+1):
            phi = (j / stacks) * (math.pi*2)
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.cos(theta)
            z = radius * math.sin(theta) * math.sin(phi)
            vertices.extend([x, y, z, j/stacks, i/rings, x/radius, y/radius, z/radius,])

    indices = []
    for i in range(rings):
        for j in range(stacks):
            first = i * (stacks + 1) + j
            second = first + stacks + 1
            
            # Two triangles per sector
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

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

def create_wire(player_pos, bobber_pos) -> Mesh:
    # Use np (not numpy) consistently
    vertices = np.array([
        player_pos.x, player_pos.y, player_pos.z,
        bobber_pos.x, bobber_pos.y, bobber_pos.z
    ], dtype=np.float32)
    
    indices = np.array([0, 1], dtype=np.uint32)
    
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    
    # Initialize buffer with correct size (6 floats × 4 bytes)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 6 * 4, vertices, GL_DYNAMIC_DRAW)
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    # Position attribute (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    glBindVertexArray(0)
    
    mesh = Mesh()
    mesh.indicies = indices.size
    mesh.VAO = VAO
    mesh.VBO = VBO
    mesh.EBO = EBO
    return mesh

def update_wire(player_pos, bobber_pos, wire_mesh):
    # Create properly formatted numpy array
    vertices = np.array([
        player_pos.x, player_pos.y, player_pos.z,
        bobber_pos.x, bobber_pos.y+0.04, bobber_pos.z
    ], dtype=np.float32)
    
    glBindBuffer(GL_ARRAY_BUFFER, wire_mesh.VBO)
    # Upload exactly 6 floats (24 bytes)
    glBufferSubData(GL_ARRAY_BUFFER, 0, 6 * 4, vertices)

def get_fishing_line_offset(camera_pos, camera_front, right_offset=0.1, vertical_offset=0.1):
    """Calculate position slightly to the right of the camera"""
    camera_right = glm.normalize(glm.cross(camera_front, glm.vec3(0, 1, 0)))
    offset_pos = camera_pos + (camera_right * right_offset) + (glm.vec3(0, 1, 0) * vertical_offset)
    return offset_pos

class ui_rect:
    def __init__(self, x, y, width, height, colour, texture=None):
        self.pos = glm.vec2(x,y)
        self.size = glm.vec2(width, height)
        self.colour = glm.vec4(colour, 1.0)
        vertices = np.array([
        # Positions   # Texture Coords
        -0.5, -0.5,  0.0, 0.0,  # Bottom-left
         0.5, -0.5,  1.0, 0.0,  # Bottom-right
         0.5,  0.5,  1.0, 1.0,  # Top-right
        -0.5,  0.5,  0.0, 1.0   # Top-left
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(VAO)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        
        mesh = Mesh()
        mesh.indicies = indices.size
        mesh.VAO = VAO
        mesh.VBO = VBO
        mesh.EBO = EBO
        self.mesh = mesh

        #More class setup
        self.texture = texture
        self.shader = Shader("ui.vs", "ui.fs") #UI is quite simple so we wont need differing shaders
        self.proj = glm.ortho(0, 1960, 1080, 0)

    def draw(self):
        glDisable(GL_DEPTH_TEST)
        self.shader.use()
        self.shader.setMat4Uniform("projection", self.proj)
        self.shader.setVec4Uniform("colour", self.colour)
        if self.texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            self.shader.setInt("texture1", 0)
            self.shader.setInt("useTexture", 1)
        else:
            self.shader.setInt("useTexture", 0)
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(self.pos.x, self.pos.y, 0))
        model = glm.scale(model, glm.vec3(self.size.x, self.size.y, 1))
        self.shader.setMat4Uniform("model", model)

        glBindVertexArray(self.mesh.VAO)
        glDrawElements(GL_TRIANGLES, self.mesh.indicies, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

class text_ui(ui_rect):
    def __init__(self, x, y, width, height, colour,text, texture=None):
        super().__init__(x, y, width, height, colour, texture)
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 32)
        self.window_size = (1960, 1080)
        self.text = text
    def draw_text(self):
        text_surface = self.font.render(self.text, True, self.colour)
        text_surface = pygame.transform.flip(text_surface, False, True)
        w, h = text_surface.get_size()
        
        # Convert to OpenGL texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        texture_data = pygame.image.tostring(text_surface, "RGBA", True)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glDisable(GL_DEPTH_TEST)
        self.shader.use()
        self.shader.setMat4Uniform("projection", self.proj)
        self.shader.setVec4Uniform("colour", self.colour)
        if self.texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            self.shader.setInt("texture1", 0)
            self.shader.setInt("useTexture", 1)
        else:
            self.shader.setInt("useTexture", 0)
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(self.pos.x, self.pos.y, 0))
        model = glm.scale(model, glm.vec3(self.size.x, self.size.y, 1))
        self.shader.setMat4Uniform("model", model)

        glBindVertexArray(self.mesh.VAO)
        glDrawElements(GL_TRIANGLES, self.mesh.indicies, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

def main():
    cam = camera.Camera(glm.vec3(45.0, 1.0, 45.0), 45.0)
    window = createWindow("My fishing game!", (1960, 1080), cam)
    bobberShader = Shader("wave.vs")
    waveShader = Shader("wave.vs", "wave.fs")
    wireShader = Shader("wire.vs", "wire.fs")
    model = glm.mat4(1.0)  # Identity matrix
    
    pygame.mixer.init()
    pygame.mixer.set_num_channels(8)

    """
    sound = pygame.mixer.Sound("ambientAudio.mp3")
    sound.set_volume(1.0)
    channel = pygame.mixer.find_channel()
    channel.play(sound, loops=-1)
    """
    deltatime = 0.0
    lastframe = 0.0

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    rect = ui_rect(64, 32, 160, 160, glm.vec3(0.5, 0.5, 0.5), loadTexture("fish.png", False))
    text = text_ui(100 + 32, 36, 32, 32, glm.vec3(0.0, 0.0, 0.0), "x5")

    frame_count = 0
    last_fps_time = 0.0
    fps = 0.0
    sphere = Model(cam.pos.x, 0.01, cam.pos.z, bobberShader)
    sphere.Mesh = create_sphere(20, 20, 0.05)

    listWaterTiles = []
    waterTexture = loadTexture("waterTexture.png")
    bobberTexture = loadTexture("bobber.png")
    sphere.texture = bobberTexture

    for x in range(9):
        for z in range(9):
            water = Model(x * 20, 0, z * 20, waveShader)
            water.Mesh = create_plane(80, 80, 20)
            listWaterTiles.append(water)
            water.texture = waterTexture

    wire_model = Model(0, 0, 0, wireShader)
    wire_model.Mesh = wire_mesh = create_wire(cam.pos, sphere.pos)
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

        waveShader.use()
        global view, proj
        view = cam.view
        proj = cam.projection
        waveShader.setFloat("time", currentTime)
        waveShader.setVec3Uniform("viewPos", cam.pos)
        for i in listWaterTiles:
            i.draw()

        bobberShader.use()
        bobberShader.setVec3Uniform("viewPos", cam.pos)
        bobberShader.setFloat("time", currentTime)
        sphere.draw()
        offset = get_fishing_line_offset(cam.pos, cam.front)
        wireShader.use()
        wireShader.setFloat("time", currentTime)
        wire_model.draw(True)
        update_wire(offset, sphere.pos, wire_mesh)

        rect.draw()
        text.draw_text()
        
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
        

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()