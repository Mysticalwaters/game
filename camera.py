import glfw
from OpenGL.GL import *
import glm

class Camera:
    def __init__(self, pos : glm.vec3, fov : float):
        self.pos = pos
        self.target = glm.vec3(0.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.fov = fov
        self.pitch = 0.0
        self.yaw = -90.0
        self.speed = 2.0
        self.view = glm.mat4
        self.first_mouse = True
        self.mouse_sensitivity = 0.1
        self.last_x = 400  # Half of typical window width
        self.last_y = 300  # Half of typical window height
        
        # Projection matrix (perspective)
        self.projection = glm.perspective(
            glm.radians(90.0),  # Field of view or FOV for most people
            1960 / 1080,          # Aspect ratio - Should change this to reflect a varible.
            0.1,                # Near clipping plane
            100.0              # Far clipping plane - defines how much will be rendered based on max distance
        )

    def updateCamera(self):
        self.view = glm.lookAt(
        self.pos,  # Camera position (x, y, z)
        glm.add(self.pos, self.front),  # Target (origin)
        self.up)   # Up vector
        
    

    def processInput(self, window, deltatime : float):
        front = glm.vec3()
        front.x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        front.y = glm.sin(glm.radians(self.pitch))
        front.z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.pos = glm.add(self.pos, glm.mul(self.speed * deltatime, self.front))
            
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.pos = glm.sub(self.pos, glm.mul(self.speed * deltatime, self.front))

        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.pos = glm.sub(self.pos, glm.mul(glm.normalize(glm.cross(self.front, self.up)), (self.speed * deltatime)))

        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.pos = glm.add(self.pos, glm.mul(glm.normalize(glm.cross(self.front, self.up)), (self.speed * deltatime)))

    
    def mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos  # Reversed since y-coordinates go from bottom to top
        self.last_x = xpos
        self.last_y = ypos

        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        # Constrain pitch to prevent screen flipping
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0