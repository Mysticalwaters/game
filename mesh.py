import glfw
from OpenGL.GL import *
import glm
import numpy

def loadVertexBuffer(buffer, size, dynamic):
    id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, id)
    glBufferData(GL_ARRAY_BUFFER, size, buffer, GL_DYNAMIC_DRAW if dynamic else GL_STATIC_DRAW)
    return id
    

class Mesh:
    def __init__(self):
        self.vertices = 0
        self.vertexCount = 0
        self.normals = []
        self.textCoords = []
        self.shader = 0
        self.vboId = [0, 0] # 0 = Position, 1 = Texcoords, stores vbo's
                        #IMPORTANT, CURRENTLY ONLY POSITION IS IMPLEMENTED!
        self.VAO = 0 # contains VBOs
        self.model = glm.mat4(1.0)
        
    def translate(self, pos : glm.vec3):
        self.model = glm.translate(self.model, pos)

    def scale(self, scale):
        self.model = glm.scale(self.model, scale)

    def uploadMesh(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.vboId[0] = loadVertexBuffer(self.vertices.ptr, self.vertices.nbytes, False)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0)  # 3 floats per vertex
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    
    def draw(self, position, size):
        #Pass model to shader and enable shader
        self.shader.use()
        self.model = glm.mat4(1.0)
        self.translate(position)
        #self.scale(size)
        self.shader.setMat4Uniform("model", self.model)

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, self.vertexCount)

#Creates a mesh() of a plane using the width and height. ResX and y represent the number of subdivions in the mesh, based off of raylibs source code
def createPlaneMesh(resX, resZ, width, length):
    resX += 1
    resZ += 1

    vertexCount = resX*resZ
    vertices = glm.array.zeros(vertexCount, glm.vec3)
    
    for z in range(resZ):
        zPos = (float(z)/(resZ - 1) - 0.5) * length
        for x in range(resX):
            xPos = (float(x)/(resX - 1) - 0.5)*width
            vertices[x + z*resX] = glm.vec3(xPos, 0.0, zPos)
    
    mesh = Mesh()
    mesh.vertexCount = vertexCount
    mesh.vertices = vertices
    mesh.uploadMesh()
    return mesh
   
            