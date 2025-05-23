from PIL import Image
from OpenGL.GL import *
import glfw
import numpy

def loadImage(filePath, flip=True):
    image = Image.open(filePath)

    if image.mode != "RGBA":
        image.convert("RGBA")

    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    imageData = numpy.array(image, dtype=numpy.uint8)
    return imageData, image.width, image.height


def loadTexture(filePath, flip=True):
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId)

    imageData, imageWidth, imageHeight = loadImage(filePath, flip)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData)
    glGenerateMipmap(GL_TEXTURE_2D)
    return textureId
    