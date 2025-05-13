from PIL import Image
from OpenGL.GL import *
import glfw
import numpy

def loadImage(filePath):
    image = Image.open(filePath)

    if image.mode != "RGBA":
        image.convert("RGBA")

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    imageData = numpy.array(image, dtype=numpy.uint8)
    return imageData, image.width, image.height


def loadTexture(filePath):
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId)

    imageData, imageWidth, imageHeight = loadImage(filePath)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData)
    glGenerateMipmap(GL_TEXTURE_2D)
    return textureId
    