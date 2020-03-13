"""
This document contains type translations between GLSL, numpy and internal types
"""
import numpy as np
from OpenGL.arrays._arrayconstants import GL_FLOAT
from OpenGL.constant import IntConstant


class Type:
    """
    A holder for type translations
    """

    def __init__(self, gl_type: IntConstant, numpy_type: type, internal_type: str):
        self._gl_type = gl_type
        self._numpy_type = numpy_type
        self._internal_type = internal_type

    @property
    def gl_type(self) -> IntConstant:
        """OpenGL type constant"""
        return self._gl_type

    @property
    def numpy_type(self) -> type:
        """The Numpy type that corresponds to the openGL type"""
        return self._numpy_type

    @property
    def internal_type(self):
        """A string constant that is used internally for Sockets"""
        return self._internal_type


def from_gl_type(gl_type: IntConstant) -> Type:
    """Constructs a Type object with type translations from an OpenGL type constant"""
    assert gl_type in GL_TO_TYPE, "{} is either not a valid OpenGL type constant or it is not supported yet!".format(gl_type)
    return GL_TO_TYPE[gl_type]


# ----- Internal Types --------

INTERNAL_TYPE_FLOAT = "type_float"
INTERNAL_TYPE_RGB = "type_rgb"


# ----- Translations from GLSL --------
GL_TO_TYPE = {
    GL_FLOAT: Type(GL_FLOAT, np.float32, INTERNAL_TYPE_FLOAT),
}

# ----- Translations from Internal -----
INTERNAL_TO_TYPE = {
    INTERNAL_TYPE_FLOAT: GL_TO_TYPE[GL_FLOAT]
}
