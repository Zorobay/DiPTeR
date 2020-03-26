"""
This document contains type translations between GLSL, numpy and internal types
"""
import OpenGL.raw.GL.VERSION.GL_2_0 as gl_types_2
import OpenGL.raw.GL.VERSION.GL_4_3 as gl_types_4
import numpy as np
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

ARRAY = "type_array_"
INTERNAL_TYPE_FLOAT = "type_float"
INTERNAL_TYPE_ARRAY_FLOAT = ARRAY + "float"
INTERNAL_TYPE_ARRAY_RGB = ARRAY + "rgb"
INTERNAL_TYPE_ARRAY_RGBA = ARRAY + "rgba"

# ----- Translations from GLSL --------
GL_TO_TYPE = {
    gl_types_4.GL_FLOAT: Type(gl_types_4.GL_FLOAT, np.float32, INTERNAL_TYPE_FLOAT),
    gl_types_2.GL_FLOAT_VEC3: Type(gl_types_2.GL_FLOAT_VEC3, np.ndarray, INTERNAL_TYPE_ARRAY_FLOAT)
}

# ----- Translations from Internal -----
INTERNAL_TO_TYPE = {
    INTERNAL_TYPE_FLOAT: GL_TO_TYPE[gl_types_4.GL_FLOAT],
    INTERNAL_TYPE_ARRAY_FLOAT: GL_TO_TYPE[gl_types_2.GL_FLOAT_VEC3]
}
