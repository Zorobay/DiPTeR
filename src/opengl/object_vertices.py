import numpy as np

from src.shaders import IN_VERTEX_POS_NAME

VERTEX_COORD_MAXES = np.array((1.0, 1.0, 1.0))
VERTEX_COORD_MINS = np.array((-1.0, -1.0, -1.0))


def get_2d_plane():
    """
    Returns a tuple of a VertexBuffer of 4 vertices corresponding to a plane as well as an IndexBuffer with vertex indices for triangles.
    """
    V = np.zeros(4, [(IN_VERTEX_POS_NAME, np.float32, 3)])
    V[IN_VERTEX_POS_NAME] = np.array([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=np.float32)
    I = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return V, I


def get_3d_cube():
    """
    Returns a tuple of a VertexBuffer of 8 vertices corresponding to a cube as well as an IndexBuffer with vertex indices for triangles.
    """
    V = np.zeros(8, [(IN_VERTEX_POS_NAME, np.float32, 3)])
    V[IN_VERTEX_POS_NAME] = np.array([(-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])
    I = np.array([0, 1, 2, 0, 2, 3, 0, 4, 5, 0, 5, 1, 0, 4, 7, 0, 7, 3, 3, 7, 6, 3, 6, 2, 1, 5, 6, 1, 6, 2, 4, 5, 6, 4, 6, 7], dtype=np.uint32)
    return V, I
