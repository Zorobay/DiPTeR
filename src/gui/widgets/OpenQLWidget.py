import math
import time

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QOpenGLWidget
from glumpy import gl, glm, gloo

from src.opengl import object_vertices
from src.shaders.color_shader import ColorShader
from src.shaders.gradient_shader import GradientShader
from src.shaders import OBJECT_MATRIX_NAME, VIEW_MATRIX_NAME, PROJECTION_MATRIX_NAME


class OpenGLWidget(QOpenGLWidget):

    def __init__(self, width, height):
        super().__init__()
        # Set Widget settings
        self.setMinimumSize(width, height)
        self.setMouseTracking(False)

        self._program = None
        self._timer = None
        self._frame_rate = 60
        self._I = None

        # Define variables for mouse and keyboard interaction
        self._scroll_speed = 0.2
        self._rotate_speed = 0.2
        self._last_mouse_x = -1
        self._last_mouse_y = -1

        # Define variables for transformation matrices
        self._object_to_world = np.eye(4, dtype=np.float32)
        self._world_to_view = np.eye(4, dtype=np.float32)
        self._view_to_projection = np.eye(4, dtype=np.float32)

    @property
    def frame_rate(self):
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, value: int):
        assert value <= 1000, "Frame rate can not be set higher than 1000 fps"
        self.frame_rate = value
        self._timer.setInterval(int(1000 / self.frame_rate))

    def initializeGL(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glFrontFace(gl.GL_CCW)
        self._init_camera()
        self._init_shaders()

        # Start an update timer to refresh rendering
        self._timer = QTimer()
        self._timer.setInterval(int(1000 / self._frame_rate))
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def _init_camera(self):
        # Translate out view in negative z dir
        glm.translate(self._world_to_view, 0, 0, -5)

    def _init_shaders(self):
        gs = GradientShader()
        V, I = object_vertices.get_3d_cube()
        V = V.view(gloo.VertexBuffer)
        I = I.view(gloo.IndexBuffer)
        self._I = I

        self._program = gs.get_program(len(V))

        self._program.bind(V)
        self._program["rot"] = 0.0
        self._program["color1"] = np.array((1.0, 0.1, 0.4), dtype=np.float32)
        self._program["color2"] = np.array((0.0, 0.1, 0.8), dtype=np.float32)

        self._program[OBJECT_MATRIX_NAME] = self._object_to_world
        self._program[VIEW_MATRIX_NAME] = self._world_to_view
        self._program[PROJECTION_MATRIX_NAME] = self._view_to_projection

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if self._program:
            self._program.draw(gl.GL_TRIANGLES, self._I)

            # Send the updated transformation matrices to the shader
            self._program[OBJECT_MATRIX_NAME] = self._object_to_world
            self._program[VIEW_MATRIX_NAME] = self._world_to_view

    def resizeGL(self, w: int, h: int):
        ratio = w / float(h)
        self._program[PROJECTION_MATRIX_NAME] = glm.perspective(45.0, ratio, znear=2, zfar=100.0)

    def wheelEvent(self, wheel_event):
        scroll_steps = wheel_event.angleDelta().y() / 8 / 15  # Get actual number of steps (default is 15 deg/step)

        view_y = self._world_to_view[-1, 2]
        if (view_y >= 0 and scroll_steps > 0) or view_y <= -600:
            return

        glm.translate(self._world_to_view, 0, 0, scroll_steps * self._scroll_speed)

    def mouseMoveEvent(self, mouse_event):
        if mouse_event.buttons() == Qt.LeftButton:
            x = mouse_event.x()
            y = mouse_event.y()
            x_delta = 0
            y_delta = 0

            if self._last_mouse_x >= 0:
                x_delta = x - self._last_mouse_x

            if self._last_mouse_y >= 0:
                y_delta = y - self._last_mouse_y

            self._last_mouse_x = x
            self._last_mouse_y = y
            y_angle = x_delta * self._rotate_speed  # A movement in x-dir translates to rotation around y-axis
            x_angle = y_delta * self._rotate_speed  # A movement in y-dir translates to rotation around x-axis

            glm.rotate(self._object_to_world, x_angle, 1, 0, 0)
            glm.rotate(self._object_to_world, y_angle, 0, 1, 0)

    def mouseReleaseEvent(self, mouse_event):
        # When the mouse is released, reset the last mouse positions
        self._last_mouse_x = -1
        self._last_mouse_y = -1
