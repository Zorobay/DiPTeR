import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QMouseEvent, QImage
from PyQt5.QtWidgets import QOpenGLWidget, QMenu, QFileDialog
from glumpy import gl, glm, gloo
from glumpy.gloo import Program

from src.gui.node_editor.material import Material
from src.opengl import object_vertices
from src.shaders import OBJECT_MATRIX_NAME, VIEW_MATRIX_NAME, PROJECTION_MATRIX_NAME
from src.shaders.brick_shader import BrickShader


class OpenGLWidget(QOpenGLWidget):

    def __init__(self, width: int, height: int, material: Material):
        super().__init__()

        # Set Widget settings
        self.setMinimumSize(width, height)
        self.setMouseTracking(False)

        # Define variables to track objects
        self.material = material
        self._program = None
        self._shader = None
        self._timer = None
        self._I = None
        self._V = None

        # Define variables for settings
        self._clear_color = np.array((0.4, 0.4, 0.4, 1.0), dtype=np.float32)
        self._frame_rate = 60
        self._far_clip_z = 100.0
        self._near_clip_z = 0.5

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
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glFrontFace(gl.GL_CCW)
        gl.glClearColor(*self._clear_color)
        gl.glDisable(gl.GL_CULL_FACE)
        self._init_camera()
        self._init_default_shader()

        self.material.program_ready.connect(self._graph_changed)

        # Start an update timer to refresh rendering
        self._timer = QTimer()
        self._timer.setInterval(int(1000 / self._frame_rate))
        self._timer.timeout.connect(self.update)
        self._timer.start()

    # @pyqtSlot()
    def _graph_changed(self, program: Program):
        self._program = program
        self.material.bind_vertices(self._V)

    def _init_camera(self):
        # Translate out view in negative z dir
        glm.translate(self._world_to_view, 0, 0, -2)

    def _init_default_shader(self):
        self._shader = BrickShader()
        if self._I is None or self._V is None:
            V, I = object_vertices.get_3d_cube()
            self.set_vertices(V, I)

        self._program = self._shader.get_program(len(self._V))
        self._program.bind(self._V)
        for nf, nu, t, ra, de in self._shader.get_inputs():
            self._program[nu] = de

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if self._program is not None:
            # Send the updated transformation matrices to the shader
            self._program[OBJECT_MATRIX_NAME] = self._object_to_world
            self._program[VIEW_MATRIX_NAME] = self._world_to_view
            self._program[PROJECTION_MATRIX_NAME] = self._view_to_projection

            self._program.draw(gl.GL_TRIANGLES, self._I)

    def resizeGL(self, w: int, h: int):
        self._set_perspective_projection(w, h)

    def set_vertices(self, V: np.ndarray, I: np.ndarray):
        """
        Set the vertices to be rendered.
        :param V: An array of vertex positions
        :param I: An array of triangle indices
        """
        self._I = I.view(gloo.IndexBuffer)
        self._V = V.view(gloo.VertexBuffer)
        if self._program is not None:
            self._program.bind(self._V)

    def _set_perspective_projection(self, w: int, h: int):
        ratio = w / float(h)
        self._view_to_projection = glm.perspective(45.0, ratio, znear=self._near_clip_z, zfar=self._far_clip_z)
        self._program[PROJECTION_MATRIX_NAME] = self._view_to_projection

    def _get_texture(self) -> QImage:
        # Reset the view to default so that the texture fits to the view
        temp_obj_to_world = self._object_to_world
        temp_world_to_view = self._world_to_view
        temp_view_to_proj = self._view_to_projection
        self._object_to_world = np.eye(4, dtype=np.float32)
        self._world_to_view = np.eye(4, dtype=np.float32)
        self._view_to_projection = np.eye(4, dtype=np.float32)

        # Set the rendered object to be a plane
        temp_V, temp_I = self._V, self._I
        V, I = object_vertices.get_2d_plane()
        self.set_vertices(V, I)

        self.update()
        img: QImage = self.grabFramebuffer()

        # Reset the view
        self._object_to_world = temp_obj_to_world
        self._world_to_view = temp_world_to_view
        self._view_to_projection = temp_view_to_proj
        self.set_vertices(temp_V, temp_I)

        return img

    # ============== ============ ==============
    # ============== HANDLE EVENTS ==============
    # ============== ============ ==============
    def wheelEvent(self, wheel_event):
        scroll_steps = wheel_event.angleDelta().y() / 8 / 15  # Get actual number of steps (default is 15 deg/step)

        view_y = self._world_to_view[-1, 2]
        if (view_y >= 0 and scroll_steps > 0) or (view_y <= -self._far_clip_z and scroll_steps < 0):
            return

        distance_speedup = np.clip(abs(view_y) / 4.0, 1.0, 10.0)
        glm.translate(self._world_to_view, 0, 0, scroll_steps * self._scroll_speed * distance_speedup)

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

    def mouseReleaseEvent(self, mouse_event: QMouseEvent):

        # if a right click occurred, open context menu
        if mouse_event.button() == Qt.RightButton:
            menu = QMenu()
            save_image = menu.addAction("Export texture")
            action = menu.exec_(mouse_event.globalPos())
            if action == save_image:
                img = self._get_texture()
                filename, _ = QFileDialog.getSaveFileName(self, "Save Texture", filter="Image File (*.png)", directory="texture.png")
                if filename:
                    img.save(filename)

        # When the mouse is released, reset the last mouse positions
        self._last_mouse_x = -1
        self._last_mouse_y = -1