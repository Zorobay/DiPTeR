import numpy as np
import typing
from PIL import Image
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QSurfaceFormat, QColorSpace
from PyQt5.QtWidgets import QOpenGLWidget
from glumpy import gl
from glumpy.gloo import Program

from src.opengl import object_vertices
from src.shaders import OBJECT_MATRIX_NAME, VIEW_MATRIX_NAME, PROJECTION_MATRIX_NAME


class OpenGLTestRenderer(QOpenGLWidget):
    render_ready = pyqtSignal(object)

    def __init__(self, width: int, height: int, program: Program, callback: typing.Callable=None, cb_freg=0, frame_rate=60, frame_count=10):
        super().__init__()

        # Set Widget settings
        self.setFixedSize(width, height)
        self.setMouseTracking(False)

        # Define variables to track objects
        self.program = program
        self._timer = None
        self.I = None
        self.V = None
        self.callback = callback
        self.cb_freg = cb_freg
        self.rendered_image = None

        # Define variables for settings
        self._clear_color = np.array((0.,0.,0.,0.), dtype=np.float32)
        self._frame_rate = frame_rate
        self._frame_count = frame_count
        self._current_frame = 0
        self._target_frame = -1
        self._far_clip_z = 100.0
        self._near_clip_z = 0.5

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
        gl.glShadeModel(gl.GL_FLAT)
        gl.glDisable(gl.GL_BLEND)
        gl.glFrontFace(gl.GL_CCW)
        gl.glClearColor(*self._clear_color)
        gl.glDisable(gl.GL_CULL_FACE)

        # Setup object
        self.V, self.I = object_vertices.get_2d_plane()
        self.program.bind(self.V)

        # Send the updated transformation matrices to the shader
        self.program[OBJECT_MATRIX_NAME] = self._object_to_world
        self.program[VIEW_MATRIX_NAME] = self._world_to_view
        self.program[PROJECTION_MATRIX_NAME] = self._view_to_projection

        # Start an update timer to refresh rendering
        self._timer = QTimer()
        self._timer.setInterval(int(1000 / self._frame_rate))
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.program.draw(gl.GL_TRIANGLES, self.I)
        self._current_frame += 1

        if self.callback and (self._current_frame % self.cb_freg) == 0 and self._current_frame > 0:
            self.callback(self.get_image())

        if self._current_frame >= self._frame_count:
            self.rendered_image = self.get_image()
            self.close()

    def get_image(self) -> np.ndarray:
        img: QImage = self.grabFramebuffer()
        img = img.convertToFormat(QImage.Format_RGBX8888)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = (np.array(ptr).reshape(img.height(), img.width(), 4) / 255.).astype(np.float32)
        return arr[:,:,:3]
