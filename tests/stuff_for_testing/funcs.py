import datetime
import logging

import numpy as np
from PIL import Image
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QOpenGLWidget, QApplication
from glumpy import gloo, gl
from glumpy.gloo import Program

from src.opengl import object_vertices
from src.shaders import OBJECT_MATRIX_NAME, VIEW_MATRIX_NAME, PROJECTION_MATRIX_NAME

_logger = logging.getLogger(__name__)


class OpenGLTestRenderer(QOpenGLWidget):

    def __init__(self, width: int, height: int, program: Program, frame_rate=60, frame_count=20):
        super().__init__()

        # Set Widget settings
        self.setFixedSize(width, height)
        self.setMouseTracking(False)

        # Define variables to track objects
        self.program = program
        self._timer = None
        self.I = None
        self.V = None
        tex = np.zeros((width, height, 4), dtype=np.float32).view(gloo.TextureFloat2D)
        self.framebuffer = gloo.FrameBuffer(color=[tex])
        self.rendered_tex = None

        # Define variables for settings
        self._clear_color = np.array((0.4, 0.4, 0.4, 1.0), dtype=np.float32)
        self._frame_rate = frame_rate
        self._frame_count = frame_count
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
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
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
        if self._frame_count <= 0:
            self.close()

        self._frame_count -= 1
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.framebuffer.activate()
        self.program.draw(gl.GL_TRIANGLES, self.I)
        self.rendered_tex = self.framebuffer.color[0].get()
        self.framebuffer.deactivate()


def save_images(filenames, images):
    now = datetime.datetime.now().strftime("%H-%M-%S")
    if isinstance(images, list):
        for fn, im in zip(filenames, images):
            Image.fromarray((im * 255.).astype(np.uint8)).save("test_image_output/{}_{}.png".format(fn, now))
    else:
        Image.fromarray((images * 255.).astype(np.uint8)).save("test_image_output/{}_{}.png".format(filenames, now))


def render_opengl(width, height, program: Program):
    app = QApplication([])
    renderer = OpenGLTestRenderer(width, height, program)
    renderer.show()
    app.exec_()
    return renderer.rendered_tex


def assert_abs_mean_diff(render1: np.ndarray, render2: np.ndarray, msg="Test failed, as absolute mean difference of {} was higher than allowed!",
                         tol=0.01, test_name: str = "?"):
    err = np.mean(np.abs(render1 - render2))
    _logger.debug("Test <%s> yielded an absolute mean error of %f. Tolerance set to %f.", test_name, err, tol)
    assert err <= tol, msg.format(err)
