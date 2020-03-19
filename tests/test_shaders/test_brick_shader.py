import logging
import sys
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow

from src.misc.render_funcs import render
from src.opengl.object_vertices import get_2d_plane
from src.shaders.brick_shader import BrickShader
from tests.stuff_for_testing.funcs import save_images, render_opengl, assert_abs_mean_diff, randomize_inputs
from tests.stuff_for_testing.opengl_renderer import OpenGLTestRenderer

_logger = logging.getLogger(__name__)


class TestBrickShader:

    @classmethod
    def setup_class(cls):
        cls.AVG_PIXEL_TOLERANCE = 0.005
        cls.W = 100
        cls.H = 100

    def setup(self):
        self.shader = BrickShader()
        self.V, self.I = get_2d_plane()
        self.program = self.shader.get_program(len(self.V), set_defaults=True)
        #self.program.bind(self.V)
        self.default_args = [t[-1] for t in self.shader.get_inputs()]

    def render_both(self):
        python_render = render(self.W, self.H, self.shader.shade, *self.default_args)
        opengl_render = render_opengl(self.W, self.H, self.program)
        return python_render, opengl_render

    def set_uniform_and_params(self, key, index, value):
        self.program[key] = value
        self.default_args[index] = value

    def test_pixel_likeness_default_parameters(self):
        python_render, opengl_render = self.render_both()
        assert_abs_mean_diff(python_render, opengl_render, "The average absolute pixel difference of {} is too large!")

    def test_color_clipping(self):
        self.set_uniform_and_params("color_brick", 4, np.array((2.0, -1, 0.0, 1.0)))

        py, op = self.render_both()
        save_images(["python_brick_clipping", "opengl_brick_clipping"], [py, op])
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when clipping colors!")

    def test_brick_elongate_zero(self):
        key = "brick_elongate"
        key_i = 2
        val = 0.0
        self.set_uniform_and_params(key, key_i, val)
        python_render, opengl_render = self.render_both()
        assert_abs_mean_diff(python_render, opengl_render, "Average pixel difference of {} is too large for zero 'brick_elongate'!")

    def test_brick_elongate_small(self):
        key = "brick_elongate"
        key_i = 2
        val = 1.0
        self.set_uniform_and_params(key, key_i, val)
        py, op = self.render_both()
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when 'brick_elongate' is low.")

    def test_brick_elongate_max(self):
        key = "brick_elongate"
        key_i = 2
        val = 100.
        self.set_uniform_and_params(key, key_i, val)
        py, op = self.render_both()
        assert_abs_mean_diff(py, op, "Average pixel difference of {} is too large when 'brick_elongate' is at 100 (max)!")

    # def test_brick_random(self):
    #     self.W = 100
    #     self.H = 100
    #
    #     def handle(texture):
    #         rand_inputs = randomize_inputs(self.shader.get_inputs())
    #         for tup, input_widgets in zip(self.shader.get_inputs(), rand_inputs):
    #             self.program[tup[1]] = input_widgets
    #
    #         python_render = render(self.W, self.H, self.shader.shade, *self.default_args)
    #         assert_abs_mean_diff(python_render, 3)
    #
    #     app = QApplication(sys.argv)
    #     renderer = OpenGLTestRenderer(self.W, self.H, self.program, frame_count=-1)
    #
    #     window = QMainWindow()
    #     window.setCentralWidget(renderer)
    #     window.show()
    #     renderer.render_ready.connect(lambda x: handle(x))
    #
    #     #renderer.show()
    #     #window.callback_pb_load()
    #     renderer.renderN(40)
    #     #renderer.close()
