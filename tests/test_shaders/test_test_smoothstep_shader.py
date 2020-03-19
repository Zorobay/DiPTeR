from src.opengl.object_vertices import get_2d_plane
from tests.stuff_for_testing.funcs import render_opengl, save_images, assert_abs_mean_diff
from tests.stuff_for_testing.test_shaders.test_smoothstep_shader import TestSmoothstepShader
from src.misc.render_funcs import render


class TestTestSmoothstepShader:

    def setup(self):
        self.V, self.I = get_2d_plane()
        self.shader = TestSmoothstepShader()
        self.program = self.shader.get_program(len(self.V), set_defaults=True)
        self.program.bind(self.V)

        self.default_args = [t[-1] for t in self.shader.get_inputs()]
        self.W = 100
        self.H = 100

        self.AVG_PIXEL_TOLERANCE = 0.001

    def render_both(self):
        python_render = render(self.W, self.H, self.shader.shade, *self.default_args)
        opengl_render = render_opengl(self.W, self.H, self.program)
        return python_render, opengl_render

    def test_smoothstep_for_very_low_res(self):
        self.W = 5
        self.H = 5
        py, op = self.render_both()

        assert_abs_mean_diff(py,op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_smoothstep_for_low_res")

    def test_smoothstep_for_low_res(self):
        self.W = 50
        self.H = 50
        py, op = self.render_both()

        assert_abs_mean_diff(py,op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_smoothstep_for_low_res")

    def test_smoothstep_for_high_res(self):
        self.W = 200
        self.H = 200
        py, op = self.render_both()

        assert_abs_mean_diff(py,op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_smoothstep_for_high_res")
