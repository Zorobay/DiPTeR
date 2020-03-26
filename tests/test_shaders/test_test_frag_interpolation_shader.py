from src.misc.render_funcs import render
from src.opengl.object_vertices import get_2d_plane
from tests.stuff_for_testing.funcs import render_opengl, save_images, assert_abs_mean_diff
from tests.stuff_for_testing.shaders.test_frag_interpolation_shader import TestFragInterpolationShader


class TestTestFragInterpolationShader:

    def setup(self):
        self.V, self.I = get_2d_plane()
        self.shader = TestFragInterpolationShader()
        self.program = self.shader.get_program()
        self.program.bind(self.V)

        self.default_args = [t[-1] for t in self.shader.get_inputs()]
        self.W = 100
        self.H = 100

        self.AVG_PIXEL_TOLERANCE = 0.001

    def render_both(self):
        python_render = render(self.W, self.H, self.shader.shade, *self.default_args)
        opengl_render = render_opengl(self.W, self.H, self.program)
        return python_render, opengl_render

    def test_frag_interpolation_for_very_low_res(self):
        self.W = 4
        self.H = 4
        py, op = self.render_both()

        #save_images(["python_frag_interpolation_LowRes", "opengl_frag_interpolation_LowRes"], [py, op])
        assert_abs_mean_diff(py, op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_frag_interpolation_for_low_res")

    def test_frag_interpolation_for_low_res(self):
        self.W = 30
        self.H = 30
        py, op = self.render_both()

        save_images(["python_frag_interpolation_LowRes", "opengl_frag_interpolation_LowRes"], [py, op])
        assert_abs_mean_diff(py, op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_frag_interpolation_for_low_res")

    def test_frag_interpolation_for_high_res(self):
        self.W = 200
        self.H = 200
        py, op = self.render_both()

        #save_images(["python_frag_interpolation_HiRes", "opengl_frag_interpolation_HiRes"], [py, op])
        assert_abs_mean_diff(py, op, tol=self.AVG_PIXEL_TOLERANCE, test_name="test_frag_interpolation_for_high_res")
