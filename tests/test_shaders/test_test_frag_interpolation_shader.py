from tests.stuff_for_testing.funcs import save_images, assert_abs_mean_diff, assert_abs_max_diff
from tests.stuff_for_testing.shader import ShaderTest
from tests.stuff_for_testing.shaders.test_frag_interpolation_shader import TestFragInterpolationShader


class TestTestFragInterpolationShader(ShaderTest):

    def __init__(self):
        super().__init__(TestFragInterpolationShader)

    def test_frag_interpolation_for_very_low_res(self):
        self.W = 4
        self.H = 4
        py, op = self.render_both()

        # save_images(["python_frag_interpolation_LowRes", "opengl_frag_interpolation_LowRes"], [py, op])
        assert_abs_mean_diff(py, op, tol=self.PIXEL_TOLERANCE, test_name="test_frag_interpolation_for_low_res")
        assert_abs_max_diff(py, op, self.PIXEL_TOLERANCE)

    def test_frag_interpolation_for_low_res(self):
        self.W = 30
        self.H = 30
        py, op = self.render_both()

        save_images(["python_frag_interpolation_LowRes", "opengl_frag_interpolation_LowRes"], [py, op])
        assert_abs_mean_diff(py, op, tol=self.PIXEL_TOLERANCE, test_name="test_frag_interpolation_for_low_res")
        assert_abs_max_diff(py, op, self.PIXEL_TOLERANCE)

    def test_frag_interpolation_for_high_res(self):
        self.W = 200
        self.H = 200
        py, op = self.render_both()

        # save_images(["python_frag_interpolation_HiRes", "opengl_frag_interpolation_HiRes"], [py, op])
        assert_abs_mean_diff(py, op, tol=self.PIXEL_TOLERANCE, test_name="test_frag_interpolation_for_high_res")
        assert_abs_max_diff(py, op, self.PIXEL_TOLERANCE)
