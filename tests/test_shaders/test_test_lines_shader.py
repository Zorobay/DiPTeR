from tests.stuff_for_testing.shader import ShaderTest
from tests.stuff_for_testing.funcs import assert_abs_mean_diff, assert_abs_max_diff
from tests.stuff_for_testing.shader import ShaderTest
from tests.stuff_for_testing.shaders.test_lines_shader import TestLinesShader


class TestTestLinesShader(ShaderTest):

    def __init__(self):
        super().__init__(TestLinesShader)

    def test_alignment_for_low_res(self):
        self.W = 50
        self.H = 50
        py, gl = self.render_both()

        # save_images(["python_alignment_LowRes", "opengl_alignment_LowRes"], [py, op])
        assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE, test_name="test_alignment_for_low_res")
        assert_abs_max_diff(py, gl, self.PIXEL_TOLERANCE)

    def test_alignment_for_high_res(self):
        self.W = 200
        self.H = 200
        py, gl = self.render_both()

        # save_images(["python_alignment_HiRes", "opengl_alignment_HiRes"], [py, op])
        assert_abs_mean_diff(py, gl, tol=self.PIXEL_TOLERANCE, test_name="test_alignment_for_high_res")
        assert_abs_max_diff(py, gl, self.PIXEL_TOLERANCE)
