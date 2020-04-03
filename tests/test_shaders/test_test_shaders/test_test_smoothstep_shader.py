from tests.stuff_for_testing.funcs import assert_abs_mean_diff, assert_abs_max_diff
from tests.stuff_for_testing.shader import ShaderTest
from tests.stuff_for_testing.shaders.test_smoothstep_shader import TestSmoothstepShader


class TestTestSmoothstepShader(ShaderTest):

    def __init__(self):
        super().__init__(TestSmoothstepShader)

    def test_smoothstep_for_very_low_res(self):
        self.W = 5
        self.H = 5
        py, op = self.render_both()

        assert_abs_mean_diff(py, op, tol=self.PIXEL_TOLERANCE, test_name="test_smoothstep_for_low_res")
        assert_abs_max_diff(py, op, tol=self.PIXEL_TOLERANCE)

    def test_smoothstep_for_low_res(self):
        self.W = 50
        self.H = 50
        py, op = self.render_both()

        assert_abs_mean_diff(py, op, tol=self.PIXEL_TOLERANCE, test_name="test_smoothstep_for_low_res")
        assert_abs_max_diff(py, op, tol=self.PIXEL_TOLERANCE)

    def test_smoothstep_for_high_res(self):
        self.W = 200
        self.H = 200
        py, op = self.render_both()

        assert_abs_mean_diff(py, op, tol=self.PIXEL_TOLERANCE, test_name="test_smoothstep_for_high_res")
        assert_abs_max_diff(py, op, tol=self.PIXEL_TOLERANCE)
