import datetime
import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageChops
from PyQt5.QtWidgets import QApplication
from glumpy.gloo import Program

from src.opengl.internal_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGB, INTERNAL_TYPE_ARRAY_RGBA
from src.shaders.shader_super import Shader
from tests.stuff_for_testing.opengl_test_renderer import OpenGLTestRenderer

_logger = logging.getLogger(__name__)


def randomize_inputs(inputs: typing.List[typing.Any]) -> typing.List[typing.Any]:
    output = []
    for _, _, internal_type, ran, default in inputs:
        output.append(randomize_input(internal_type, ran, default))

    return output


def randomize_input(internal_type, range, default):
    if INTERNAL_TYPE_ARRAY_RGB in internal_type:
        return np.random.random(default.shape)
    elif internal_type == INTERNAL_TYPE_FLOAT:
        return np.random.uniform(range[0], range[1])


def randomize_inputs_torch(args, shader: Shader, alpha_one=True) -> typing.List[torch.Tensor]:
    out = []
    for (i, arg), info in zip(enumerate(args), shader.get_inputs()):
        ran = info[3]
        typ = info[2]
        shape = arg.shape
        val = torch.from_numpy(np.random.uniform(ran[0], ran[1], shape)).float()
        if typ == INTERNAL_TYPE_ARRAY_RGBA and alpha_one:
            val[-1] = 1.

        out.append(val)

    return out


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
    return renderer.rendered_image


def render_opengl_callback_loop(width, height, program: Program, callback: typing.Callable, iter: int, freq=2):
    app = QApplication([])
    renderer = OpenGLTestRenderer(width, height, program, callback, cb_freg=60 / freq, frame_rate=60, frame_count=(60 / freq) * iter)
    renderer.show()
    app.exec_()

def assert_abs_mean_diff(pys: typing.Union[np.ndarray, list], gls: typing.Union[np.ndarray, list],
                         msg="Test failed, as absolute mean difference of {} was higher than allowed!",
                         tol=0.01, test_name: str = "?"):
    if not isinstance(pys, list) and not isinstance(gls, list):
        pys = [pys]
        gls = [gls]

    for py, gl in zip(pys, gls):
        err = np.mean(np.abs(py - gl))
        _logger.debug("Test <%s> yielded an absolute mean error of %f. Tolerance set to %f.", test_name, err, tol)

        if err > tol:
            fig = plt.figure(figsize=(12, 4))
            ax1, ax2, ax3 = fig.subplots(1, 3)
            i1 = Image.fromarray((py[:, :, :3] * 255).astype(np.uint8))
            i2 = Image.fromarray((gl[:, :, :3] * 255).astype(np.uint8))
            diff = ImageChops.difference(i1, i2)
            ax1.imshow(i1)
            ax1.set_title("Python Render")
            ax2.imshow(i2)
            ax2.set_title("GL Render")
            ax3.imshow(diff)
            ax3.set_title("Difference")
            plt.show()

        assert err <= tol, msg.format(err)


def assert_abs_max_diff(pys, gls, tol=0.01):

    if not isinstance(pys, list) and not isinstance(gls, list):
        pys = [pys]
        gls = [gls]

    for py, gl in zip(pys, gls):
        err = np.max(np.abs(py - gl))
        assert err <= tol, "Absolute max difference of {} is not under tolerance of {}".format(err, tol)
