import datetime
import logging
import typing

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication
from glumpy.gloo import Program

from src.opengl.shader_types import INTERNAL_TYPE_RGB, INTERNAL_TYPE_FLOAT
from tests.stuff_for_testing.opengl_renderer import OpenGLTestRenderer

_logger = logging.getLogger(__name__)

def randomize_inputs(inputs: typing.List[typing.Any]) -> typing.List[typing.Any]:
    output = []
    for _,_,internal_type,ran,default in inputs:
        if INTERNAL_TYPE_RGB in internal_type:
            output.append(np.random.random(default.shape))
        elif internal_type == INTERNAL_TYPE_FLOAT:
            output.append(np.random.uniform(ran[0],ran[1]))

    return output

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
