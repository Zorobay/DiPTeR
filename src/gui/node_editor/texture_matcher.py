import typing

import autograd.numpy as anp
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QFileDialog
from autograd import grad
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.rendering.opengl_widget import OpenGLWidget
from src.shaders.shader_super import Shader


class TextureMatcher(QWidget):

    def __init__(self, shader: Shader):
        super().__init__(parent=None)

        # self._openGL = openGL
        self._shader = shader

        # Define components
        self._load_texture_button = QPushButton("Load Texture")
        self._texture_label = QLabel("load texture...")
        self._match_button = QPushButton("Match Texture")
        self._openGL = OpenGLWidget(400, 400, None, OpenGLWidget.TEXTURE_RENDER_MODE)
        self._figure = Figure(figsize=(4, 4), tight_layout=True)
        self._figure_canvas = FigureCanvas(self._figure)
        self._tex_axis = self._figure.add_subplot(211)
        self._loss_axis = self._figure.add_subplot(212)
        self._layout = QGridLayout()

        # Define data
        self._image_to_match = None

        self._init_widget()

    def _init_widget(self):
        self._match_button.setEnabled(False)
        self._match_button.clicked.connect(self._run_gradient_descent_torch)

        self._load_texture_button.clicked.connect(self._load_texture)

        # Setup Image axis
        self._tex_axis.set_title("Target Texture")
        self._tex_axis.set_axis_off()

        # Setup Loss axis
        self._loss_axis.set_title("Loss")
        self._loss_axis.set_ylabel("Loss")
        self._loss_axis.set_xlabel("Iteration")

        # Setup openGL renderer
        self._openGL.init_done.connect(self._set_gl_program)

        self._layout.addWidget(self._match_button, 0, 0)
        self._layout.addWidget(self._openGL, 1, 0)
        self._layout.addWidget(self._load_texture_button, 0, 1)
        self._layout.addWidget(self._figure_canvas, 1, 1)

        self.setLayout(self._layout)

    def _load_texture(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Texture", filter="Image File (*.png *.jpg *.jpeg *.bmp)")

        if filename:
            self._image_to_match = Image.open(filename)
            self._tex_axis.imshow(self._image_to_match)
            self._figure_canvas.draw()
            self._match_button.setEnabled(True)

    def _set_gl_program(self):
        program = self._shader.get_program_copy()
        self._openGL.set_program(program)

    def _run_gradient_descent_torch(self):
        width, height = 50, 50
        lr = 0.15
        decay = 0.97
        max_iter = 100

        global truth
        global f
        truth = torch.from_numpy(np.asarray(self._image_to_match.resize((width, height))) / 255.)
        f = self._shader.shade_torch

        init_params = self._shader.get_parameters_list_torch(requires_grad=True)

        def callback(losses: list, lr: float):
            self._loss_axis.plot(losses)
            self._figure_canvas.draw()

        gradient_descent_torch(init_params, lr=lr, lr_decay=decay, max_iter=max_iter, early_stopping_thresh=0.01, iter_callback=callback)

    def _run_gradient_descent(self):
        width, height = 50, 50
        lr = 0.15
        decay = 0.97
        max_iter = 100

        global truth
        global f
        truth = np.asarray(self._image_to_match.resize((width, height))) / 255.
        f = self._shader.shade

        loss_grad = grad()
        init_params = self._shader.get_parameters_list()

        gradient_descent(init_params, lr=lr, lr_decay=decay, max_iter=max_iter, early_stopping_thresh=0.01)


global truth
global f


def loss(*args):
    width = truth.shape[0]
    height = truth.shape[1]
    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = anp.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0)
    y_pos = anp.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0)
    loss_sum = 0

    for x in range(width):
        for y in range(height):
            vert_pos = anp.array((x_pos[x], y_pos[y], 0.))
            val = f(vert_pos, *args)
            loss_sum += anp.sum(anp.abs(truth[y, x, :] - val))

    return loss_sum / (width * height * 4)


def loss_torch(*args):
    width = truth.shape[0]
    height = truth.shape[1]
    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = torch.from_numpy(anp.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0))
    y_pos = torch.from_numpy(anp.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0))
    loss_sum = 0

    for x in range(width):
        for y in range(height):
            vert_pos = torch.tensor((x_pos[x], y_pos[y], 0.))
            val = f(vert_pos, *args)
            loss_sum += torch.sum(torch.abs(truth[y, x, :] - val))

    return loss_sum / (width * height * 4)

def gradient_descent(loss_grad, init_params, lr=0.01, lr_decay=0.99, max_iter=150, early_stopping_thresh=0.01):
    params = init_params
    loss_hist = []

    for i in range(max_iter):
        gradient = loss_grad(*params)
        params.subtract([lr * g for g in gradient])
        new_loss = loss(*params)
        loss_hist.append(new_loss)

        print("{}. new loss: {:.5f}, lr: {:.5f}, Params: {}".format(i, new_loss, lr, params))
        if new_loss <= early_stopping_thresh:
            return params, loss_hist
        lr = lr * lr_decay

    return params, loss_hist

def gradient_descent_torch(init_params, lr=0.01, lr_decay=0.99, max_iter=150, early_stopping_thresh=0.01, iter_callback: typing.Callable = None):
    params = init_params
    loss_hist = []

    for i in range(max_iter):
        new_loss = loss_torch(*params)
        grads = torch.autograd.grad(outputs=new_loss, inputs=params, create_graph=True, retain_graph=True, allow_unused=True)
        loss_hist.append(new_loss)

        # for p in params:
        #     p.retain_grad()
        # new_loss.backward(params, retain_graph=True)

        print("{}. new loss: {:.5f}, lr: {:.5f}, Params: {}".format(i, new_loss, lr, params))

        with torch.no_grad():
            for p,g in zip(params, grads):
                if g is not None:
                    p -= lr * g

            lr *= lr_decay

            # for p in params:
            #     p.grad.zero_()

        if new_loss <= early_stopping_thresh:
            return params, loss_hist

        if iter_callback:
            iter_callback(loss_hist, lr)

    return params, loss_hist
