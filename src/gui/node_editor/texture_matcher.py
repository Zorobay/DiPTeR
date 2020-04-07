import logging
import time
import typing

from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QFileDialog, QDockWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.widgets.labelled_input import LabelledInput
from src.misc import render_funcs
from src.gui.rendering.opengl_widget import OpenGLWidget
from src.gui.widgets.line_input import FloatInput, IntInput
from src.shaders.shader_super import Shader

_logger = logging.getLogger(__name__)

MAX_ITER = "max_iter"
EARLY_STOPPING_THRESH = "early_stopping_thresh"
LEARNING_RATE = "learning_rate"
DECAY = "decay"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SettingsPanel(QWidget):
    match_start = pyqtSignal()
    match_stop = pyqtSignal()
    texture_loaded = pyqtSignal(Image.Image)
    settings_changed = pyqtSignal(str, dict)  # changed key as well as all settings

    def __init__(self, *args):
        super().__init__(*args)
        # Define gui elements
        self._layout = QVBoxLayout()
        self._load_texture_button = QPushButton("Load Texture")
        self._texture_label = QLabel("load texture...")
        self._match_button = QPushButton("Match Texture")
        self._max_iter_input = IntInput(0, 10000)
        self._early_stopping_loss_thresh = FloatInput(0, 1)
        self._learning_rate = FloatInput(0, 1)

        # Define data
        self.loaded_image = None
        self._max_iter = 100
        self.settings = {}
        self._is_running = False

        self._init_widget()

    def _init_widget(self):
        self._match_button.setEnabled(False)
        self._match_button.clicked.connect(self._toggle_matching)
        self._load_texture_button.clicked.connect(self._load_texture)

        self._max_iter_input.input_changed.connect(lambda: self._settings_changed(MAX_ITER, self._max_iter_input.get_gl_value()))
        self._max_iter_input.set_default_value(100)

        self._early_stopping_loss_thresh.input_changed.connect(lambda: self._settings_changed(EARLY_STOPPING_THRESH,
                                                                                              self._early_stopping_loss_thresh.get_gl_value()))
        self._early_stopping_loss_thresh.set_default_value(0.01)

        self._learning_rate.input_changed.connect(lambda: self._settings_changed(LEARNING_RATE, self._learning_rate.get_gl_value()))
        self._learning_rate.set_default_value(0.1)

        self._layout.addWidget(self._match_button)
        self._layout.addWidget(self._load_texture_button)
        self._layout.addWidget(LabelledInput("Max iterations", self._max_iter_input))
        self._layout.addWidget(LabelledInput("Learning Rate", self._learning_rate))
        self._layout.addWidget(LabelledInput("Early stopping loss thresh", self._early_stopping_loss_thresh))

        self._layout.setAlignment(Qt.AlignTop)
        self.setLayout(self._layout)

    def _load_texture(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Texture", filter="Image File (*.png *.jpg *.jpeg *.bmp)")

        if filename:
            self.loaded_image = Image.open(filename)
            self._match_button.setEnabled(True)
            self.texture_loaded.emit(self.loaded_image)

    def set_gd_finished(self):
        self._match_button.setText("Match Texture")

    def _settings_changed(self, key: str, new_val: typing.Any):
        self.settings[key] = new_val
        _logger.debug("New value for setting {} -> {}.".format(key, new_val))
        self.settings_changed.emit(key, self.settings)

    def _toggle_matching(self):
        if not self._is_running:
            self._is_running = True
            self._match_button.setText("Stop")
            self.match_start.emit()
        else:
            self._is_running = False
            self._match_button.setText("Match Texture")
            self.match_stop.emit()


class TextureMatcher(QWidget):

    def __init__(self, shader: Shader):
        super().__init__(parent=None)

        # self._openGL = openGL
        self._shader = shader.__class__()  # Instantiate a new shader
        self._shader.set_inputs(shader.get_parameters_list_torch())

        # Define components
        self._settings_drawer = QDockWidget(self, Qt.Drawer)
        self._settings_panel = SettingsPanel(self)
        self._openGL = OpenGLWidget(400, 400, None, OpenGLWidget.TEXTURE_RENDER_MODE)
        self._figure = Figure(figsize=(4, 4), tight_layout=True)
        self._figure_canvas = FigureCanvas(self._figure)
        self._tex_axis = self._figure.add_subplot(211)
        self._loss_axis = self._figure.add_subplot(212)
        self._layout = QGridLayout()
        self._program = self._shader.get_program()

        # Define properties
        self._loss_plot_color = (1., 0.6, 0., 1.0)
        self._loss_plot_style = 'default'

        # Define data
        self._image_to_match = None
        self._settings = {}
        self.thread = None
        self.gd = None

        self._init_widget()

    def _init_widget(self):
        self._settings_panel.texture_loaded.connect(self._set_image_to_match)
        self._settings_panel.match_start.connect(self._run_gradient_descent_torch)
        self._settings_panel.match_stop.connect(self._stop_gradient_descent)
        # self._settings_panel.settings_changed.connect(self._update_settings)

        # Setup Image axis
        self._tex_axis.set_title("Target Texture")
        self._tex_axis.set_axis_off()

        # Setup Loss axis
        self._loss_axis.set_title("Loss")
        self._loss_axis.set_ylabel("Loss")
        self._loss_axis.set_xlabel("Iteration")

        # Setup openGL renderer
        self._openGL.init_done.connect(self._set_gl_program)

        self._layout.addWidget(self._openGL, 0, 0)
        self._layout.addWidget(self._figure_canvas, 0, 1)
        self._layout.addWidget(self._settings_panel, 0, 2)

        self.setLayout(self._layout)

    def _set_image_to_match(self, image):
        self._image_to_match = image
        self._tex_axis.imshow(self._image_to_match, vmin=0, vmax=1)
        self._figure_canvas.draw()

    def _set_gl_program(self):
        self._openGL.set_program(self._program)

    def _run_gradient_descent_torch(self):
        self.gd = GradientDescent(self._image_to_match, self._shader, self._settings_panel.settings)
        self.thread = QThread()
        self.gd.gd_iteration.connect(self._gd_iter_callback)
        self.gd.moveToThread(self.thread)
        self.thread.started.connect(self.gd.run)
        self.gd.finished.connect(self._finish_gradient_descent)

        _logger.debug("Started Gradient Descent Thread...")
        self.thread.start()

    def _stop_gradient_descent(self):
        if self.thread.isRunning():
            self.thread.quit()
            self._finish_gradient_descent()

    def _finish_gradient_descent(self):
        self._settings_panel.set_gd_finished()

    def _gd_iter_callback(self, props):
        self._loss_axis.plot(props['loss_hist'], color=self._loss_plot_color)
        self._loss_axis.set_xlabel("Iteration ({:.3f}s/iter)".format(props['iter_time']))
        self._figure_canvas.draw()
        self._figure_canvas.flush_events()
        params = props['params']

        self._shader.set_inputs(params)
        _logger.info("{}. loss: {}, params: {}".format(props['iter'], props['loss'], params))

    def _update_settings(self, key: str, settings: dict):
        self._settings = settings


global truth
global f


class GradientDescent(QObject):
    gd_iteration = pyqtSignal(dict)
    finished = pyqtSignal(list, np.ndarray)

    def __init__(self, image_to_match: Image.Image, shader: Shader, settings: dict):
        super().__init__()
        self.image_to_match = image_to_match
        self.shader = shader
        self.width, self.height = 100,100
        self.lr = 0.15
        self.decay = 0.97
        self.max_iter = 500
        self.early_stopping_thresh = 0.01
        self.truth = None
        self.f = None

        self._read_settings(settings)

    def _read_settings(self, settings: dict):
        for key, val in settings.items():
            if key == MAX_ITER:
                self.max_iter = val
            elif key == EARLY_STOPPING_THRESH:
                self.early_stopping_thresh = val
            elif key == LEARNING_RATE:
                self.lr = val
            elif key == DECAY:
                self.decay = val

    @pyqtSlot(name='run')
    def run(self):
        loader = transforms.Compose([
            transforms.Resize((self.width, self.height)),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor

        self.truth = loader(self.image_to_match).to(device, torch.float32)
        #self.truth = torch.from_numpy(np.asarray(self.image_to_match.resize((self.width, self.height))) / 255.)
        self.f = self.shader.shade_torch

        init_params = self.shader.get_parameters_list_torch(requires_grad=True)

        params, loss_hist = self._gradient_descent_torch(init_params, lr=self.lr, lr_decay=self.decay, max_iter=self.max_iter,
                                                         early_stopping_thresh=self.early_stopping_thresh)
        self.finished.emit(params, loss_hist)

    def _gradient_descent_torch(self, init_params:list, lr=0.01, lr_decay=0.99, max_iter=150, early_stopping_thresh=0.01) -> typing.Tuple[list,
                                                                                                                                          np.ndarray]:
        params = init_params
        loss_hist = np.empty(max_iter, dtype=np.float32)

        for i in range(max_iter):
            start = time.time()
            new_loss = self.loss_torch2(*params)
            new_loss_np = float(new_loss.detach())
            grads = torch.autograd.grad(outputs=new_loss, inputs=params, create_graph=True, retain_graph=True, allow_unused=False, only_inputs=True)
            loss_hist[i] = new_loss_np
            props = {'iter': i, 'loss': new_loss_np, 'loss_hist': loss_hist[:i + 1], 'learning_rate': lr, 'params': params}

            with torch.no_grad():
                for p, g in zip(params, grads):
                    if g is not None:
                        p -= lr * g

                lr *= lr_decay

            props['iter_time'] = time.time() - start
            self.gd_iteration.emit(props)

            if new_loss <= early_stopping_thresh:
                break

        return params, loss_hist

    def loss_torch2(self, *args):
        render = render_funcs.render_torch2(self.width, self.height, self.f, *args)
        return F.mse_loss(render, self.truth)

    def loss_torch(self, *args):
        render = render_funcs.render_torch(self.width, self.height, self.f, *args)
        return torch.mean(torch.abs(self.truth - render))

    def loss_torch_(self, *args):
        width = self.truth.shape[0]
        height = self.truth.shape[1]
        x_res = 1.0 / width
        y_res = 1.0 / height
        x_pos = torch.from_numpy(np.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0))
        y_pos = torch.from_numpy(np.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0))
        loss_sum = 0

        for x in range(width):
            for y in range(height):
                vert_pos = torch.tensor((x_pos[x], y_pos[y], 0.))
                val = self.f(vert_pos, *args)
                loss_sum += torch.sum(torch.abs(self.truth[y, x, :] - val))

        return loss_sum / (width * height * 4)
