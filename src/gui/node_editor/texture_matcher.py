import logging
import time
import typing

import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QFileDialog, QDockWidget, QVBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.rendering.opengl_widget import OpenGLWidget
from src.gui.widgets.labelled_input import LabelledInput
from src.gui.widgets.line_input import FloatInput, IntInput
from src.misc import render_funcs, losses, image_funcs
from src.shaders.shader_super import FunctionShader

_logger = logging.getLogger(__name__)

LOSS_FUNC = "loss_func"
MAX_ITER = "max_iter"
EARLY_STOPPING_THRESH = "early_stopping_thresh"
LEARNING_RATE = "learning_rate"
DECAY = "decay"

ren_i = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

CHANNELS = 3


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
        self._loss_combo_box = QComboBox()
        self._max_iter_input = IntInput(0, 10000)
        self._early_stopping_loss_thresh = FloatInput(0, 1)
        self._learning_rate = FloatInput(0, 1)

        # Define data
        self._loss_func_map = {"Squared Bin Loss": losses.squared_bin_loss, "MSE Loss": losses.mean_squared_error}
        self._selected_loss_func = None
        self.loaded_image = None
        self._max_iter = 100
        self.settings = {}
        self._is_running = False

        self._init_widget()

    def _init_widget(self):
        self._match_button.setEnabled(False)
        self._match_button.clicked.connect(self._toggle_matching)
        self._load_texture_button.clicked.connect(self._load_texture)

        self._loss_combo_box.addItems(list(self._loss_func_map))
        self._loss_combo_box.currentTextChanged.connect(self._set_loss_func)
        self._loss_combo_box.setCurrentIndex(0)

        self._max_iter_input.input_changed.connect(lambda: self._settings_changed(MAX_ITER, self._max_iter_input.get_gl_value()))
        self._max_iter_input.set_default_value(100)

        self._early_stopping_loss_thresh.input_changed.connect(lambda: self._settings_changed(EARLY_STOPPING_THRESH,
                                                                                              self._early_stopping_loss_thresh.get_gl_value()))
        self._early_stopping_loss_thresh.set_default_value(0.01)

        self._learning_rate.input_changed.connect(lambda: self._settings_changed(LEARNING_RATE, self._learning_rate.get_gl_value()))
        self._learning_rate.set_default_value(0.1)

        self._layout.addWidget(self._match_button)
        self._layout.addWidget(self._load_texture_button)
        self._layout.addWidget(LabelledInput("Loss primary_function", self._loss_combo_box))
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

    def _set_loss_func(self, key: str):
        f = self._loss_func_map[key]
        self._selected_loss_func = f
        self._settings_changed(LOSS_FUNC, self._selected_loss_func)

    def set_gd_finished(self):
        self._match_button.setText("Match Texture")

    def set_gd_finishing(self):
        self._match_button.setText("Stopping...")

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

    def __init__(self, shader: FunctionShader):
        super().__init__(parent=None)

        # self._openGL = openGL
        self._shader = shader.__class__()  # Instantiate a new shader
        self._shader.set_inputs(shader.get_parameters_list())

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
        self.ren_i = 0

        self._init_widget()

    def _init_widget(self):
        self._settings_panel.texture_loaded.connect(self._set_image_to_match)
        self._settings_panel.match_start.connect(self._run_gradient_descent_torch)
        self._settings_panel.match_stop.connect(self._stop_gradient_descent)
        # self._settings_panel.settings_changed.connect(self._update_settings)

        # Setup Image axis
        self._tex_axis.set_title("Target Texture")
        # self._tex_axis.set_axis_off()

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
        self._image_to_match = image.convert("RGB")
        # We want to display the same image that we are testing the loss against. This image is columns majos (x,y)
        # which is not what matplotlib want's so we have to transpose it back to row major
        tensor_image = image_funcs.image_to_tensor(image, (100, 100)).transpose(0, 1)

        self._tex_axis.imshow(tensor_image, vmin=0, vmax=1)
        self._tex_axis.invert_yaxis()
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
            _logger.info("Stopping Gradient Descent Thread...")
            self._settings_panel.set_gd_finishing()
            self.gd.stop()
            self.thread.quit()

    def _finish_gradient_descent(self):
        self._settings_panel.set_gd_finished()
        _logger.info("Gradient Descent Thread Stopped.")

    def _gd_iter_callback(self, props):
        if not self.ren_i % 1 == 0:
            self.ren_i += 1
            return
        self.ren_i += 1
        self._loss_axis.plot(props['loss_hist'], '.-', color=self._loss_plot_color, label="Loss")
        self._loss_axis.set_xlabel("Iteration ({:.3f}s/iter)".format(props['iter_time']))
        self._figure_canvas.draw()
        self._figure_canvas.flush_events()
        params = props['params']

        self._shader.set_inputs(params)
        _logger.info("{}. loss: {}, params: {}".format(props['iter'], props['loss'], params))

    def _update_settings(self, key: str, settings: dict):
        self._settings = settings


class GradientDescent(QObject):
    gd_iteration = pyqtSignal(dict)
    finished = pyqtSignal(list, np.ndarray)

    def __init__(self, image_to_match: Image.Image, shader: FunctionShader, settings: dict, optimizer: torch.optim.Optimizer = torch.optim.Adam):
        super().__init__()
        self.image_to_match = image_to_match
        self.shader = shader
        self.optimizer = optimizer
        self.width, self.height = 100, 100
        self.lr = 0.15
        self.decay = 0.97
        self.max_iter = 500
        self.early_stopping_thresh = 0.01
        self.truth = None
        self.f = None
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.loss_func = self.mse_loss
        self._stop = False

        self._read_settings(settings)

    def _read_settings(self, settings: dict):
        for key, val in settings.items():
            if key == LOSS_FUNC:
                self.loss_func = val
            elif key == MAX_ITER:
                self.max_iter = val
            elif key == EARLY_STOPPING_THRESH:
                self.early_stopping_thresh = val
            elif key == LEARNING_RATE:
                self.lr = val
            elif key == DECAY:
                self.decay = val

    def stop(self):
        self._stop = True

    @pyqtSlot(name='run')
    def run(self):
        self.truth = image_funcs.image_to_tensor(self.image_to_match, (self.width, self.height))
        self.f = self.shader.shade

        init_params = self.shader.get_parameters_list(requires_grad=True, randomize=False)

        params, loss_hist = self._run_gd(init_params, lr=self.lr, max_iter=self.max_iter, early_stopping_thresh=self.early_stopping_thresh)
        self.finished.emit(params, loss_hist)

    def _run_gd(self, init_params: list, lr=0.01, max_iter=150, early_stopping_thresh=0.01) -> typing.Tuple[list, np.ndarray]:
        P = init_params
        optimizer = self.optimizer(P, lr=lr)
        loss_hist = np.empty(max_iter, dtype=np.float32)

        for i in range(max_iter):
            if self._stop:
                return P, loss_hist

            optimizer.zero_grad()
            start = time.time()
            new_loss = self.render_and_loss(*P)
            new_loss_np = float(new_loss.detach())
            loss_hist[i] = new_loss_np
            props = {'iter': i, 'loss': new_loss_np, 'loss_hist': loss_hist[:i + 1], 'learning_rate': lr, 'params': P, 'iter_time': 0.0}

            # We need to break here, otherwise the parameters will change when we call optimizer.step()
            if new_loss <= early_stopping_thresh:
                self.gd_iteration.emit(props)
                break

            new_loss.backward(retain_graph=False, create_graph=False)

            optimizer.step()

            props['iter_time'] = time.time() - start
            self.gd_iteration.emit(props)

        return P, loss_hist

    def render_and_loss(self, *args):
        # Ps = [p.repeat(self.width, self.height, 1) for p in args]
        render = render_funcs.render_torch_loop(self.width, self.height, self.f, *args)
        return self.loss_func(render, self.truth)
