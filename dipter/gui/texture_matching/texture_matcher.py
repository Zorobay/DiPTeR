import inspect
import logging
import typing

import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PIL import Image
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QFileDialog, QDockWidget, QVBoxLayout, QComboBox, QMenuBar
from gui.texture_matching.loss_visualizer import LossVisualizer
from node_graph.parameter import Parameter
from dipter.gui.node_editor.g_shader_node import GMaterialOutputNode
from dipter.gui.rendering.image_plotter import ImagePlotter
from dipter.gui.rendering.opengl_widget import OpenGLWidget
from dipter.gui.widgets.node_input.labelled_input import LabelledInput
from dipter.gui.widgets.node_input.line_input import FloatInput, IntInput
from dipter.misc import image_funcs
from dipter.optimization import losses
from dipter.optimization.gradient_descent import GradientDescent, GradientDescentSettings
from torch import optim, nn

sns.set()

_logger = logging.getLogger(__name__)

device = "cpu"

CHANNELS = 3


class SettingsPanel(QWidget):
    match_start = pyqtSignal()
    match_stop = pyqtSignal()
    reset_requested = pyqtSignal()
    texture_loaded = pyqtSignal(Image.Image)
    settings_changed = pyqtSignal(GradientDescentSettings)  # changed key as well as all settings

    def __init__(self, *args):
        super().__init__(*args)
        # Define gui elements
        self._layout = QVBoxLayout()
        self._load_texture_button = QPushButton("Load Texture")
        self._texture_label = QLabel("load texture...")
        self._match_button = QPushButton("Match Texture")
        self._loss_combo_box = QComboBox()
        self._optimizer_combo_box = QComboBox()
        self._width_input = IntInput(0, 1000)
        self._height_input = IntInput(0, 1000)
        self._max_iter_input = IntInput(0, 10000)
        self._early_stopping_loss_thresh = FloatInput(0, 1)
        self._learning_rate = FloatInput(0, 1)

        # Define data
        self._loss_func_map = {"MSE Loss": losses.MSELoss(reduction='mean'), "Squared Bin Loss": losses.SquaredBinLoss(), "Neural Loss":
            losses.NeuralLoss()}

        self._optimizer_map = {"Adam": optim.Adam, "AdamW": optim.AdamW, "Adagrad": optim.Adagrad, "RMSprop": optim.RMSprop}
        self.loaded_image = None
        self._max_iter = 100
        self.settings = GradientDescentSettings()
        self._is_running = False
        self._is_cleared = True

        self._init_widget()

    def _init_widget(self):
        self._match_button.setEnabled(False)
        self._match_button.clicked.connect(self._toggle_matching)
        self._load_texture_button.clicked.connect(self._load_texture)

        # --- Setup loss function combo box ---
        self._loss_combo_box.addItems(list(self._loss_func_map))
        self._loss_combo_box.currentIndexChanged.connect(self._set_loss_func)
        self._loss_combo_box.setCurrentIndex(0)
        self.settings.loss_func = self._loss_func_map[self._loss_combo_box.currentText()]

        # --- Setup optimizer combo box ---
        self._optimizer_combo_box.addItems(list(self._optimizer_map))
        self._optimizer_combo_box.currentTextChanged.connect(self._set_optimizer)
        self.settings.optimizer = self._optimizer_map["Adam"]

        # --- Setup render size input ---
        self._width_input.set_value(self.settings.render_width)
        self._height_input.set_value(self.settings.render_height)
        self._width_input.input_changed.connect(lambda: self._change_settings("render_width", self._width_input.get_gl_value()))
        self._height_input.input_changed.connect(lambda: self._change_settings("render_height", self._height_input.get_gl_value()))

        # --- Setup max iterations input ---
        self._max_iter_input.input_changed.connect(lambda: self._change_settings("max_iter", self._max_iter_input.get_gl_value()))
        self._max_iter_input.set_value(100)

        # --- Setup early stopping loss threshold input ---
        self._early_stopping_loss_thresh.input_changed.connect(lambda: self._change_settings("early_stopping_thresh",
                                                                                             self._early_stopping_loss_thresh.get_gl_value()))
        self._early_stopping_loss_thresh.set_value(0.01)

        # --- Setup learning rate input ---
        self._learning_rate.input_changed.connect(lambda: self._change_settings("learning_rate", self._learning_rate.get_gl_value()))
        self._learning_rate.set_value(0.1)

        self._layout.addWidget(self._match_button)
        self._layout.addWidget(self._load_texture_button)
        self._layout.addWidget(LabelledInput("Loss function", self._loss_combo_box))
        self._layout.addWidget(LabelledInput("Optimizer", self._optimizer_combo_box))
        self._layout.addWidget(LabelledInput("Render width", self._width_input))
        self._layout.addWidget(LabelledInput("Render height", self._height_input))
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

    def _set_loss_func(self, index: int):
        loss_func = self._loss_func_map[self._loss_combo_box.currentText()]

        if isinstance(loss_func, losses.NeuralLoss):
            self._width_input.set_value(224)
            self._height_input.set_value(224)

        self._change_settings("loss_func", loss_func)

    def _set_optimizer(self, text: str):
        optimizer = self._optimizer_map[text]
        self._change_settings("optimizer", optimizer)

    def set_gd_finished(self):
        self._match_button.setText("Reset")
        self._is_running = False

    def set_gd_finishing(self):
        self._match_button.setText("Stopping...")

    def _change_settings(self, var: str, new_val: typing.Any):
        _logger.debug("New value for setting {} -> {}.".format(var, new_val))
        setattr(self.settings, var, new_val)
        self.settings_changed.emit(self.settings)

    def _toggle_matching(self):
        if not self._is_running:
            if not self._is_cleared:
                self._is_cleared = True
                self.reset_requested.emit()
                self._match_button.setText("Match Texture")
            else:
                self._is_running = True
                self._is_cleared = False
                self._match_button.setText("Stop")
                self.match_start.emit()
        else:
            self._is_running = False
            self._is_cleared = False
            self._match_button.setText("Reset")
            self.match_stop.emit()


class TextureMatcher(QWidget):

    def __init__(self, mat_output_node: GMaterialOutputNode):
        super().__init__(parent=None)

        self._out_node = mat_output_node

        # Define components
        self._menu_bar = QMenuBar()
        self._settings_drawer = QDockWidget(self, Qt.Drawer)
        self._settings_panel = SettingsPanel(self)
        self._openGL = OpenGLWidget(400, 400, None, OpenGLWidget.TEXTURE_RENDER_MODE)
        self._image_plotter = ImagePlotter("Render")
        self._target_plotter = ImagePlotter("Target")
        self._loss_plotter = pg.PlotWidget(title="Loss")
        self._layout = QGridLayout()
        self._shader = self._out_node.get_shader()
        self._program = self._out_node.get_program()
        self._loss_visualizer = LossVisualizer(mat_output_node)

        # Define properties
        self._loss_plot_color = (1., 0.6, 0., 1.0)
        self._loss_plot_style = 'default'

        # Define data
        self._target_image = None
        self._target_matrix = None
        self._settings = {}
        self.thread = None
        self.gd = None
        self.ren_i = 0
        self._params = {}

        self._init_widget()

    def _init_widget(self):
        self.setWindowTitle("Texture Matcher")

        # Setup menu
        visualize_menu = self._menu_bar.addMenu("visualize")
        visualize_loss_action = visualize_menu.addAction("visualize loss")
        visualize_loss_action.triggered.connect(self._open_loss_viz_window)

        # Setup settings panel
        self._settings_panel.texture_loaded.connect(self._set_image_to_match)
        self._settings_panel.match_start.connect(self._run_gradient_descent_torch)
        self._settings_panel.match_stop.connect(self._stop_gradient_descent)
        self._settings_panel.reset_requested.connect(self._reset)
        self._settings_panel.setMaximumWidth(250)

        # Setup plots
        self._openGL.init_done.connect(self._set_gl_program)
        self._image_plotter.hide_axes()
        self._image_plotter.set_user_input(False)
        self._target_plotter.hide_axes()
        self._target_plotter.set_user_input(False)

        # Setup openGL rendering window
        opengl_layout = QVBoxLayout()
        opengl_layout.setContentsMargins(20, 0, 20, 10)
        opengl_title = QLabel("OpenGL Render")
        opengl_title_font = QFont()
        opengl_title_font.setPointSize(14)
        opengl_title.setFont(opengl_title_font)
        opengl_layout.addWidget(opengl_title, alignment=Qt.AlignHCenter)
        opengl_layout.addWidget(self._openGL)

        # Setup strech factors
        self._layout.setColumnStretch(0, 2)
        self._layout.setColumnStretch(1, 2)
        self._layout.setColumnStretch(2, 2)
        self._layout.setColumnStretch(3, 1)

        self._layout.setMenuBar(self._menu_bar)
        self._layout.addLayout(opengl_layout, 0, 0)
        self._layout.addWidget(self._image_plotter, 0, 1)
        self._layout.addWidget(self._target_plotter, 0, 2)
        self._layout.addWidget(self._loss_plotter, 1, 0, 1, 3)
        self._layout.addWidget(self._settings_panel, 0, 3)

        self.setLayout(self._layout)

    def _set_image_to_match(self, image: Image):
        # We want to display the same image that we are testing the loss against. This image is columns major (input,y)
        # which is not what matplotlib want's so we have to transpose it back to row major
        self._target_image = image.convert("RGB")
        self._target_matrix = image_funcs.image_to_tensor(self._target_image)
        self._target_plotter.set_image(self._target_matrix)

    def _set_gl_program(self):
        self._openGL.set_program(self._program)

    def _run_gradient_descent_torch(self):
        self._reset()

        self.gd = GradientDescent(self._target_image, self._out_node, self._settings_panel.settings)
        self.thread = QThread()
        self.gd.iteration_done.connect(self._gd_iter_callback)
        self.gd.first_render_done.connect(self._set_parameter_values)
        self.gd.moveToThread(self.thread)
        self.thread.started.connect(self.gd.run)
        self.gd.finished.connect(self._finish_gradient_descent)

        _logger.debug("Started Gradient Descent Thread...")
        self.thread.start()

    def _reset(self):
        # Reset procedural model parameters
        for key in self._params:
            self._params[key].restore_value()
            self._set_parameter_values(self._params)

        # Reset plots
        self._loss_plotter.plotItem.clear()
        self._image_plotter.clear()

    def _stop_gradient_descent(self):
        if self.thread.isRunning():
            _logger.info("Stopping Gradient Descent Thread...")
            self._settings_panel.set_gd_finishing()
            self.gd.stop()
            self.thread.quit()
            self._settings_panel.set_gd_finished()

    def _finish_gradient_descent(self, params, loss_hist):
        self._params = params
        self._stop_gradient_descent()
        _logger.info("Gradient Descent finished with a final loss of {:.4f}.".format(loss_hist[-1]))

    def _gd_iter_callback(self, props):
        loss_hist = props['loss_hist']
        iter = props['iter']
        params = props['params']
        render = props['render']

        self._image_plotter.set_image(render)

        x = np.linspace(0, iter, num=iter + 1, endpoint=True)
        self._loss_plotter.plot(x, loss_hist, symbol='o')

        self._set_parameter_values(params)
        _logger.info("{}. loss: {}, params: {}".format(props['iter'], props['loss'], params))

    def _set_parameter_values(self, params: dict):
        for uniform_key, value in params.items():
            try:
                if isinstance(value, Parameter):
                    value = value.get_value()

                self._program[uniform_key] = value
            except IndexError as e:
                _logger.error("Uniform {} does not exist in program!".format(uniform_key))
                raise e

    def _update_settings(self, key: str, settings: dict):
        self._settings = settings

    def _open_loss_viz_window(self):
        self._loss_visualizer.open(self._settings_panel.settings, self._target_image, self._out_node)
