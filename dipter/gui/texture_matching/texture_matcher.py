import ast
import inspect
import logging
import numbers
import pydoc
import typing
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PIL import Image
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QFileDialog, QDockWidget, QVBoxLayout, QComboBox, QMenuBar, QGroupBox
from torch import optim

from dipter.gui.node_editor.control_center import ControlCenter
from dipter.gui.node_editor.g_shader_node import GMaterialOutputNode
from dipter.gui.rendering.image_plotter import ImagePlotter
from dipter.gui.rendering.opengl_widget import OpenGLWidget
from dipter.gui.texture_matching.loss_visualizer import LossVisualizer
from dipter.gui.widgets.node_input.labelled_input import LabelledInput
from dipter.gui.widgets.node_input.line_input import FloatInput, IntInput, StringInput, MultipleIntInput, MultipleFloatInput
from dipter.misc import image_funcs, runtime_funcs, qwidget_funcs, string_funcs, number_funcs
from dipter.node_graph.data_type import DataType
from dipter.node_graph.parameter import Parameter
from dipter.optimization import losses
from dipter.optimization.gradient_descent import GradientDescent, GradientDescentSettings, run_in_thread
from dipter.optimization.losses import Loss

sns.set()

_logger = logging.getLogger(__name__)

device = "cpu"

CHANNELS = 3


def to_filename(mat_name, loss_name, optimizer_name):
    return "{}_{}_{}".format(mat_name, loss_name, optimizer_name)


class FunctionSettingsGroup(QGroupBox):

    def __init__(self):
        super().__init__()
        self._func = None
        self._args = dict()
        self._layout = QVBoxLayout()

        self._init()

    def _init(self):
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    def set_function(self, func):
        self._func = func
        self._create_widgets()
        self._update_title()

    def _create_widgets(self):
        widget_dict = self._get_widget_dict()
        qwidget_funcs.clear_layout(self._layout)

        func = self._func
        args = list(runtime_funcs.get_function_arguments(func.__init__).items())[1:]  # Skip 'self' argument

        for arg_name, parameter in args:
            if arg_name in widget_dict:
                widget = widget_dict[arg_name]
            else:
                default = parameter.default
                subtype = None
                if parameter.annotation not in (None, inspect._empty):
                    type_ = parameter.annotation
                else:
                    if isinstance(default, typing.Iterable):
                        subtype = type(default[0])
                    if type(default) == int and default == 0:
                        type_ = float  # We do not know if this is supposed to be a float or an integer...
                    else:
                        type_ = type(default)

                widget = self._widget_from_type(type_, subtype)
                if widget and default is not None:
                    widget.set_value(default)
            if widget:
                self._layout.addWidget(LabelledInput(arg_name, widget), 0, Qt.AlignTop)

    def _widget_from_type(self, type_, subtype):
        if type_ in [float, DataType.Float]:
            return FloatInput()
        elif type_ in [int, DataType.Int]:
            return IntInput()
        elif type_ in [str]:
            return StringInput()
        elif type_ in [typing.Iterable[int]]:
            return MultipleIntInput()
        elif type_ in [typing.Iterable[float]]:
            return MultipleFloatInput()
        elif type_ in [tuple, list]:
            if subtype == float:
                return MultipleFloatInput()
            elif subtype == int:
                return MultipleIntInput()
        else:
            return None

    def _update_title(self):
        self.setTitle(self._func.__class__.__name__)

    def _get_widget_dict(self) -> dict:
        out =dict()
        for i in range(self._layout.count()):
            widget = self._layout.itemAt(i).widget()
            label = widget.label
            out[label] = widget.widget
        return out

    def to_dict(self) -> dict:
        out = dict()
        for i in range(self._layout.count()):
            labelled_input = self._layout.itemAt(i).widget()
            label = labelled_input.label
            value = labelled_input.widget.get_gl_value()
            out[label] = value

        return out

    def load_from_dict(self, d: dict):
        for key, value in d.items():
            for i in range(self._layout.count()):
                labelled_input = self._layout.itemAt(i).widget()
                label = labelled_input.label
                if label == key:
                    labelled_input.widget.set_value(value)
                    break


class SettingsPanel(QWidget):
    match_start = pyqtSignal()
    match_stop = pyqtSignal()
    reset_requested = pyqtSignal()
    texture_loaded = pyqtSignal(Image.Image)
    settings_changed = pyqtSignal(GradientDescentSettings)  # changed key as well as all settings
    save_data = pyqtSignal(str)

    def __init__(self, cc: ControlCenter, *args):
        super().__init__(*args)
        # Define gui elements
        self.cc = cc
        self._layout = QVBoxLayout()
        self._load_texture_button = QPushButton("Load Texture")
        self._texture_label = QLabel("load texture...")
        self._match_button = QPushButton("Match Texture")
        self._loss_combo_box = QComboBox()
        self._loss_settings_group = FunctionSettingsGroup()
        self._optimizer_combo_box = QComboBox()
        self._optimizer_settings_group = FunctionSettingsGroup()
        self._width_input = IntInput(0, 1000)
        self._height_input = IntInput(0, 1000)
        self._max_iter_input = IntInput(0, 10000)
        self._early_stopping_loss_thresh = FloatInput(0, 1)
        self._save_data_button = QPushButton("Save Data")
        self._load_settings_button = QPushButton("Load Settings")
        # self._learning_rate = FloatInput(0, 1)

        # Define data
        self._loss_func_map = {}
        ls = [losses.XSELoss, losses.SquaredBinLoss, losses.NeuralLoss]
        for l in ls:
            self._loss_func_map[l.__name__] = l

        self._optimizer_map = {}
        os = [optim.Adam, optim.AdamW, optim.Adagrad, optim.RMSprop]
        for o in os:
            self._optimizer_map[o.__name__] = o

        self.loaded_image = None
        self._max_iter = 100
        self._settings = GradientDescentSettings()
        self._is_running = False
        self._is_cleared = True
        self._last_load_path = None

        self._init_widget()

    def _init_widget(self):
        self._match_button.setEnabled(False)
        self._match_button.clicked.connect(self._toggle_matching)
        self._load_texture_button.clicked.connect(self._load_texture)

        # --- Setup loss function combo box ---
        self._loss_combo_box.addItems(list(self._loss_func_map))
        self._loss_combo_box.currentIndexChanged.connect(self._set_loss_func)
        self._loss_combo_box.setCurrentIndex(0)
        self._settings.loss_func = self._loss_func_map[self._loss_combo_box.currentText()]

        # --- Setup loss function settings group ---
        self._loss_settings_group.set_function(self._settings.loss_func)

        # --- Setup optimizer combo box ---
        self._optimizer_combo_box.addItems(list(self._optimizer_map))
        self._optimizer_combo_box.currentIndexChanged.connect(self._set_optimizer)
        self._optimizer_combo_box.setCurrentIndex(0)
        self._settings.optimizer = self._optimizer_map[self._optimizer_combo_box.currentText()]

        # --- Setup optimizer function settings group ---
        self._optimizer_settings_group.set_function(self._settings.optimizer)

        # --- Setup render size input ---
        self._width_input.set_value(self._settings.render_width)
        self._height_input.set_value(self._settings.render_height)
        self._width_input.input_changed.connect(lambda: self._change_settings("render_width", self._width_input.get_gl_value()))
        self._height_input.input_changed.connect(lambda: self._change_settings("render_height", self._height_input.get_gl_value()))

        # --- Setup max iterations input ---
        self._max_iter_input.input_changed.connect(lambda: self._change_settings("max_iter", self._max_iter_input.get_gl_value()))
        self._max_iter_input.set_value(100)

        # --- Setup early stopping loss threshold input ---
        self._early_stopping_loss_thresh.input_changed.connect(lambda: self._change_settings("early_stopping_thresh",
                                                                                             self._early_stopping_loss_thresh.get_gl_value()))
        self._early_stopping_loss_thresh.set_value(0.01)

        # --- Setup save/load data button ---
        self._save_data_button.clicked.connect(self._save_data)
        self._load_settings_button.clicked.connect(self._load_settings)

        self._layout.addWidget(self._match_button)
        self._layout.addWidget(self._load_texture_button)
        self._layout.addWidget(LabelledInput("Loss function", self._loss_combo_box))
        self._layout.addWidget(self._loss_settings_group)
        self._layout.addWidget(LabelledInput("Optimizer", self._optimizer_combo_box))
        self._layout.addWidget(self._optimizer_settings_group)
        self._layout.addWidget(LabelledInput("Render width", self._width_input))
        self._layout.addWidget(LabelledInput("Render height", self._height_input))
        self._layout.addWidget(LabelledInput("Max iterations", self._max_iter_input))
        self._layout.addWidget(LabelledInput("Early stopping loss thresh", self._early_stopping_loss_thresh))
        self._layout.addWidget(self._save_data_button)
        self._layout.addWidget(self._load_settings_button)

        self._layout.setAlignment(Qt.AlignTop)
        self.setLayout(self._layout)

    def settings(self) -> GradientDescentSettings:
        # Instantiate loss function with settings from loss function widget
        loss_func = self._settings.loss_func
        if isinstance(loss_func, Loss):
            self._settings.loss_func = loss_func.__class__

        self._settings.loss_args = self._loss_settings_group.to_dict()
        self._settings.optimizer_args = self._optimizer_settings_group.to_dict()
        return self._settings

    def _load_texture(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Texture", filter="Image File (*.png *.jpg *.jpeg *.bmp)")

        if filename:
            self.loaded_image = Image.open(filename)
            self._match_button.setEnabled(True)
            self.texture_loaded.emit(self.loaded_image)

    def _set_loss_func(self, index: int):
        loss_func = self._loss_func_map[self._loss_combo_box.currentText()]

        if loss_func == losses.NeuralLoss or isinstance(loss_func, losses.NeuralLoss):
            self._width_input.set_value(224)
            self._height_input.set_value(224)

        self._change_settings("loss_func", loss_func)
        self._loss_settings_group.set_function(loss_func)

    def _set_optimizer(self, index: int):
        optimizer = self._optimizer_map[self._optimizer_combo_box.currentText()]
        self._change_settings("optimizer", optimizer)
        self._optimizer_settings_group.set_function(optimizer)

    def set_gd_finished(self):
        self._match_button.setText("Reset")
        self._is_running = False

    def set_gd_finishing(self):
        self._match_button.setText("Stopping...")

    def _change_settings(self, var: str, new_val: typing.Any):
        _logger.debug("New value for setting {} -> {}.".format(var, new_val))
        setattr(self._settings, var, new_val)
        self.settings_changed.emit(self._settings)

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

    def _save_data(self):
        filename = to_filename(self.cc.active_material.name, self._settings.loss_func.__name__, self._settings.optimizer.__name__) + ".hdf5"
        directory = Path.cwd() / "data" / filename
        path, _ = QFileDialog.getSaveFileName(self, "Save Data", directory=str(directory), filter="HDF5 (*.hdf5)")

        if path:
            self.save_data.emit(path)

    def _load_settings(self):
        if self._last_load_path:
            path, _ = QFileDialog.getOpenFileName(self, "Load Settings", directory=str(self._last_load_path), filter="HDF5 (*.hdf5)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Load Settings", filter="HDF5 (*.hdf5)")

        if path:
            self._last_load_path = Path(path).parent
            f = h5py.File(path, "r")
            attrs = dict(f.get("data").attrs)
            for att in attrs:
                saved_att = attrs[att]
                try:
                    current_att = getattr(self._settings, att)
                    type_att = type(current_att)
                    if not isinstance(saved_att, type_att):
                        if type_att == dict:
                            saved_att = ast.literal_eval(saved_att)
                        elif type_att == type:
                            import_string = string_funcs.type_to_import_string(saved_att)
                            if att == "optimizer":  # Optimizers are weird when it comes to importing them, handle them separately
                                cls_name = import_string.split(".")[-1]
                                import_string = "torch.optim." + cls_name

                            saved_att = pydoc.locate(import_string)
                        else:
                            saved_att = type_att(saved_att)

                    if isinstance(saved_att, numbers.Real):
                        saved_att = number_funcs.round_significant(saved_att, 3)
                    elif isinstance(saved_att, dict):
                        for key, val in saved_att.items():
                            if isinstance(val, numbers.Real):
                                saved_att[key] = number_funcs.round_significant(val, 3)
                    elif isinstance(saved_att, (list, np.ndarray)):
                        for i, val in enumerate(saved_att):
                            if isinstance(val, numbers.Real):
                                saved_att[i] = number_funcs.round_significant(val, 3)

                    setattr(self._settings, att, saved_att)
                except AttributeError as e:
                    _logger.debug("Attribute {} not found in settings. Skipping...".format(att))
            f.close()
            self._update_from_settings()
            _logger.info("Loaded settings from file {}".format(path))

    def _update_from_settings(self):
        self._width_input.set_value(self._settings.render_width)
        self._height_input.set_value(self._settings.render_height)
        self._early_stopping_loss_thresh.set_value(self._settings.early_stopping_thresh)
        self._max_iter_input.set_value(self._settings.max_iter)

        if self._settings.loss_func:
            self._loss_combo_box.setCurrentIndex(list(self._loss_func_map).index(self._settings.loss_func.__name__))
            self._loss_settings_group.load_from_dict(self._settings.loss_args)

        if self._settings.optimizer:
            self._optimizer_combo_box.setCurrentIndex(list(self._optimizer_map).index(self._settings.optimizer.__name__))
            self._optimizer_settings_group.load_from_dict(self._settings.optimizer_args)


class TextureMatcher(QWidget):

    def __init__(self, cc: ControlCenter, mat_output_node: GMaterialOutputNode):
        super().__init__(parent=None)
        self.cc = cc
        self._out_node = mat_output_node

        # Define components
        self._menu_bar = QMenuBar()
        self._settings_drawer = QDockWidget(self, Qt.Drawer)
        self._settings_panel = SettingsPanel(self.cc, self)
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
        self.thread = None
        self.gd = None
        self.ren_i = 0
        self._params = {}
        self._loss_hist = None

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
        self._settings_panel.save_data.connect(self._save_data)
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
        self._layout.addWidget(self._settings_panel, 0, 3, 2, 1)

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

        self.gd = GradientDescent(self._target_image, self._out_node, self._settings_panel.settings())
        self.thread = QThread()
        _logger.debug("Started Gradient Descent Thread...")
        run_in_thread(self.gd, self.thread, iteration_done_callback=self._gd_iter_callback, first_render_done_callback=self._set_parameter_values,
                      gd_finished_callback=self._finish_gradient_descent)

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
        self._loss_hist = loss_hist
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

    def _open_loss_viz_window(self):
        self._loss_visualizer.open(self._settings_panel._settings, self._target_image, self._out_node)

    def _save_data(self, filename: str):
        mat_name = self.cc.active_material.name
        settings = self._settings_panel.settings().to_dict()
        f = h5py.File(filename, "w")
        dset = f.create_dataset("data", data=self._loss_hist, dtype="f")

        for key, value in settings.items():
            try:
                dset.attrs[key] = value
            except TypeError:
                dset.attrs[key] = str(value)
        dset.attrs["material_name"] = mat_name
        f.close()
        _logger.info("Saved data to {}".format(filename))
