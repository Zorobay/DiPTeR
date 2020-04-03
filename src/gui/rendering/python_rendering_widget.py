import logging

import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.material import Material
from src.gui.widgets.labelled_input import LabelledInput
from src.gui.widgets.line_input import IntInput
from src.misc.render_funcs import render_torch, render_torch_with_callback
from src.shaders.color_shader import ColorShader
from src.shaders.shader_super import Shader

_logger = logging.getLogger("PythonRenderingWidget")


class PythonRenderingWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, cc: ControlCenter, *args):
        super().__init__(*args)

        self.cc = cc

        # Define gui components
        self._figure = Figure(figsize=(4, 4))
        self._figure_canvas = FigureCanvas(self._figure)
        self._axis = self._figure.add_subplot(111)
        self._layout = QVBoxLayout()
        self._width_input = IntInput(1, 500)
        self._height_input = IntInput(1, 500)
        self._render_progress = QLabel("0/? rows")

        # Define widget data
        self._width, self._height = 100, 100
        self._shader = None
        self._material = None
        self._image_artist = None

        self._init_widget()

    def _init_widget(self):
        self.setWindowTitle("Python Renderer")

        # Setup settings controls
        settings_layout = QHBoxLayout()
        settings_layout.setAlignment(Qt.AlignLeft)
        self._width_input.set_default_value(self._width)
        self._height_input.set_default_value(self._height)
        settings_layout.addWidget(LabelledInput("Width", self._width_input))
        settings_layout.addWidget(LabelledInput("Height", self._height_input))
        settings_layout.addWidget(LabelledInput("Progress: ", self._render_progress))
        self._layout.addLayout(settings_layout)
        self._width_input.input_changed.connect(self._handle_size_change)
        self._height_input.input_changed.connect(self._handle_size_change)

        # Setup matplotlib
        self._layout.addWidget(self._figure_canvas)
        self._image_artist = self._axis.imshow(np.zeros((self._width, self._height, 4)))

        self.cc.active_material_changed.connect(self._material_changed)
        if self.cc.active_material:
            self._material_changed(self.cc.active_material)

        self.setLayout(self._layout)

    def _render(self):
        shader = self._material.shader
        params = shader.get_parameters_list_torch(False)
        _logger.debug("Rendering {}...".format(shader.__class__.__name__))
        img = render_torch_with_callback(self._width, self._height, self._render_cb, shader.shade_torch, *params)
        _logger.debug("Rendering DONE.")

        self._image_artist.set_data(img)
        self._axis.set_ylim([0, self._height])
        self._axis.set_xlim([0, self._width])
        self._figure_canvas.draw()
        self._figure_canvas.flush_events()

    def _material_changed(self, mat: Material):
        # Disconnect signals from previous material
        if self._material:
            self._material.shader_ready.disconnect(self._set_shader)
            self._material.changed.disconnect(self._handle_material_changed)

        self._material = mat
        self._material.shader_ready.connect(self._set_title)
        self._material.changed.connect(self._handle_material_changed)

        if self._material.shader:  # Handle the case where the shader is already available
            self._set_title()
            self._render()

    def _render_cb(self, row: int):
        self._render_progress.setText("{}/{} rows".format(row+1, self._height))

    def _set_title(self):
        shader = self._material.shader
        self._axis.set_title("Python Render ({})".format(shader.__class__.__name__))

    def _handle_material_changed(self):
        self._render()

    def _handle_size_change(self):
        self._width = self._width_input.get_gl_value()
        self._height = self._height_input.get_gl_value()
        self._render()

    def closeEvent(self, event: QCloseEvent):
        self.closed.emit()
        super().closeEvent(event)

