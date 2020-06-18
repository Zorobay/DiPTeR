import logging
import time

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.material import Material
from src.gui.rendering.image_plotter import ImagePlotter
from src.gui.widgets.labelled_input import LabelledInput
from src.gui.widgets.line_input import IntInput

_logger = logging.getLogger("PythonRenderingWidget")


class PythonRenderingWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, cc: ControlCenter, *args):
        super().__init__(*args)

        self.cc = cc

        # Define gui components
        self._image_plot = ImagePlotter()
        self._layout = QVBoxLayout()
        self._width_input = IntInput(1, 500)
        self._height_input = IntInput(1, 500)
        self._resize_button = QPushButton("Resize")

        # Define widget data
        self._width, self._height = 100, 100
        self._shader = None
        self._material = None

        self._init_widget()

    def _init_widget(self):
        self.setWindowTitle("Python Renderer")

        # Setup settings controls
        settings_layout = QHBoxLayout()
        settings_layout.setAlignment(Qt.AlignLeft)
        self._width_input.set_default_value(self._width)
        self._height_input.set_default_value(self._height)
        self._resize_button.clicked.connect(self._handle_resize)
        settings_layout.addWidget(LabelledInput("Width", self._width_input))
        settings_layout.addWidget(LabelledInput("Height", self._height_input))
        settings_layout.addWidget(self._resize_button)
        self._layout.addLayout(settings_layout)

        # Add plotting widget
        self._layout.addWidget(self._image_plot)

        self.cc.active_material_changed.connect(self._material_changed)
        if self.cc.active_material:
            self._material_changed(self.cc.active_material)

        self.setLayout(self._layout)

    def _render(self):
        node = self._material.get_material_output_node()
        start = time.time()
        img, _ = node.render(self._width, self._height)
        total_time = time.time() - start
        _logger.debug("Rendering DONE in {:.4f}s.".format(total_time))

        self._image_plot.set_image(img)

    def _material_changed(self, mat: Material):
        # Disconnect signals from previous material
        if self._material:
            self._material.shader_ready.disconnect(self._render)
            self._material.changed.disconnect(self._handle_material_changed)

        self._material = mat
        self._material.shader_ready.connect(self._render)
        self._material.changed.connect(self._handle_material_changed)

        if self._material.shader:  # Handle the case where the shader is already available
            self._render()

    def _set_title(self):
        shader = self._material.shader
        self._axis.set_title("Python Render ({})".format(shader.__class__.__name__))

    def _handle_material_changed(self):
        self._render()

    def _handle_resize(self):
        self._width = self._width_input.get_gl_value()
        self._height = self._height_input.get_gl_value()
        self._image_plot.set_x_range(0, self._width)
        self._image_plot.set_y_range(0, self._height)
        self._render()

    def closeEvent(self, event: QCloseEvent):
        self.closed.emit()
        super().closeEvent(event)
