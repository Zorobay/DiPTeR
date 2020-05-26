import logging
import time

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QCloseEvent, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.material import Material
from src.gui.widgets.labelled_input import LabelledInput
from src.gui.widgets.line_input import IntInput

_logger = logging.getLogger("PythonRenderingWidget")


class PythonRenderingWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, cc: ControlCenter, *args):
        super().__init__(*args)

        self.cc = cc

        # Define gui components
        self._plotWidget = pg.PlotWidget()
        self._imgItem = pg.ImageItem()
        self._cant_render_text = pg.TextItem(text="No shader connected to output node.", color="r", anchor=(0.5, 0.5))

        self._figure = Figure(figsize=(4, 4))
        self._figure_canvas = FigureCanvas(self._figure)
        self._axis = self._figure.add_subplot(111)
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

        # Setup pyqtgraph
        font = QFont()
        font.setPointSize(20)
        self._cant_render_text.setFont(font)
        self._plotWidget.addItem(self._imgItem)
        self._plotWidget.setYRange(0, self._height)
        self._plotWidget.setXRange(0, self._width)
        self._imgItem.setLevels(0, 1.0)
        self._layout.addWidget(self._plotWidget)

        self.cc.active_material_changed.connect(self._material_changed)
        if self.cc.active_material:
            self._material_changed(self.cc.active_material)

        self.setLayout(self._layout)

    def _render(self):
        node = self._material.get_material_output_node()

        start = time.time()
        img, _, _, _ = node.render(self._width, self._height, call_dict=None)

        total_time = time.time() - start
        _logger.debug("Rendering DONE in {:.4f}s.".format(total_time))

        if img is not None:
            self._plotWidget.removeItem(self._cant_render_text)
            self._imgItem.setImage(img.numpy(), autoLevels=False, levels=(0, 1.))
        else:
            self._plotWidget.removeItem(self._cant_render_text)
            self._plotWidget.addItem(self._cant_render_text)
            self._cant_render_text.setPos(self._width / 2, self._height / 2)

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
        self._plotWidget.setYRange(0, self._height)
        self._plotWidget.setXRange(0, self._width)
        self._render()

    def closeEvent(self, event: QCloseEvent):
        self.closed.emit()
        super().closeEvent(event)
