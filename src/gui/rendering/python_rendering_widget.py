import uuid

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.node_editor.material import Material
from src.misc.render_funcs import render
from src.shaders.color_shader import ColorShader
from src.shaders.shader_super import Shader


class PythonRenderingWidget(QWidget):

    def __init__(self, material: Material, *args):
        super().__init__(*args)

        self._material = material
        self._width, self._height = 100, 100
        self._shader = None
        self._figure = Figure(figsize=(4, 4))
        self._axis = self._figure.add_subplot(111)
        self._image_artist = None
        self._figure_canvas = FigureCanvas(self._figure)
        self._layout = QVBoxLayout()

        self._init_widget()

    def _init_widget(self):
        self._setup_default_shader()
        self._material.shader_ready.connect(self._shader_changed)
        self._material.material_changed.connect(lambda id_: self._render())

        # Setup matplotlib
        self._layout.addWidget(self._figure_canvas)

        # Render default material
        params = self._shader.get_parameters_list()
        img = render(self._width, self._height, self._shader.shade, *params)
        self._image_artist = self._axis.imshow(img, vmin=0., vmax=1.)

        self.setLayout(self._layout)

    def _render(self):
        params = self._shader.get_parameters_list()
        img = render(self._width, self._height, self._shader.shade, *params)
        self._axis.clear()
        self._axis.imshow(img)
        self._figure_canvas.draw()
        self._figure.gca().invert_yaxis()
        self._axis.set_ylim(0, self._width)
        self._axis.set_xlim(0, self._height)

    def _setup_default_shader(self):
        self._shader = ColorShader()

    def _shader_changed(self, shader: Shader):
        self._shader = shader
