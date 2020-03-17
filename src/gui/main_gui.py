import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.node_view import NodeView
from src.gui.opengl.opengl_widget import OpenGLWidget
from src.gui.node_editor.material import Material
from src.gui.opengl.opengl_settings_widget import OpenGLSettingsWidget
from src.gui.rendering_area import RenderingArea


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._TITLE = "Shader viewer"
        self._SIZE = (300, 300, 1200, 500)
        self.sliders = []

        # Define GUI layouts
        self.base_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        # Define GUI elements
        self.node_scene = NodeScene()
        self.material = Material(self.node_scene)
        self.node_view = NodeView(self.material, self.node_scene)
        self.openGL = OpenGLWidget(500, 400, self.material)
        self.opengl_settings = OpenGLSettingsWidget(self.openGL)
        self.python_renderer = RenderingArea()

        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self._TITLE)
        self.setGeometry(*self._SIZE)

        # Setup layout
        self.base_layout.addLayout(self.grid_layout)
        self.setLayout(self.base_layout)

        # Setup Node Area
        self.grid_layout.addWidget(self.node_view, 0, 0, 2, 1)

        # Setup open GL rendering widget
        self.grid_layout.addWidget(self.openGL, 1, 1, 1, 1)

        # Setup OpenGL settings widget
        self.grid_layout.addWidget(self.opengl_settings, 0, 1, 1, 1)

        # Setup Rendering Area where python shaders are rendered
        self.grid_layout.addWidget(self.python_renderer, 1,2,1,1)
        self.show()


# Setup global logging settings
FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
