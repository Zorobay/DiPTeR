import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QMainWindow, QDockWidget

from src.gui.node_editor.material import Material, MaterialSelector
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.node_view import NodeView
from src.gui.rendering.opengl_settings_widget import OpenGLSettingsWidget
from src.gui.rendering.opengl_widget import OpenGLWidget
from src.gui.rendering.python_rendering_widget import PythonRenderingWidget


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Define GUI layouts
        self.base_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        # Define GUI elements
        self.node_scene = NodeScene()
        self.material_selector = MaterialSelector(self.node_scene)
        self.node_view = NodeView(self.material_selector, self.node_scene)

        #self.render_window = QWidget(width=700, height=400)
        self.openGL = OpenGLWidget(400, 400, self.material_selector)
        self.opengl_settings = OpenGLSettingsWidget(self.material_selector, self.openGL)
        #self.python_renderer = PythonRenderingWidget(self.material)

        self._init_widget()

    def _init_widget(self):
        self.grid_layout.addWidget(self.material_selector, 0,0,1,1)
        self.grid_layout.addWidget(self.node_view, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.openGL, 0,1,2,1)
        self.grid_layout.addWidget(self.opengl_settings, 0,2,2,1)

        #self.render_window.setLayout(QGridLayout())
        #self.render_window.layout()

        #self.render_window.layout().addWidget(self.openGL, 0, 1, 2, 1)
        #self.render_window.layout().addWidget(self.opengl_settings, 0, 2, 2, 1)
        #self.render_window.layout().addWidget(self.python_renderer, 0, 1, 2, 1)

        #self.grid_layout.addWidget(self.render_window, 0, 1)

        self.setLayout(self.grid_layout)


class MainWindow(QMainWindow):

    def __init__(self, *args):
        super().__init__(*args)

        self._title = "Shader viewer"
        self._size = (300, 300, 1600, 500)

        self.main_widget = MainWidget()
        self._init_window()

    def _init_window(self):
        self.setCentralWidget(self.main_widget)

        self.setWindowTitle(self._title)
        self.setGeometry(*self._size)


# Setup global logging settings
FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
