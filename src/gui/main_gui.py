import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QMainWindow, QAction

from src.gui.node_editor.control_center import Material, ControlCenter
from src.gui.rendering.python_rendering_widget import PythonRenderingWidget
from src.gui.widgets.material_selector import MaterialSelector
from src.gui.node_editor.node_view import NodeView
from src.gui.rendering.opengl_settings_widget import OpenGLSettingsWidget
from src.gui.rendering.opengl_widget import OpenGLWidget

_logger = logging.getLogger(__name__)


class MainWidget(QWidget):
    def __init__(self, cc:Material):
        super().__init__()

        self.cc = cc
        # Define GUI layouts
        self.base_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        # Define GUI elements
        self.material_selector = MaterialSelector(self.cc)
        self.node_view = NodeView(self.cc)

        self.openGL = OpenGLWidget(400, 400, self.cc)
        self.opengl_settings = OpenGLSettingsWidget(self.cc, self.openGL)

        self._init_widget()

    def _init_widget(self):
        self.grid_layout.addWidget(self.material_selector, 0, 0, 1, 1)
        self.grid_layout.addWidget(self.node_view, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.openGL, 0, 1, 2, 1)
        self.grid_layout.addWidget(self.opengl_settings, 0, 2, 2, 1)

        # Add starting material to Scene
        self.material_selector.add_material("Material")

        self.setLayout(self.grid_layout)


class MainWindow(QMainWindow):

    def __init__(self, *args):
        super().__init__(*args)

        self.cc = ControlCenter()

        self._title = "Shader viewer"
        self._size = (300, 300, 1600, 500)

        self.main_widget = MainWidget(self.cc)
        self.py_renderer = None
        self.menu_bar = self.menuBar()
        self.window_menu = self.menu_bar.addMenu("Windows")
        self.open_py_renderer_action = QAction("Python renderer")
        self._init_window()

    def _init_window(self):
        self.setCentralWidget(self.main_widget)

        self.window_menu.addAction(self.open_py_renderer_action)
        self.open_py_renderer_action.triggered.connect(self.open_py_renderer)

        self.setWindowTitle(self._title)
        self.setGeometry(*self._size)

    def open_py_renderer(self):
        _logger.debug("Launching Python Rendering Widget...")
        self.py_renderer = PythonRenderingWidget(self.cc)
        self.py_renderer.closed.connect(self._remove_py_renderer)
        self.py_renderer.show()

    def _remove_py_renderer(self):
        self.py_renderer.deleteLater()
        self.py_renderer = None


# Setup global logging settings
FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
