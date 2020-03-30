import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QMainWindow

from src.gui.node_editor.control_center import ControlCenter
from src.gui.widgets.material_selector import MaterialSelector
from src.gui.node_editor.node_view import NodeView
from src.gui.rendering.opengl_settings_widget import OpenGLSettingsWidget
from src.gui.rendering.opengl_widget import OpenGLWidget


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.cc = ControlCenter()

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
