from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QGraphicsView

from src.gui.widgets.OpenGLSettingsWidget import OpenGLSettingsWidget
from src.gui.widgets.OpenQLWidget import OpenGLWidget
from src.gui.widgets.Node import Node
from src.gui.widgets.NodeScene import NodeScene


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._TITLE = "Shader viewer"
        self._SIZE = (300, 300, 1200, 500)
        self.sliders = []

        # Define GUI layouts
        self.grid_layout = None
        self.base_layout = None

        # Define GUI elements
        self.image_viewer = None
        self.node_scene = None
        self.opengl_settings = None

        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self._TITLE)
        self.setGeometry(*self._SIZE)

        # Initialize layout
        self.base_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        self.base_layout.addLayout(self.grid_layout)
        self.setLayout(self.base_layout)

        # Setup Node Area
        self.node_scene = NodeScene()
        self.node_view = QGraphicsView(self.node_scene)
        self.node_scene.addItem(Node(self.node_scene, "A title"))
        self.node_scene.addItem(Node(self.node_scene, "Node 2"))
        self.grid_layout.addWidget(self.node_view, 0, 0, 2, 1)

        # Setup OpenGL settings widget
        self.opengl_settings = OpenGLSettingsWidget()
        self.grid_layout.addWidget(self.opengl_settings, 0, 1, 1, 1)

        # Setup open GL rendering widget
        self.openGL = OpenGLWidget(500, 400)
        self.grid_layout.addWidget(self.openGL, 1, 1, 1, 1)



        self.show()


app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
