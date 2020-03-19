from PyQt5.QtWidgets import QWidget, QGridLayout, QMainWindow
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.node_view import NodeView
from src.gui.node_editor.material import Material


class NodeEditor(QWidget):

    def __init__(self, *args):
        super().__init__(*args)

        # Define GUI layouts
        self.grid_layout = QGridLayout()

        # Define GUI elements
        self.node_scene = NodeScene()
        self.material = Material(self.node_scene)
        self.node_view = NodeView(self.material, self.node_scene)

        self._init_widget()

    def _init_widget(self):
        self.setLayout(self.grid_layout)

        self.grid_layout.addWidget(self.node_view, 0, 0, 2, 1)
