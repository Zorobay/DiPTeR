import logging
import typing
import uuid

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QComboBox, QWidget, QPushButton, QHBoxLayout, QInputDialog, QMessageBox
from glumpy.gloo import Program, VertexBuffer

from src.gui.node_editor.edge import Edge
from src.gui.node_editor.node import Node
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.shaders.shader_super import Shader

_logger = logging.getLogger(__name__)


class Material(QObject):
    """Class that keeps track of all nodes present on the NodeScene, as well as their relation to each other."""

    shader_ready = pyqtSignal(Shader)
    program_ready = pyqtSignal(Program)
    material_changed = pyqtSignal(uuid.UUID)
    edge_spawned = pyqtSignal(object)
    edge_despawned = pyqtSignal()

    def __init__(self, name: str, scene: NodeScene):
        super().__init__()
        self._name = name
        self._scene = scene

        # Define data variables to track nodes and edges
        self._nodes = {}
        self._edges = {}

        self._program = None
        self._shader = None
        self._id = None  # TODO stop using this!
        self._is_drawing_edge = False

    @property
    def name(self):
        return self._name

    def add_node(self, shader: typing.Type[Shader]):
        if not isinstance(shader, Shader):
            shader = shader()  # Instantiate the shader if only a type was given

        self._shader = shader
        node = Node(self._scene, self._shader.get_name(), self._shader)
        node.edge_started.connect(self._spawn_edge)
        node.input_changed.connect(lambda id_: self.material_changed.emit(id_))
        self._nodes[node.id] = node
        self._id = node.id
        self._program = node.get_program()

        # Spawn node to scene
        self._scene.addItem(node)

        # Emit event to tell OpenGL that the Program and Shader is ready
        self.program_ready.emit(self.program)
        self.shader_ready.emit(self.shader)

    def remove_edge(self, edge: Edge):
        self._scene.removeItem(edge)
        self._is_drawing_edge = False

    @property
    def program(self):
        return self._program

    @property
    def shader(self):
        return self._shader

    def _spawn_edge(self, node_id: uuid.UUID, socket: Socket):
        if not self._is_drawing_edge:
            self._is_drawing_edge = True
            edge = Edge()
            edge.start_pos = socket.pos()
            edge.end_pos = socket.pos()
            self._scene.addItem(edge)
            _logger.debug("Edge spawned at {}.".format(edge.start_pos))
            self.edge_spawned.emit(edge)


class MaterialSelector(QWidget):
    material_ready = pyqtSignal(Material)

    def __init__(self, scene: NodeScene, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._scene = scene

        # Define components
        self._add_button = QPushButton("+")
        self._material_combo_box = QComboBox()
        self._layout = QHBoxLayout()

        # Define data
        self._materials = []
        self._active_material = None

        self._init_widget()

    def _init_widget(self):
        self._layout.setAlignment(Qt.AlignLeft)

        self._add_button.clicked.connect(self._add_button_clicked)
        self._add_button.setFixedWidth(20)
        self._layout.addWidget(self._add_button)

        self._material_combo_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._material_combo_box.currentIndexChanged.connect(self._selected_material_changed)
        self._layout.addWidget(self._material_combo_box)

        self.setLayout(self._layout)

    @property
    def active_material(self) -> Material:
        return self._active_material

    def _add_button_clicked(self):
        name, ok = QInputDialog.getText(self, "New Material Name", "Material Name", text="Material{}".format(len(self._materials)))

        if name and ok:
            material = Material(name, self._scene)
            self._materials.append(material)
            self._material_combo_box.addItem(material.name)
            self.material_ready.emit(material)

    def _selected_material_changed(self, index):
        self._active_material = self._materials[index]
        self.material_ready.emit(self.active_material)

    def add_node(self, shader: typing.Type[Shader]):
        if self.active_material:
            self.active_material.add_node(shader)
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("No Material Selected")
            msg.setText("Please select or create a new material before adding nodes.")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()


