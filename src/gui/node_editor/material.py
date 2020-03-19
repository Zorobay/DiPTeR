import logging
import typing
import uuid

from PyQt5.QtCore import QObject, pyqtSignal
from glumpy.gloo import Program, VertexBuffer

from src.gui.node_editor.edge import Edge
from src.gui.node_editor.node import Node
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.shaders.shader_super import Shader

_logger = logging.getLogger(__name__)


class Material(QObject):
    """Class that keeps track of all nodes present on the NodeScene, as well as their relation to each other."""

    program_ready = pyqtSignal(Program)
    edge_spawned = pyqtSignal(object)
    edge_despawned = pyqtSignal()

    def __init__(self, scene: NodeScene):
        super().__init__()

        # Define data variables to track nodes and edges
        self._nodes = {}
        self._edges = {}

        self._scene = scene
        self._program = None
        self._id = None  # TODO stop using this!
        self._is_drawing_edge = False

    def bind_vertices(self, V: VertexBuffer):
        if self._program is not None:
            self._program.bind(V)

    def new_node(self, shader: typing.Type[Shader]):
        if not isinstance(shader, Shader):
            shader = shader()  # Instantiate the shader if only a type was given

        node = Node(self._scene, shader.get_name(), shader)
        node.edge_started.connect(self._spawn_edge)
        self._nodes[node.id] = node
        self._id = node.id
        self._program = node.get_program()

        # Spawn node to scene
        self._scene.addItem(node)

        # Emit event to tell OpenGL that the program is ready
        self.program_ready.emit(self._program)

    def remove_edge(self, edge: Edge):
        self._scene.removeItem(edge)
        self._is_drawing_edge = False

    def get_program(self) -> Program:
        return self._program

    def _spawn_edge(self, node_id: uuid.UUID, socket: Socket):
        if not self._is_drawing_edge:
            self._is_drawing_edge = True
            edge = Edge()
            edge.start_pos = socket.pos()
            edge.end_pos = socket.pos()
            self._scene.addItem(edge)
            _logger.debug("Edge spawned at {}.".format(edge.start_pos))
            self.edge_spawned.emit(edge)
