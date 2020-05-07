import logging
import typing
import uuid

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from glumpy.gloo import Program

from src.gui.node_editor.edge import Edge
from src.gui.node_editor.node import ShaderNode
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.shaders.shader_super import Shader

_logger = logging.getLogger(__name__)


class Material(QObject):
    """Class that keeps track of all nodes present on the NodeScene, as well as their relation to each other."""

    shader_ready = pyqtSignal(Shader)
    program_ready = pyqtSignal(Program)
    changed = pyqtSignal()
    edge_spawned = pyqtSignal(object)
    edge_despawned = pyqtSignal()

    def __init__(self, cc, name: str):
        super().__init__()
        self.cc = cc
        self._name = name
        self._id = uuid.uuid4()

        self.node_scene = NodeScene()

        # Define data variables to track nodes and edges
        self._nodes = {}
        self._edges = {}

        self._program = None
        self._shader = None
        self._is_drawing_edge = False

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    def add_node(self, shader: typing.Type[Shader]):
        if not isinstance(shader, Shader):
            shader = shader()  # Instantiate the shader if only a type was given

        self._shader = shader
        node = ShaderNode(self.node_scene, self._shader.get_name(), self._shader)
        node.edge_started.connect(self._spawn_edge)
        node.edge_ended.connect(self._end_edge)
        node.input_changed.connect(lambda id_: self.changed.emit())
        self._nodes[node.id] = node
        self._id = node.id
        self._program = node.get_program()

        # Spawn node to scene
        self.node_scene.addItem(node)

        # Emit event to tell OpenGL that the Program and Shader is ready
        self.program_ready.emit(self.program)
        self.shader_ready.emit(self.shader)
        self.changed.emit()

        _logger.debug("Added new node {} to material {}.".format(node.title, self.name))

    def get_nodes(self) -> typing.Dict[uuid.UUID, ShaderNode]:
        return self._nodes

    def remove_edge(self, edge: Edge):
        self.node_scene.removeItem(edge)
        self._is_drawing_edge = False

    @property
    def program(self):
        return self._program

    @property
    def shader(self):
        return self._shader

    def _spawn_edge(self, node_id: uuid.UUID, edge: Edge):
        self.node_scene.addItem(edge)
        self.edge_spawned.emit(edge)

        _logger.debug("Added new Edge ({}) at pos ({}) -> ({}) to material {}.".format(edge.id, edge.out_pos, edge.in_pos, self.name))

    def _end_edge(self, node_id: uuid.UUID, edge: Edge):
        items = edge.collidingItems(Qt.IntersectsItemShape)

        node = self._nodes[node_id]
        connected_socket_id = edge.out_socket_id if edge.out_socket_id else edge.in_socket_id
        connected_socket_type = node.get_socket_type(connected_socket_id)

        for item in items:
            if isinstance(item, Socket) \
                    and item.id != connected_socket_id \
                    and item.socket_type != connected_socket_type \
                    and not node.has_socket(item.id):

                if item.socket_type == Socket.SOCKET_INPUT:
                    edge.in_socket_id = item.id
                else:
                    edge.out_socket_id = item.id

                # Connect edge to socket where edge was released. Connection of socket to starting node is done in the Socket class.
                item.add_connected_edge(edge)
                return

        # Edge did not intersect with another valid socket, so it will be removed
        self.node_scene.removeItem(edge)
