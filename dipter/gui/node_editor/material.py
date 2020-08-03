import logging
import typing
import uuid

from PyQt5.QtCore import QObject, pyqtSignal, Qt, QPointF
from glumpy.gloo import Program

from dipter.gui.node_editor.g_edge import GEdge
from dipter.gui.node_editor.g_node_socket import GNodeSocket
from dipter.gui.node_editor.g_shader_node import ShaderNode, GMaterialOutputNode, GShaderNode
from dipter.gui.node_editor.node_scene import NodeScene
from dipter.misc import material_serializer as ms
from dipter.shaders.shaders.material_output_shader import MaterialOutputShader
from dipter.shaders.shader_super import FunctionShader, CompilableShader

_logger = logging.getLogger(__name__)


class Material(QObject):
    """Class that keeps track of all nodes present on the NodeScene, as well as their relation to each other."""

    shader_ready = pyqtSignal(CompilableShader)
    program_ready = pyqtSignal(Program)
    changed = pyqtSignal()
    edge_spawned = pyqtSignal(object)
    edge_despawned = pyqtSignal()

    def __init__(self, cc, name: str, material_filepath: str = None):
        super().__init__()
        self.cc = cc
        self._name = name
        self._id = uuid.uuid4()

        self.node_scene = NodeScene()

        # Define data variables to track nodes and edges
        self._mat_output_node = None
        self._nodes = {}
        self._edges = {}
        self._taken_numbers = {}  # mapping shader class -> list(int)

        self._program = None
        self._shader = None
        self._is_drawing_edge = False

        if material_filepath:
            self._load_material(material_filepath)
        else:
            self._add_material_output_node()

    def __str__(self):
        return "Material ({})".format(self.name)

    @property
    def name(self):
        return self._name

    def id(self):
        return self._id

    def add_node(self, shader: typing.Type[FunctionShader] = None, backend_node: ShaderNode = None) -> GShaderNode:
        assert not (shader is None and backend_node is None), "At least one of shader or backend_node has to be specified!"
        label = ""
        if shader is not None:
            if not isinstance(shader, FunctionShader):
                shader = shader()  # Instantiate the shader if only a type was given
                label = shader.get_name()

        node = GShaderNode(self.node_scene, label=label, shader=shader, backend_node=backend_node)
        self._add_node(node)
        _logger.debug("Added new node {} to material {}.".format(node.label(), self.name))

        return node

    def _add_node(self, node: 'GShaderNode'):
        self._assign_node_number(node)
        node.edge_started.connect(self._spawn_edge)
        node.edge_ended.connect(self._end_edge)
        node.input_changed.connect(lambda _: self.changed.emit())
        self._nodes[node.id()] = node
        self.node_scene.addItem(node)

        self.changed.emit()

    def _add_material_output_node(self, backend_node: ShaderNode = None) -> GMaterialOutputNode:
        node = GMaterialOutputNode(self.node_scene, backend_node=backend_node)
        self._add_node(node)

        node.graph_changed.connect(self._handle_graph_change)
        self._shader = node.get_shader()
        self._mat_output_node = node
        self._program = node.get_program(copy=False)

        # Emit event to tell OpenGL that the Program and Shader is ready
        self.program_ready.emit(self.program)
        self.shader_ready.emit(self.shader)

        _logger.debug("Added default Material Output Node {} to material {}.".format(node.label(), self.name))
        return node

    def get_material_output_node(self) -> GMaterialOutputNode:
        return self._mat_output_node

    def _handle_graph_change(self):
        self._program = self._mat_output_node.get_program(copy=False)
        self.program_ready.emit(self.program)
        self.changed.emit()

    def _assign_node_number(self, node: 'Node'):
        """
        Assigns a number to a node that is unique among the nodes with the same shader type.
        """
        cls = node.get_shader().__class__
        if cls not in self._taken_numbers:
            self._taken_numbers[cls] = [None]

        numbers = self._taken_numbers[cls]

        for i, n in enumerate(numbers):  # Iterate numbers and find an empty slot
            if n is None:
                numbers[i] = i
                node.set_num(i)
                return

        # If no empty slot is found, add the next bigger number to the end of the list
        num = len(numbers)
        numbers.append(num)
        node.set_num(num)

    def _unassign_node_number(self, node: 'Node'):
        shader_cls = node.get_shader().__class__
        num = node.get_num()
        numbers = self._taken_numbers[shader_cls]

        if num == len(numbers) - 1:  # If this node had the last number in the list, shorten the list
            del numbers[num]
        else:
            numbers[num] = None

    def delete_node(self, node_id: uuid.UUID):
        """Delete a node with the matching id from this material as well as the scene.

        :returns: True if successful, False otherwise.
        """
        nodes = self.get_nodes()

        try:
            node = nodes[node_id]
            self._unassign_node_number(node)
            node.delete()
            del nodes[node_id]  # Remove nodes from list of material nodes

            self.node_scene.removeItem(node)

            _logger.info("Deleted node {} from material {}".format(node.label(), self.name))
            self.changed.emit()
            return True
        except KeyError:
            return False

    def get_nodes(self) -> typing.Dict[uuid.UUID, GShaderNode]:
        return self._nodes

    def remove_edge(self, edge: GEdge):
        self.node_scene.removeItem(edge)
        self._is_drawing_edge = False

    @property
    def program(self):
        return self._program

    @property
    def shader(self):
        return self._shader

    def _spawn_edge(self, node_id: uuid.UUID, edge: GEdge):
        self.node_scene.addItem(edge)
        self.edge_spawned.emit(edge)

        # _logger.debug("Added new Edge ({}) at pos ({}) -> ({}) to material {}.".format(edge.id, edge.out_pos, edge.in_pos, self.title))

    def _end_edge(self, node_id: uuid.UUID, edge: GEdge):
        items = edge.collidingItems(Qt.IntersectsItemShape)

        source_node = self._nodes[node_id]
        connected_socket = edge.get_source_socket() if edge.get_source_socket() else edge.get_destination_socket()
        connected_socket_type = connected_socket.type()

        for item in items:
            if isinstance(item, GNodeSocket) \
                    and item != connected_socket \
                    and item.type() != connected_socket_type \
                    and not source_node.has_socket(item):

                if item.type() == GNodeSocket.INPUT:
                    edge.set_destination_socket(item)
                else:
                    edge.set_source_socket(item)

                # Connect edge to socket where edge was released. Connection of socket to starting node is done in the Socket class.
                edge.connect_sockets()
                _logger.info("Connected source node {} and node {} with edge".format(source_node.label(), item.parent_node().label()))
                return

        # Edge did not intersect with another valid socket, so it will be removed
        _logger.debug("Discarded edge started from node {}".format(source_node.label()))
        self.node_scene.removeItem(edge)

    def _load_material(self, filename):
        """Loads a saved material from file."""
        mon, mat_dict = ms.load_material(filename)
        self._build_node_graph(mon, mat_dict)

    def _build_node_graph(self, node: ShaderNode, mat_dict: dict, parent: GShaderNode = None):
        if isinstance(node.get_shader(), MaterialOutputShader):
            parent_node = self._add_material_output_node(backend_node=node)
        else:
            parent_node = self.add_node(backend_node=node)

        # Set scene position for parent node
        mat_dict = mat_dict[parent_node.id().hex]
        try:
            pos = mat_dict[ms.POSITION]
            parent_node.setPos(QPointF(*pos))
        except KeyError:
            if parent:
                shift = QPointF(270, 0)
                parent_pos = parent.scenePos()
                parent_node.setPos(parent_pos - shift)

        for inp in node.get_input_sockets():
            if inp.is_connected():
                for next_node in inp.get_connected_nodes():
                    list_inputs = mat_dict[ms.INPUTS]
                    for d_inp in list_inputs:
                        if inp.label() == d_inp[ms.ARGUMENT]:
                            sub_dict = d_inp[ms.VALUE]
                            self._build_node_graph(next_node, sub_dict, parent_node)

                for out_socket in inp.get_connected_sockets():
                    edge = inp.find_connecting_edge(out_socket)
                    gedge = GEdge.from_edge(edge)
                    gedge.connect_sockets()
                    self.node_scene.addItem(gedge)
