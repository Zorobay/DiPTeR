import abc
import typing
import uuid

import torch
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QFont, QColor, QPalette, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsWidget
from boltons.setutils import IndexedSet
from glumpy.gloo import Program
from node_graph.edge import Edge
from node_graph.node import ShaderNode
from node_graph.node_socket import NodeSocket
from src.gui.node_editor.g_edge import GEdge
from src.gui.node_editor.g_node_socket import GNodeSocket
from src.gui.node_editor.layouts import GraphicsGridLayout
from src.gui.node_editor.node_scene import NodeScene
from src.gui.widgets.array_input import ArrayInput
from src.gui.widgets.color_input import ColorInput
from src.gui.widgets.io_module import SocketModule, OutputModule
from src.gui.widgets.line_input import FloatInput, IntInput
from src.gui.widgets.shader_input import ShaderInput
from src.shaders.material_output_shader import MaterialOutputShader, DataType
from src.shaders.shader_super import Shader

TYPE_VALUE = "type_value"
TYPE_FUNC = "type_func"
TYPE = "type"
VALUE = "value"
ARGS = "args"
MODIFIED_ARG = "mod_arg"
NAME_ARG = "arg"
NAME_MODIFIED_ARG = "mod_arg"


def set_program_uniforms_from_node(program: Program, node: 'GShaderNode'):
    inputs = node.get_input(exclude_connected=True)

    for arg, mod_arg, value in inputs:
        program[mod_arg] = value


class GShaderNode(QGraphicsWidget):
    """This abstract class defines the look and feel of a Node. Specialized classes can subclass this instead of the Node class.
    """
    edge_started = pyqtSignal(uuid.UUID, GEdge)
    edge_ended = pyqtSignal(uuid.UUID, GEdge)
    connection_changed = pyqtSignal(GNodeSocket, object)  # Node
    input_changed = pyqtSignal(object)  # Node

    @abc.abstractmethod
    def __init__(self, node_scene: NodeScene, shader: Shader = None, label: str = "", parent=None, backend_node: ShaderNode = None):
        super(QGraphicsWidget, self).__init__(parent)

        self._node = backend_node
        if self._node is None:
            if shader is None:
                raise ValueError("shader or backend_node needs to be provided!")
            else:
                self._node = ShaderNode(shader=shader, label=label, set_default_inputs=True, container=self)
        else:
            self._node.set_container(self)
        if label:
            self._node.set_label(label)

        self.node_scene = node_scene
        self._num = -1

        self._in_socket_modules = []
        self._out_socket_modules = []
        self._socket_connections = {}

        # define Node properties
        self._selected = False
        self._deletable = True
        self._input_index = 1
        self._output_index = 1
        self._width = 250
        self._height = 50
        self._rounding = 5
        self._padding = 8
        self._bg_color = QColor(80, 80, 100, 200)
        self._title_color = Qt.white
        self._title_font = QFont("Corbel", 11)
        self._title_font.setBold(True)
        self._title_item = QGraphicsTextItem(self)

        # Define layout
        self._master_layout = GraphicsGridLayout()

        # Define widgets properties
        self._input_label_font = QFont("Corbel", 8)
        self._input_label_palette = QPalette()
        self._input_label_palette.setColor(QPalette.Background, QColor(0, 0, 0, 0))
        self._input_label_palette.setColor(QPalette.Foreground, self._title_color)

        # Set flags to enable the widget to be moved and selected
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

        self._init_title()
        self._init_layout()
        self._init_sockets()

    def _init_layout(self):
        self._master_layout.setContentsMargins(-5, 4, 4, -15)
        self._master_layout.setRowSpacing(0, self._title_font.pointSize() + 12)  # Add empty space for first row so that title is visible
        self._master_layout.setHorizontalSpacing(2.0)
        self._master_layout.setVerticalSpacing(0.0)
        self._master_layout.setColumnAlignment(2, Qt.AlignRight)
        self._master_layout.setColumnFixedWidth(0, 15)  # Input socket column
        self._master_layout.setColumnFixedWidth(2, 15)  # Output socket column
        self._master_layout.setColumnFixedWidth(1, self._width - 15 - 15)

        self.setLayout(self._master_layout)

    def _init_sockets(self):
        shader = self._node.get_shader()

        for i in range(self._node.num_output_sockets()):
            shader_output = shader.get_outputs()[i]
            label = shader_output.get_display_label()
            socket = self._node.get_output_socket(i)
            socket.set_index(i)
            self._add_output_module(output_label=label, node_socket=socket)

        for i in range(self._node.num_input_sockets()):
            shader_input = shader.get_inputs()[i]
            label = shader_input.get_display_label()
            ran = shader_input.get_range()
            socket = self._node.get_input_socket(i)
            socket.set_index(i)
            self._add_input_module(input_label=label, node_socket=socket, input_range=ran, is_connectable=shader_input.is_connectable())

    def _notify_change(self):
        """Event is called when any of this node's widget's inputs are changed"""
        self.input_changed.emit(self)

    def delete(self):
        # TODO Implement!
        pass

    def id(self) -> uuid.UUID:
        return self._node.id()

    def label(self) -> str:
        return self._node.label()

    def set_label(self, label: str = None):
        """
        Sets the label of this node and updates the node title accordingly.
        :param label: the new label, or None to use the old label and just update the title.
        """
        if label is None:
            label = self.label()
        self._node.set_label(label)
        self._title_item.setPlainText(self.label() + " (" + str(self.get_num()) + ")")

    def get_num(self) -> int:
        """Returns the number that is assigned to this node. This number is unique among nodes with the same shader type."""
        return self._num

    def set_num(self, num: int):
        self._num = num
        self.set_label()

    def _init_title(self):
        self._title_item.setDefaultTextColor(self._title_color)
        self._title_item.setFont(self._title_font)
        self._title_item.setPos(self._padding, 0)
        self._title_item.setTextWidth(self._width - self._padding)
        self.set_label()

    def is_deletable(self) -> bool:
        return self._deletable

    def set_deletable(self, value: bool):
        self._deletable = value

    def get_input(self, exclude_connected: True) -> typing.List[typing.Tuple[str, str, typing.Any]]:
        """Returns a list of argument names, its modified name as well as the the value in the node input for that argument.

        Note that is the shader held by this node has not been recompiled, the modified name of the argument is undefined."""
        out = []
        for socket, mod in self._in_socket_modules:
            if not exclude_connected or not socket.is_connected():
                value = mod.get_gl_value()
                argument: str = socket.label()
                modified_name = self.get_shader().get_parsed_code().primary_function.get_argument(argument).get_modified_name()

                out.append((argument, modified_name, value))

        return out

    def get_ancestor_nodes(self, add_self: bool = False) -> IndexedSet:
        """
        Returns a list of all connected ancestors of this node.
        :param add_self: if True, adds this node to the set of returned nodes.
        :return: a Set of nodes that are ancestors of this node.
        """
        nodes = self._node.get_ancestor_nodes(add_self=add_self)
        out = IndexedSet()
        for n in nodes:
            out.add(n.get_container())
        return out

    def save_graph_state(self):
        """Saves the state (socket values) of this node and all ancestor nodes."""
        self._node.save_graph_state()

    def save_state(self):
        """Saves the values of all sockets of this node."""
        self._node.save_state()

    def restore_graph_state(self):
        """Restores the state (socket values) of this node and all ancestor nodes."""
        self._node.restore_graph_state()

    def restore_state(self):
        """Restores the values of all sockets of this node."""
        self._node.restore_state()

    def get_backend_node(self) -> ShaderNode:
        return self._node

    def get_shader(self) -> Shader:
        return self._node.get_shader()

    def has_socket(self, socket: GNodeSocket) -> bool:
        return self._node.has_socket(socket.get_backend_socket())

    def get_input_sockets(self) -> typing.List[GNodeSocket]:
        return [s.get_container() for s in self._node.get_input_sockets()]

    def get_input_socket(self, index: int) -> typing.Union[GNodeSocket, None]:
        return self._node.get_input_socket(index).get_container()

    def get_input_module(self, socket=None, index=None):
        assert not (socket is None and index is None), "Specify at least one identifier to find input module!"

        for i, (s, m) in enumerate(self._in_socket_modules):
            if socket and s == socket:
                return m
            elif index and i == index:
                return m

        return None

    def get_output_sockets(self) -> typing.List[GNodeSocket]:
        return [s.get_container() for s in self._node.get_output_sockets()]

    def get_output_socket(self, identifier) -> 'GNodeSocket':
        return self._node.get_output_socket(identifier).get_container()

    def _create_g_socket(self, socket: NodeSocket) -> GNodeSocket:
        socket = GNodeSocket(self, socket)
        socket.edge_started.connect(self._spawn_edge)
        socket.edge_released.connect(self._release_edge)
        socket.connection_changed.connect(self._handle_socket_connection)

        return socket

    def _add_input_module(self, input_label: str, node_socket: NodeSocket, input_range: (float, float) = (0, 1), is_connectable: bool = True):
        dtype = node_socket.dtype()
        socket = self._create_g_socket(node_socket)

        if dtype == DataType.Float:
            # Create an widgets widget
            input_widget = FloatInput(min_=input_range[0], max_=input_range[1], dtype=dtype)
        elif dtype == DataType.Int:
            input_widget = IntInput(min_=input_range[0], max_=input_range[1], dtype=dtype)
        elif dtype == DataType.Vec3_RGB:
            input_widget = ColorInput(dtype)
        elif dtype == DataType.Shader:
            input_widget = ShaderInput(dtype)
        elif dtype == DataType.Vec3_Float:
            size = 3
            input_widget = ArrayInput(size, min_=input_range[0], max_=input_range[1], dtype=dtype)

        else:
            raise TypeError("Data Type {} is not yet supported!".format(dtype))

        # Create a module and add to this node
        module = SocketModule(socket, input_label, input_widget)
        module.input_changed.connect(self._notify_change)
        module.set_label_palette(self._input_label_palette)
        module.set_value(socket.value())
        self._in_socket_modules.append((socket, module))

        module_item = self.node_scene.addWidget(module)
        self._master_layout.addItem(socket, self._input_index, 0, Qt.AlignVCenter)
        self._master_layout.addItem(module_item, self._input_index, 1, Qt.AlignVCenter)
        self._input_index += 1
        self._height = self._input_index * 40

        if not is_connectable:
            socket.setVisible(False)

    def _add_output_module(self, output_label: str, node_socket: NodeSocket):
        socket = self._create_g_socket(node_socket)
        module = OutputModule(output_label)
        module.set_label_palette(self._input_label_palette)
        module_item = self.node_scene.addWidget(module)
        self._out_socket_modules.append((socket, module))

        self._master_layout.addItem(module_item, self._output_index, 1, Qt.AlignVCenter)
        self._master_layout.addItem(socket, self._input_index, 2, Qt.AlignVCenter)
        self._output_index += 1
        self._input_index += 1

    def _handle_socket_connection(self, socket: GNodeSocket, _):

        if socket.is_connected():  # Disable socket module to show that is not user controllable anymore
            for (s, mod) in self._in_socket_modules:
                if s == socket:
                    mod.setEnabled(False)

        self.connection_changed.emit(socket, self)

    def _spawn_edge(self, edge):
        self.edge_started.emit(self.id(), edge)

    def _release_edge(self, edge):
        self.edge_ended.emit(self.id(), edge)

    def boundingRect(self) -> QtCore.QRectF:
        return QRectF(0, 0, self._width, self._height).normalized()

    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            painter.setPen(QPen(Qt.black))  # Disables the border
        else:
            painter.setPen(Qt.NoPen)

        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(0, 0, self._width, self._height, self._rounding, 1)

    def render(self, width: int, height: int, retain_graph=False) -> typing.Tuple[torch.Tensor, dict]:
        """
        Renders an image from this node graph.

        :param width: pixel width of rendered image
        :param height: pixel height of rendered image
        :param retain_graph: If True, updated socket values will not be fetched, instead, saved tensor values will be used. If using
            backpropagation that updates the returned tensor parameters in-place, set this to True, otherwise set to False so that parameter values
            are fetched from input Sockets.
        :return: a Tensor containing the rendered image and a list of parameter Tensors (one for each unconnected graph input)
        """
        return self._node.render(width, height, retain_graph=retain_graph)

    def __str__(self):
        cls = self.__class__
        return "{} '{}({})'".format(cls, self.label(), self.get_num())

    def __hash__(self):
        return self._node.__hash__()

    def __eq__(self, other):
        if isinstance(other, GShaderNode):
            return self._node.__eq__(other.get_backend_node())

        return False


class GMaterialOutputNode(GShaderNode):
    graph_changed = pyqtSignal()

    def __init__(self, node_scene: NodeScene, **kwargs):
        super().__init__(node_scene, shader=MaterialOutputShader(), label="Material Output Node", **kwargs)

        # define data properties
        self._deletable = False
        self._program = self.get_shader().get_program()
        self._io_mapping = {}  # node.id() -> (uniform name, modified uniform name)
        self._connected_node = None

    def _handle_socket_connection(self, socket: GNodeSocket, edge: Edge):
        # Start tracking the connected node, so that each time it gets connected, the Material Output Node is notified
        self._connected_node = socket.get_connected_nodes()[0]
        self._handle_graph_change()

    def _handle_graph_change(self, *args):
        connected_nodes = self._connected_node.get_ancestor_nodes(add_self=True)

        for n in connected_nodes:
            n.connection_changed.connect(self._handle_graph_change)
            n.input_changed.connect(self._handle_input_changed)

        self._recompile()  # Compile and get new program
        # self._create_io_mapping(connected_nodes)  # needs to be called after the code has been compiled.
        for node in connected_nodes:
            self._handle_input_changed(node)  # send current input from not node_graph to program
        self.graph_changed.emit()  # Notify the material that the node_graph has changed and there's a new program ready

    def _recompile(self):
        self.get_shader().recompile(self)
        self._program = self.get_shader().get_program()

    def _create_io_mapping(self, nodes: typing.List[ShaderNode]):
        """Creates a mapping that tracks nodes and their modified argument names."""
        for node in nodes:
            node_id = node.id()
            if node_id not in self._io_mapping:
                self._io_mapping[node_id] = {}

            node_inputs = node.get_input(exclude_connected=True)

            for arg, mod_arg, _ in node_inputs:
                self._io_mapping[node_id][arg] = mod_arg

    def _handle_input_changed(self, node: GShaderNode):
        set_program_uniforms_from_node(self._program, node)

    def get_program(self, copy=True, set_input=True) -> Program:
        """
        Gets the full compiled Program of the node tree.
        :param copy: If True, recompiles the shader and returns a copy of the Program or else returns the instance held by this Node.
        :param set_input: If True, will set the appropriate input to the program from all connected nodes. Only applicable is copy=True.
        :return: a gloo.Program instance.
        """
        if copy:
            _, _, program = self.get_shader().compile(self)
            if set_input:
                connected_nodes = self._connected_node.get_ancestor_nodes(add_self=True)
                for node in connected_nodes:
                    set_program_uniforms_from_node(program, node)

            return program
        else:
            return self._program

    def can_render(self) -> bool:
        """
        Checks whether any shader is connected to this output node and is thus ready to render.

        :return: True if this node can render, False otherwise.
        """
        for socket in self.get_input_sockets():
            if socket.is_connected():
                return True

        return False

    def render(self, width, height, retain_graph=False) -> typing.Tuple[typing.Union[None, torch.Tensor], dict]:
        if self.can_render():
            return super().render(width, height, retain_graph=retain_graph)

        return None, dict()  # The input is not getting fed a shader, and we can't render anything
