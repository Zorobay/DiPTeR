import abc
import typing
import uuid

import numpy as np
import torch
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QFont, QColor, QPalette, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsWidget, QGraphicsLinearLayout
from glumpy.gloo import Program

# from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.edge import Edge
from src.gui.node_editor.layouts import GraphicsGridLayout
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.gui.widgets.array_input import ArrayInput
from src.gui.widgets.color_input import ColorInput
from src.gui.widgets.io_module import InputModule, OutputModule
from src.gui.widgets.line_input import FloatInput, IntInput
from src.gui.widgets.shader_input import ShaderInput
from src.opengl.internal_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGB, INTERNAL_TYPE_SHADER, \
    INTERNAL_TYPE_ARRAY_FLOAT, INTERNAL_TYPE_INT
from src.shaders.material_output_shader import MaterialOutputShader
from src.shaders.shader_super import Shader, CompilableShader

TYPE_VALUE = "type_value"
TYPE_FUNC = "type_func"
TYPE = "type"
VALUE = "value"
ARGS = "args"
MODIFIED_ARG = "mod_arg"
NAME_ARG = "arg"
NAME_MODIFIED_ARG = "mod_arg"


def set_program_uniforms_from_node(program: Program, node: 'Node'):
    inputs = node.get_input(exclude_connected=True)

    for arg, mod_arg, value in inputs:
        program[mod_arg] = value


class Node(QGraphicsWidget):
    """This abstract class defines the look and feel of a Node. Specialized classes can subclass this instead of the Node class.
    """
    edge_started = pyqtSignal(uuid.UUID, Edge)
    edge_ended = pyqtSignal(uuid.UUID, Edge)
    connection_changed = pyqtSignal(Socket, Edge, object)  # Node

    @abc.abstractmethod
    def __init__(self, node_scene: NodeScene, title: str, parent=None):
        super().__init__(parent)

        self.node_scene = node_scene
        self._title = title
        self._id = uuid.uuid4()
        self._num = -1

        self._in_sockets = {}
        self._out_sockets = {}
        self._socket_modules = []
        self._socket_connections = {}

        # define Node properties
        self._selected = False
        self._deletable = True
        self._input_index = 2
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

    def _init_layout(self):
        self._master_layout.setContentsMargins(-5, 4, 4, -5)
        self._master_layout.setRowSpacing(0, self._title_font.pointSize() + 12)  # Add empty space for first row so that title is visible
        self._master_layout.setHorizontalSpacing(2.0)
        self._master_layout.setVerticalSpacing(0.0)
        self._master_layout.setColumnAlignment(2, Qt.AlignRight)
        self._master_layout.setRowAlignment(1, Qt.AlignVCenter)
        self._master_layout.setColumnFixedWidth(0, 15)  # Input socket column
        self._master_layout.setColumnFixedWidth(2, 15)  # Output socket column
        self._master_layout.setColumnFixedWidth(1, self._width-15-15)

        self.setLayout(self._master_layout)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id

        return False

    def __hash__(self):
        return self.id.__hash__()

    def __str__(self):
        return "({}) {}".format(self.get_title(), self.__class__)

    def delete(self):
        pass

    def get_title(self):
        return self._title

    def set_title_text(self, title_text: str):
        self._title = title_text
        self._title_item.setPlainText(self.get_title())

    @property
    def id(self):
        return self._id

    def get_num(self) -> int:
        """Returns the number that is assigned to this node. This number is unique among nodes with the same shader type."""
        return self._num

    def set_num(self, num: int):
        self._num = num

    def _init_title(self):
        self._title_item.setDefaultTextColor(self._title_color)
        self._title_item.setFont(self._title_font)
        self._title_item.setPos(self._padding, 0)
        self._title_item.setTextWidth(self._width - self._padding)
        self.set_title_text(self._title)

    def isDeletable(self) -> bool:
        return self._deletable

    def setDeletable(self, value: bool):
        self._deletable = value

    def get_ancestor_nodes(self, add_self: bool = False) -> typing.Set['Node']:
        """
        Returns a list of all connected ancestors of this node.
        :param add_self: if True, adds this node to the set of returned nodes.
        :return: a Set of nodes that are ancestors of this node.
        """
        out = set()
        if add_self:
            out.add(self)

        for _, socket in self._in_sockets.items():
            if socket.isConnected():
                for node in [s.parent_node for s in socket.get_connected_sockets()]:
                    out.update(node.get_ancestor_nodes(add_self=True))

        return out

    def get_socket_type(self, socket_id: uuid.UUID) -> int:
        return self.get_all_sockets()[socket_id].socket_type

    def has_socket(self, socket: Socket) -> bool:
        return socket in self.get_all_sockets()

    def get_in_sockets(self) -> typing.Dict[uuid.UUID, Socket]:
        return self._in_sockets

    def get_out_sockets(self) -> typing.Dict[uuid.UUID, Socket]:
        return self._out_sockets

    def get_all_sockets(self) -> typing.Dict[uuid.UUID, Socket]:
        return {**self._in_sockets, **self._out_sockets}

    def create_input_socket(self, argument: str) -> Socket:
        return self._create_socket(Socket.SOCKET_INPUT, argument)

    def create_output_socket(self) -> Socket:
        return self._create_socket(Socket.SOCKET_OUTPUT, None)

    def _create_socket(self, socket_type: int, argument: str) -> Socket:
        socket = Socket(self, socket_type, argument)
        socket.edge_started.connect(self._spawn_edge)
        socket.edge_released.connect(self._release_edge)
        socket.connection_changed.connect(self._handle_socket_connection)

        if socket.socket_type == Socket.SOCKET_INPUT:
            self._in_sockets[socket.id] = socket
        else:
            self._out_sockets[socket.id] = socket

        return socket

    def _handle_socket_connection(self, socket: Socket, edge: Edge):
        self.connection_changed.emit(socket, edge, self)

    def _spawn_edge(self, edge):
        self.edge_started.emit(self.id, edge)

    def _release_edge(self, edge):
        self.edge_ended.emit(self.id, edge)

    def boundingRect(self) -> QtCore.QRectF:
        return QRectF(0, 0, self._width, self._height).normalized()

    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            painter.setPen(QPen(Qt.black))  # Disables the border
        else:
            painter.setPen(Qt.NoPen)

        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(0, 0, self._width, self._height, self._rounding, 1)


class ShaderNode(Node):
    input_changed = pyqtSignal(object)  # Node

    def __init__(self, node_scene: NodeScene, title: str, shader: Shader, parent=None):
        super().__init__(node_scene, title, parent)

        # define data properties
        self._socket_modules = []  # Tracks which module belongs to which socket
        self._shader = shader

        # Initialize the widget
        self._init_widget()
        self.set_title_text(self._title)

    def _init_widget(self):
        for nf, nu, t, ra, de in self._shader.get_inputs():
            self.add_input(nf, nu, t, ra, de)

        for nf, t in self._shader.get_outputs():
            self.add_output(nf, t)

    def _notify_change(self, uniform_var: str, value: typing.Any, internal_type: str, input_id: uuid.UUID):
        """Event is called when any of this node's widget's inputs are changed"""
        self.input_changed.emit(self)

    def get_title(self):
        return super().get_title() + " ({})".format(self.get_num())

    def set_num(self, num: int):
        super().set_num(num)
        self.set_title_text(self._title)  # Update title text (calls subclass' get_title())

    def get_shader(self) -> Shader:
        return self._shader

    def get_input(self, exclude_connected: True) -> typing.List[typing.Tuple[str, str, typing.Any]]:
        """Returns a list of argument names, its modified name as well as the the value in the node input for that argument.

        Note that is the shader held by this node has not been recompiled, the modified name of the argument is undefined."""
        out = []
        for socket, mod in self._socket_modules:
            if not exclude_connected or not socket.isConnected():
                value = mod.get_gl_value()
                argument: str = mod.get_argument()
                modified_name = self.get_shader().get_parsed_code().primary_function.get_argument(argument).get_modified_name()

                out.append((argument, modified_name, value))

        return out

    def add_input(self, input_name: str, uniform_var: str, internal_type: str, input_range: (float, float) = (0, 1),
                  default_value: typing.Any = None):
        socket = self.create_input_socket(uniform_var)

        if internal_type == INTERNAL_TYPE_FLOAT:
            # Create an widgets widget
            input_widget = FloatInput(min_=input_range[0], max_=input_range[1], internal_type=internal_type)
        elif internal_type == INTERNAL_TYPE_INT:
            input_widget = IntInput(min_=input_range[0], max_=input_range[1], internal_type=internal_type)
        elif internal_type == INTERNAL_TYPE_ARRAY_RGB:
            input_widget = ColorInput(internal_type)
        elif internal_type == INTERNAL_TYPE_SHADER:
            input_widget = ShaderInput(internal_type)
        elif internal_type == INTERNAL_TYPE_ARRAY_FLOAT:
            if isinstance(default_value, np.ndarray):
                size = default_value.size
            elif isinstance(default_value, torch.Tensor):
                size = default_value.numel()
            else:
                raise TypeError("Type of default value needs to be numpy array or torch Tensor!")
            input_widget = ArrayInput(size, min_=input_range[0], max_=input_range[1], internal_type=internal_type)

        else:
            raise TypeError("Internal type {} is not yet supported!".format(internal_type))

        # Create an widgets module and add to this node
        module = InputModule(input_name, internal_type, uniform_var, input_widget)
        module.input_changed.connect(self._notify_change)
        module.set_label_palette(self._input_label_palette)
        module.set_default_value(default_value)
        self._socket_modules.append((socket, module))

        module_item = self.node_scene.addWidget(module)
        self._master_layout.addItem(socket, self._input_index, 0)
        self._master_layout.addItem(module_item, self._input_index, 1)
        self._master_layout.setRowAlignment(self._input_index, Qt.AlignBottom)
        self._input_index += 1
        self._height = self._input_index * 40

    def add_output(self, output_name: str, internal_type: str):
        socket = self.create_output_socket()
        output_module = OutputModule(output_name)
        output_module.set_label_palette(self._input_label_palette)
        module_item = self.node_scene.addWidget(output_module)

        self._master_layout.addItem(module_item, 1, 1)
        self._master_layout.addItem(socket, 1, 2)

    def render(self, width: int, height: int, call_dict: dict = None) -> typing.Tuple[torch.Tensor, dict, list, dict]:
        """
        Renders an image from the connected nodes. A dictionary describing the call stack can be provided with parameter 'call_dict'. If only
        rendering, it is not needed. However, if gradient is needed, this needs to be provided from a previous call in order to use the same input
        Tensors.

        :param width: width of the image (in pixels) to be rendered.
        :param height: height of the image (in pixels) to be rendered.
        :param call_dict: A dictionary describing the call stack (as returned by this function). If not supplied, it will be generated and
        returned. Only use if you're changing the values of the Tensors directly. If values is to be taken from the nodes input, leave as None.
        :return: A tuple with a [W,H,3] tensor representing the rendered image, or None if no image could be rendered, a call dictionary, a list of
        parameter Tensors as well as a dictionary of OpenGL uniform values.
        """
        Shader.set_render_size(width, height)
        args_list = []
        if call_dict is None:
            call_dict = dict()
            self._build_call_dict(call_dict, args_list)

        # Evaluate call dict
        args_dict = dict()
        uniform_dict = dict()
        self._eval_dict(args_dict, call_dict, uniform_dict)
        img = self._shade(args_dict)

        return img, call_dict, args_list, uniform_dict

    def _build_call_dict(self, call_dict: dict, args_list: list):
        shader_inputs = self.get_shader().get_inputs()
        assert len(shader_inputs) == len(self._socket_modules)

        for i, (socket, mod) in enumerate(self._socket_modules):
            arg = socket.get_argument()
            mod_arg = self.get_shader().get_parsed_code().get_modified_arg_name(arg)
            type_ = TYPE_VALUE

            assert shader_inputs[i][1] == arg  # Assert that we're working with the same argument
            if socket.isConnected():
                nodes = socket.get_connected_nodes()
                assert len(nodes) == 1  # It should be an input node, so it should only be able to have 1 connected node
                node = nodes[0]
                type_ = TYPE_FUNC
                args = {}
                value = node._shade
                node._build_call_dict(args, args_list)
            else:
                value = torch.tensor(mod.get_gl_value())
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)  # Ensure that we never get zero dimensional tensors
                args_list.append(value)
                args = {}
                # value = value.repeat(width, height, 1)  # Turn it into a matrix argument

            call_dict[arg] = {
                TYPE: type_,
                VALUE: value,
                ARGS: args,
                MODIFIED_ARG: mod_arg
            }

    def _eval_dict(self, args, call_dict, uniform_dict):
        for arg, d in call_dict.items():
            type_ = d[TYPE]
            mod_arg = d[MODIFIED_ARG]
            if type_ == TYPE_FUNC:
                func = d[VALUE]
                func_args = {}
                self._eval_dict(func_args, d[ARGS], uniform_dict)
                args[arg] = func(func_args)
            if type_ == TYPE_VALUE:
                args[arg] = d[VALUE]
                uniform_dict[mod_arg] = d[VALUE].detach().numpy()

    def _shade(self, args: dict):
        width, height = Shader.width, Shader.height
        mat_args = dict()
        for key, arg in args.items():
            if len(arg.shape) == 3 and arg.shape[0] == width and arg.shape[1] == height:
                mat_args[key] = arg
            else:
                mat_args[key] = arg.repeat(width, height, 1)

        return self.get_shader().shade_mat(**mat_args)


class MaterialOutputNode(ShaderNode):
    graph_changed = pyqtSignal()

    def __init__(self, node_scene: NodeScene):
        super().__init__(node_scene, "Material Output Node", MaterialOutputShader())

        # define data properties
        self._deletable = False
        self._program = self._shader.get_program()
        self._io_mapping = {}  # node.id -> (uniform name, modified uniform name)
        self._connected_node = None

    def _handle_socket_connection(self, socket: Socket, edge: Edge):
        # Start tracking the connected node, so that each time it gets connected, the Material Output Node is notified
        if edge.out_socket == socket:
            other_socket = edge.in_socket
        else:
            other_socket = edge.out_socket

        other_node = other_socket.parent_node
        self._connected_node = other_node
        self._handle_graph_change()

    def _handle_graph_change(self, *args):
        connected_nodes = self._connected_node.get_ancestor_nodes(add_self=True)

        for n in connected_nodes:
            n.connection_changed.connect(self._handle_graph_change)
            n.input_changed.connect(self._handle_input_changed)

        self._recompile()  # Compile and get new program
        self._create_io_mapping(connected_nodes)  # needs to be called after the code has been compiled.
        for node in connected_nodes:
            self._handle_input_changed(node)  # send current input from not graph to program
        self.graph_changed.emit()  # Notify the material that the graph has changed and there's a new program ready

    def _recompile(self):
        self.get_shader().recompile(self)
        self._program = self.get_shader().get_program()

    def _create_io_mapping(self, nodes: typing.List[ShaderNode]):
        """Creates a mapping that tracks nodes and their modified argument names."""
        for node in nodes:
            node_id = node.id
            if node_id not in self._io_mapping:
                self._io_mapping[node_id] = {}

            node_inputs = node.get_input(exclude_connected=True)

            for arg, mod_arg, _ in node_inputs:
                self._io_mapping[node_id][arg] = mod_arg

    def _handle_input_changed(self, node: ShaderNode):
        set_program_uniforms_from_node(self._program, node)

    def get_shader(self) -> CompilableShader:
        return self._shader

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

    def render(self, width, height, call_dict: dict = None) -> typing.Tuple[torch.Tensor, dict, list, dict]:
        for _, socket in self.get_in_sockets().items():
            if socket.isConnected():
                return super().render(width, height, call_dict)

        return None, call_dict, [], {}  # The input is not getting fed a shader, and we can't render anything
