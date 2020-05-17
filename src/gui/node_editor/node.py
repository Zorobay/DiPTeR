import abc
import typing
import uuid

import numpy as np
import torch
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QFont, QColor, QPalette, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsWidget
from glumpy.gloo import Program

# from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.edge import Edge
from src.gui.node_editor.layouts import GraphicsGridLayout
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.gui.widgets.array_input import ArrayInput
from src.gui.widgets.color_input import ColorInput
from src.gui.widgets.io_module import InputModule, OutputModule
from src.gui.widgets.line_input import FloatInput
from src.gui.widgets.shader_input import ShaderInput
from src.opengl.internal_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGB, INTERNAL_TYPE_SHADER, \
    INTERNAL_TYPE_ARRAY_FLOAT
from src.shaders.material_output_shader import MaterialOutputShader
from src.shaders.shader_super import FunctionShader, CompilableShader, Shader


class Node(QGraphicsWidget):
    """This abstract class defines the look and feel of a Node. Specialized classes can subclass this instead of the Node class.
    """
    edge_started = pyqtSignal(uuid.UUID, Edge)
    edge_ended = pyqtSignal(uuid.UUID, Edge)
    connection_changed = pyqtSignal(object)  # Node

    @abc.abstractmethod
    def __init__(self, node_scene: NodeScene, title: str, parent=None):
        super().__init__(parent)

        self.node_scene = node_scene
        self._title = title
        self._id = uuid.uuid4()

        self._in_sockets = {}
        self._out_sockets = {}
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

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id

        return False

    def __hash__(self):
        return self.id.__hash__()

    def __str__(self):
        return "({}) {}".format(self.title, self.__class__)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value: str):
        assert isinstance(value, str)
        self._title = value
        self._title_item.setPlainText(self._title)

    @property
    def id(self):
        return self._id

    def _init_layout(self):
        self._master_layout.setContentsMargins(-5, 4, 4, 4)
        self._master_layout.setRowSpacing(0, self._title_font.pointSize() + 12)  # Add empty space for first row so that title is visible
        self._master_layout.setHorizontalSpacing(2.0)
        self._master_layout.setVerticalSpacing(0.0)

        self.setLayout(self._master_layout)

    def _init_title(self):
        self._title_item.setDefaultTextColor(self._title_color)
        self._title_item.setFont(self._title_font)
        self._title_item.setPos(self._padding, 0)
        self._title_item.setTextWidth(self._width - self._padding)
        self.title = self._title

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
        self.connection_changed.emit(self)

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
    input_changed = pyqtSignal(uuid.UUID)

    def __init__(self, node_scene: NodeScene, title: str, shader: FunctionShader, parent=None):
        super().__init__(node_scene, title, parent)

        # define data properties
        self._input_modules = {}
        self._shader = shader
        # self._program = shader.get_program()

        # Initialize the widget
        self._init_widget()

    def _init_widget(self):
        for nf, nu, t, ra, de in self._shader.get_inputs():
            self.add_input(nf, nu, t, ra, de)

        for nf, t in self._shader.get_outputs():
            self.add_output(nf, t)

    def _set_input(self, uniform_var: str, value: typing.Any, internal_type: str, input_id: uuid.UUID):
        """Event is called when any of this node's widgets is changed"""
        #self._shader.set_input_by_uniform(uniform_var, value)
        self.input_changed.emit(self.id)

    def get_shader(self) -> Shader:
        return self._shader

    def add_input(self, input_name: str, uniform_var: str, internal_type: str, input_range: (float, float) = (0, 1),
                  default_value: typing.Any = None):
        socket = self.create_input_socket(uniform_var)

        if internal_type == INTERNAL_TYPE_FLOAT:
            # Create an widgets widget
            input_widget = FloatInput(min_=input_range[0], max_=input_range[1], internal_type=internal_type)
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
        module.input_changed.connect(self._set_input)
        module.set_label_palette(self._input_label_palette)
        module.set_default_value(default_value)

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


class MaterialOutputNode(ShaderNode):

    def __init__(self, node_scene: NodeScene):
        super().__init__(node_scene, "Material Output Node", MaterialOutputShader())

        # define data properties
        self.deletable = False
        self._program = self._shader.get_program()

    def _handle_socket_connection(self, socket: Socket, edge: Edge):
        # Start tracking the connected node, so that each time it gets connected, the Material Output Node is notified
        if edge.out_socket == socket:
            other_socket = edge.in_socket
        else:
            other_socket = edge.out_socket

        other_node = other_socket.parent_node
        connected_nodes = other_node.get_ancestor_nodes(add_self=True)

        self._recompile()

        for n in connected_nodes:
            n.connection_changed.connect(self._recompile)
            n.input_changed.connect(self._handle_input_changed)

    def _recompile(self):
        self.get_shader().recompile(self)

    def _handle_input_changed(self, node_id: uuid.UUID):
        print("Node graph input changed!")
        pass

    def get_program(self) -> Program:
        return self._program
