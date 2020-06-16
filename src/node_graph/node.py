import time
import typing
import uuid

import torch
from src.graph.node_socket import NodeSocket, SocketType
from src.shaders.brick_shader import BrickShader
from src.shaders.frag_coord_shader import FragmentCoordinatesShader
from src.shaders.hsv_shader import HSVShader
from src.shaders.shader_super import Shader
from torch import Tensor

TYPE_VALUE = "type_value"
TYPE_FUNC = "type_func"
TYPE = "type"
VALUE = "value"
ARGS = "args"
MODIFIED_ARG = "mod_arg"
NAME_ARG = "arg"
NAME_MODIFIED_ARG = "mod_arg"


class Node:

    def __init__(self, label: str = ""):
        """
        Create a new node with inputs and outputs.

        :param label: an optional node label
        """
        self._id = uuid.uuid4()
        self._in_sockets = list()
        self._out_sockets = list()
        self._label = label

    def add_socket(self, type_: SocketType, label: str = "") -> NodeSocket:
        """
        Adds an input socket to this Node
        :return: the created Socket
        """
        socket = NodeSocket(self, type_, label)

        if type_ == NodeSocket.INPUT:
            self._in_sockets.append(socket)
        elif type_ == NodeSocket.OUTPUT:
            self._out_sockets.append(socket)

        return socket

    def remove_socket(self, socket: NodeSocket) -> bool:
        """
        Removes a socket from this node.
        :param socket: the socket of this node to be removed.
        :return: True if successful, False otherwise.
        """
        if socket in self._out_sockets:
            self._out_sockets.remove(socket)
            return True
        elif socket in self._in_sockets:
            self._in_sockets.remove(socket)
            return True

        return False

    def get_input_socket(self, index: int) -> typing.Union[NodeSocket, None]:
        """
        Finds an input socket by index and returns it.
        :param index: the index of the wanted socket.
        :return: The input Socket, or None if the index is invalid.
        """

        if index < len(self._in_sockets):
            return self._in_sockets[index]

        return None

    def get_output_socket(self, index: int) -> typing.Union[NodeSocket, None]:
        """
        Finds an output socket by index and returns it.
        :param index: the index of the wanted socket.
        :return: The output Socket, or None if the index is invalid.
        """

        if index < len(self._out_sockets):
            return self._out_sockets[index]

        return None

    def set_label(self, label: str):
        self._label = label

    def id(self) -> uuid.UUID:
        return self._id

    def label(self) -> str:
        return self._label

    def __eq__(self, other):
        if isinstance(other, Node):
            return other.id() == self.id()

        return False

    def __hash__(self):
        return self._id.__hash__()

    def __str__(self):
        cls = self.__class__
        return "{} '{}'".format(cls, self._label)


class ShaderNode(Node):

    def __init__(self, shader: Shader, label: str = "", set_default_inputs: bool = False):
        super().__init__(label)

        if not label:
            self.set_label(shader.__class__)

        self._shader = shader

        self._init()

        if set_default_inputs:
            self.set_default_inputs()

    def _init(self):
        for _, label, _, _, _ in self._shader.get_inputs():
            self.add_socket(NodeSocket.INPUT, label)

        for label, _ in self._shader.get_outputs():
            self.add_socket(NodeSocket.OUTPUT, label)

    def get_shader(self) -> Shader:
        return self._shader

    def set_default_inputs(self):
        """
        Sets the values of the input NodeSockets of this Node to the default values according to the specification in this Node's Shader class.
        """
        for i, (_, label, _, _, default) in enumerate(self._shader.get_inputs()):
            self._in_sockets[i].set_value(default)

    def render(self, width: int, height: int) -> Tensor:
        Shader.set_render_size(width, height)
        shader_inputs = self.get_shader().get_inputs()
        assert len(shader_inputs) == len(self._in_sockets)

        arguments = {}

        for i, socket in enumerate(self._in_sockets):
            arg = socket.label()
            if socket.is_connected():
                nodes = socket.get_connected_nodes()
                assert len(nodes) == 1  # It should be an input node, so it should only be able to have 1 connected node
                t = nodes[0].render(width, height)
            else:
                value = socket.value()
                t = torch.tensor(value, dtype=torch.float32).unsqueeze(0)

            arguments[arg] = t

        return self._shade(arguments)

    def _shade(self, args: dict):
        width, height = Shader.width, Shader.height
        mat_args = dict()
        for key, arg in args.items():
            if len(arg.shape) == 3 and arg.shape[0] == width and arg.shape[1] == height:
                mat_args[key] = arg.float()
            else:
                mat_args[key] = arg.repeat(width, height, 1).float()

        return self.get_shader().shade_mat(**mat_args)
