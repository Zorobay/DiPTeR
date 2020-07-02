import typing
import uuid

import torch
from node_graph.data_type import DataType
from node_graph.graph_element import GraphElement
from node_graph.parameter import Parameter
from src.node_graph.node_socket import NodeSocket, SocketType
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


class Node(GraphElement):

    def __init__(self, label: str = "", container=None):
        """
        Create a new node with inputs and outputs.

        :param label: an optional node label
        """
        super().__init__(container)
        self._id = uuid.uuid4()
        self._in_sockets = list()
        self._out_sockets = list()
        self._label = label

    def add_socket(self, type_: SocketType, label: str = "", dtype: DataType = None) -> NodeSocket:
        """
        Adds an input socket to this Node
        :return: the created Socket
        """
        socket = NodeSocket(self, type_, dtype=dtype, label=label)

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

    def get_input_sockets(self) -> typing.List[NodeSocket]:
        """Returns a list with all NodeSockets that are inputs to this Node."""
        return self._in_sockets.copy()

    def num_input_sockets(self) -> int:
        """Returns the number of input NodeSockets of this Node"""
        return len(self._in_sockets)

    def get_output_socket(self, index: int) -> typing.Union[NodeSocket, None]:
        """
        Finds an output socket by index and returns it.
        :param index: the index of the wanted socket.
        :return: The output Socket, or None if the index is invalid.
        """

        if index < len(self._out_sockets):
            return self._out_sockets[index]

        return None

    def get_output_sockets(self) -> typing.List[NodeSocket]:
        """Returns a list of NodeSockets that are outputs to this Node."""
        return self._out_sockets.copy()

    def set_value(self, index: int, value: typing.Any):
        """
        Set the value of an input socket. If this socket is connected, does nothing.
        :param index: Index of the socket
        :param value: The new value of the socket.
        """
        socket = self.get_input_socket(index)
        if socket:
            socket.set_value(value)

    def get_ancestor_nodes(self, add_self: bool = False) -> typing.Set['Node']:
        """
        Returns a list of all connected ancestors of this node.
        :param add_self: if True, adds this node to the set of returned nodes.
        :return: a Set of nodes that are ancestors of this node.
        """
        out = set()
        if add_self:
            out.add(self)

        for socket in self.get_input_sockets():
            if socket.is_connected():
                for node in [n for n in socket.get_connected_nodes()]:
                    out.update(node.get_ancestor_nodes(add_self=True))

        return out

    def save_graph_state(self):
        """Saves the state (socket values) of this node and all ancestor nodes."""

        for node in self.get_ancestor_nodes(add_self=True):
            node.save_state()

    def save_state(self):
        """Saves the values of all sockets of this node."""
        for s in self._in_sockets:
            s.save_value()

    def restore_graph_state(self):
        """Restores the state (socket values) of this node and all ancestor nodes."""
        for node in self.get_ancestor_nodes(add_self=True):
            node.restore_state()

    def restore_state(self):
        """Restores the values of all sockets of this node."""
        for s in self._in_sockets:
            s.restore_value()

    def num_output_sockets(self) -> int:
        """Returns the number of output NodeSockets of this Node"""
        return len(self._out_sockets)

    def has_socket(self, socket: NodeSocket) -> bool:
        """Returns a boolean indicating whether the input NodeSocket 'socket' belongs to this Node."""
        if socket in self._out_sockets:
            return True
        elif socket in self._in_sockets:
            return True

        return False

    def set_label(self, label: str):
        self._label = label

    def label(self) -> str:
        return self._label

    def __str__(self):
        cls = self.__class__
        return "{} '{}'".format(cls, self._label)


class ShaderNode(Node):

    def __init__(self, shader: Shader, set_default_inputs: bool = True, **kwargs):
        """
        Creates a new ShaderNode that can render an image. A shader must be supplied.
        :param shader: The Shader class for this node.
        :param set_default_inputs: If True, will set the values of input sockets to the defaults specified in the Shader class.
        :param kwargs:
        """
        super().__init__(**kwargs)

        if "label" not in kwargs:
            self.set_label(shader.__class__)

        self._shader = shader
        self._render_parameters = dict()

        self._init()

        if set_default_inputs:
            self.set_default_inputs()

    def _init(self):
        for inp in self._shader.get_inputs():
            self.add_socket(NodeSocket.INPUT, label=inp.get_argument(), dtype=inp.dtype())

        for out in self._shader.get_outputs():
            self.add_socket(NodeSocket.OUTPUT, label=out.get_display_label(), dtype=out.dtype())

    def get_shader(self) -> Shader:
        return self._shader

    def set_default_inputs(self):
        """
        Sets the values of the input NodeSockets of this Node to the default values according to the specification in this Node's Shader class.
        """
        for i, inp in enumerate(self._shader.get_inputs()):
            self._in_sockets[i].set_value(inp.get_default())

    def render(self, width: int, height: int, retain_graph: bool = False) -> typing.Tuple[Tensor, list]:
        """
        Renders an image from this node graph.

        :param width: pixel width of rendered image
        :param height: pixel height of rendered image
        :param retain_graph: If True, updated socket values will not be fetched, instead, saved tensor values will be used. If using
            backpropagation that updates the returned tensor parameters in-place, set this to True, otherwise set to False so that parameter values
            are fetched from input Sockets.
        :return: a Tensor containing the rendered image and a dictionary of modified argument names mapped to parameter Tensors (one for each
            unconnected graph input)
        """
        Shader.set_render_size(width, height)
        shader_inputs = self.get_shader().get_inputs()
        assert len(shader_inputs) == len(self._in_sockets)

        complete_params_dict = {}
        arguments = {}

        for i, socket in enumerate(self._in_sockets):
            arg = socket.label()
            inp = self._shader.get_input_by_arg(arg)
            mod_arg = self.get_shader().get_parsed_code().get_modified_arg_name(arg)
            if socket.is_connected():
                nodes = socket.get_connected_nodes()
                assert len(nodes) == 1  # It should be an input node, so it should only be able to have 1 connected node
                t, ad = nodes[0].render(width, height, retain_graph=retain_graph)
                complete_params_dict.update(ad)
            else:
                if arg in self._render_parameters and retain_graph:  # Argument is already fetched, get saved reference
                    p = self._render_parameters[arg]
                else:
                    value = socket.value()
                    if isinstance(value, torch.Tensor):
                        t = value.clone().detach()
                    else:
                        t = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
                    p = Parameter(inp, t)
                    self._render_parameters[arg] = p
                complete_params_dict[mod_arg] = p
                t = p.tensor()

            arguments[arg] = t

        return self._shade(arguments), complete_params_dict

    def _shade(self, args: dict):
        width, height = Shader.width, Shader.height
        mat_args = dict()
        for key, arg in args.items():
            if len(arg.shape) == 3 and arg.shape[0] == width and arg.shape[1] == height:
                mat_args[key] = arg.float()
            else:
                mat_args[key] = arg.repeat(width, height, 1).float()

        return self.get_shader().shade_mat(**mat_args)
