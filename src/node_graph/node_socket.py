import enum
import logging
import typing
import uuid

from node_graph.edge import Edge
from node_graph.data_type import DataType
from node_graph.graph_element import GraphElement

_logger = logging.getLogger(__name__)


class SocketType(enum.IntEnum):
    INPUT = 0
    OUTPUT = 1


class NodeSocket(GraphElement):
    INPUT: SocketType = SocketType.INPUT
    OUTPUT: SocketType = SocketType.OUTPUT

    def __init__(self, parent_node: 'Node', socket_type: SocketType, dtype: DataType = None, label: str = "", container=None):
        super().__init__(container)
        assert isinstance(socket_type, SocketType), "socket_type must be of enum type SocketType!"
        self._parent_g_node = parent_node
        self._type = socket_type
        self._dtype = dtype
        self._label = label
        self._id = uuid.uuid4()
        self._index = -1
        self._value = None
        self._saved_value = None
        self._connected = False
        self._connected_sockets = set()
        self._connected_edges = set()

    def connect_to(self, other_socket: 'NodeSocket') -> Edge:
        """
        Connects this NodeSocket to another NodeSocket.
        :param other_socket: Other NodeSocket to connect this socket to.
        :return: the Edge that was created between the sockets, or the old Edge if there already exists a connection.
        """
        assert self.id() != other_socket.id(), "Can not connect socket to itself!"
        assert self.type() != other_socket.type(), "Can not connect sockets of the same type!"

        # Check if this socket is already connected to 'socket'
        if other_socket in self._connected_sockets:
            edge = self._find_connecting_edge(other_socket)
            return edge

        # Create a directed edge
        if self.type() == NodeSocket.OUTPUT:
            edge = Edge(source=self, destination=other_socket)
        else:
            edge = Edge(source=other_socket, destination=self)

        self._connected_edges.add(edge)
        other_socket._connected_edges.add(edge)
        self._connected_sockets.add(other_socket)
        other_socket._connected_sockets.add(self)
        self._connected = True
        other_socket._connected = True

        return edge

    def disconnect_from(self, other_socket: 'NodeSocket'):
        """
        Disconnects this NodeSocket from another NodeSocket.
        :param other_socket: Another NodeSocket to disconnect from.
        """
        assert other_socket in self._connected_sockets, "Can not disconnect from a socket that is not connected!"

        edge = self._find_connecting_edge(other_socket)

        if edge is not None:
            self._connected_sockets.remove(other_socket)
            other_socket._connected_sockets.remove(self)

            self._connected_edges.remove(edge)
            other_socket._connected_edges.remove(edge)
            del edge

            self._connected = False
            other_socket._connected = False
        else:
            raise RuntimeError("Sockets are indicated as connected but could not find their connecting Edge. The Node Graph is corrupt!")

    def get_parent_node(self) -> 'Node':
        return self._parent_g_node

    def get_connected_nodes(self) -> typing.List['Node']:
        """
        Returns each node that is connected to this NodeSocket.
        :return: A list of Nodes connected to this NodeSocket.
        """

        nodes = []
        for socket in self._connected_sockets:
            nodes.append(socket.get_parent_node())

        return nodes

    def get_connected_sockets(self) -> list:
        """
        Returns each NodeSocket that this NodeSocket is connected to.
        :return: A list of NodeSockets connected to this NodeSocket.
        """

        return self._connected_sockets.copy()

    def set_index(self, index):
        """Set the index of this socket. This value is used to figure out what input/output this socket corresponds to."""
        self._index = index

    def get_index(self) -> int:
        """Gets the index of this socket."""
        return self._index

    def set_value(self, value: typing.Any):
        self._value = value

    def save_value(self):
        """Saves the value of this socket internally. This value can be reassigned by calling 'restore_value()'."""

        self._saved_value = self._value.detach().clone()

    def restore_value(self):
        """Restores the value of this socket to the last saved value."""

        self._value = self._saved_value.detach().clone()

    def value(self) -> typing.Any:
        return self._value

    def label(self) -> str:
        return self._label

    def is_connected(self) -> bool:
        """Returns a boolean indicating whether this socket is connected to any other sockets with an Edge."""
        return self._connected

    def type(self) -> SocketType:
        """Returns the type of this NodeSocket indicating whether it accepts input or returns output."""
        return self._type

    def dtype(self) -> DataType:
        return self._dtype

    def _find_connecting_edge(self, other_socket: 'NodeSocket') -> typing.Union[None, Edge]:
        for edge in self._connected_edges:
            if edge.connects(self, other_socket):
                return edge

        return None
