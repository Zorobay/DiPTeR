import logging
import typing

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsLineItem
from node_graph.edge import Edge

_logger = logging.getLogger(__name__)


class GEdge(QGraphicsLineItem):

    def __init__(self, source: 'GNodeSocket' = None, destination: 'GNodeSocket' = None):
        super().__init__()
        assert (source or destination), "Need to set (at least) either source or destination socket on GEdge creation!"
        self._src_socket = source
        self._dest_socket = destination
        self._tip_pos = None

        self._edge_pen = QPen(QColor(200, 130, 0, 255), 3)
        self._init_item()

    def _init_item(self):
        self.setPen(self._edge_pen)

    @classmethod
    def from_edge(cls, edge: Edge) -> 'GEdge':
        gedge = GEdge(edge.get_source().get_container(), edge.get_destination().get_container())
        gedge.update_edge()
        return gedge

    def set_source_socket(self, socket: 'GNodeSocket'):
        """Sets the socket that is to be the source of this Edge."""
        self._src_socket = socket
        self.update_edge()

    def get_source_socket(self) -> 'GNodeSocket':
        """Gets the socket that has been set as source for this Edge, or None if not set."""
        return self._src_socket

    def set_destination_socket(self, socket: 'GNodeSocket'):
        """Sets the socket that is to be the destination of this Edge."""
        self._dest_socket = socket
        self.update_edge()

    def get_destination_socket(self) -> 'GNodeSocket':
        """Gets the socket that has been set as destination for this Edge, or None if not set."""
        return self._dest_socket

    def get_other_socket(self, socket: 'GNodeSocket') -> typing.Union['GNodeSocket', None]:
        """Gets the socket of this edge that is not equal to the input socket 'socket'. If 'socket' is not part of this edge or if 'socket' is the
        only socket connected to this edge, returns None."""
        if socket == self._src_socket:
            return self._dest_socket
        elif socket == self._dest_socket:
            return self._src_socket
        return None

    def connect_sockets(self):
        """Connects destination and source sockets internally if they are both specified."""
        if self.is_fully_connected():
            self._src_socket.add_connecting_edge(self)
            self._dest_socket.add_connecting_edge(self)
            self._src_socket.connect_to(self._dest_socket)
            _logger.debug("Connected source socket {} of node {} to destination socket {} of node {}."
                          .format(self._src_socket.label(), self._src_socket.parent_node().label(), self._dest_socket.label(),
                                  self._dest_socket.parent_node().label()))

    def is_fully_connected(self) -> bool:
        """
        Indicates whether this Edge has connected source and destionation sockets.
        :return: True is connected at both ends, False otherwise.
        """
        return self._dest_socket is not None and self._src_socket is not None

    def src_pos(self) -> typing.Union[QPointF, None]:
        if self._src_socket:
            return self._src_socket.get_scene_pos()
        return self._tip_pos

    def dest_pos(self) -> typing.Union[QPointF, None]:
        if self._dest_socket:
            return self._dest_socket.get_scene_pos()
        return self._tip_pos

    def set_tip_pos(self, pos: QPointF):
        """
        Set a new position for the tip of this edge while moving it.
        :param pos: the new position on the scene.
        """
        self._tip_pos = pos
        self.update_edge()

    def update_edge(self):
        if self.src_pos() and self.dest_pos():
            dest, source = self.src_pos(), self.dest_pos()
            self.setLine(source.x(), source.y(), dest.x(), dest.y())
