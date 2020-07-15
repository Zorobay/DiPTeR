import logging
import uuid

import typing
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QColor, QBrush, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLinearLayout, QGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent
from boltons.setutils import IndexedSet
from node_graph.data_type import DataType
from node_graph.edge import Edge
from node_graph.node_socket import NodeSocket, SocketType

from src.gui.node_editor.g_edge import GEdge

_logger = logging.getLogger(__name__)


class GNodeSocket(QGraphicsWidget):
    edge_started = pyqtSignal(GEdge)
    edge_released = pyqtSignal(GEdge)
    connection_changed = pyqtSignal(object, object)  # Socket, Edge or None
    position_changed = pyqtSignal(QPointF)

    INPUT = SocketType.INPUT
    OUTPUT = SocketType.OUTPUT

    def __init__(self, parent_node: 'GShaderNode', socket: NodeSocket):
        super().__init__(parent=parent_node)
        self._socket = socket
        self._socket.set_container(self)
        self._parent_g_node = parent_node
        self._connected_g_edges = IndexedSet()

        # Define socket properties
        self._circle_connected_brush = QColor(255, 130, 0, 255) if self._socket.type() == NodeSocket.INPUT else QColor(130, 255, 0, 255)
        self._circle_disconnected_brush = QColor(102, 50, 0, 255) if self._socket.type() == NodeSocket.INPUT else QColor(50, 102, 0, 255)
        self._circle_hover_brush = QColor(170, 130, 0, 255) if self._socket.type() == NodeSocket.INPUT else QColor(130, 170, 0, 255)
        self._border_connected_brush = QPen(QColor(255, 255, 255, 255))
        self._border_disconnected_brush = QPen(Qt.black)
        self._circle_brush = self._circle_disconnected_brush
        self._border_brush = self._border_disconnected_brush
        self._bbox = QRectF(0, 0, 10, 10)
        self._moving_edge = False
        self._current_edge = None
        self._layout = QGraphicsLinearLayout(Qt.Horizontal)

        self._init_socket()

    def _init_socket(self):
        self.setLayout(self._layout)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setAcceptHoverEvents(True)

    def type(self) -> SocketType:
        return self._socket.type()

    def value(self):
        return self._socket.value()

    def set_value(self, value: typing.Any):
        self._socket.set_value(value)

    def save_value(self):
        """Saves the value of this socket internally. This value can be reassigned by calling 'restore_value()'."""

        self._socket.save_value()

    def restore_value(self):
        """Restores the value of this socket to the last saved value."""

        self._socket.restore_value()

    def parent_node(self) -> 'GShaderNode':
        return self._parent_g_node

    def get_size(self):
        """Returns a tuple with the width,height of this socket."""
        return self._bbox.right(), self._bbox.bottom()

    def get_backend_socket(self) -> NodeSocket:
        return self._socket

    def get_connected_nodes(self) -> typing.List['GShaderNode']:
        return [n.get_container() for n in self._socket.get_connected_nodes()]

    def get_connected_sockets(self) -> typing.List['GNodeSocket']:
        return [s.get_container() for s in self._socket.get_connected_sockets()]

    def set_index(self, index: int):
        self._socket.set_index(index)

    def get_index(self) -> int:
        return self._socket.get_index()

    def is_connected(self) -> bool:
        return self._socket.is_connected()

    def boundingRect(self):
        return self._bbox

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(self._border_brush)
        painter.setBrush(self._circle_brush)
        painter.drawEllipse(self._bbox)

    def get_scene_pos(self) -> QPointF:
        """Gets the center position of this socket in scene coordinates."""
        pos = self.scenePos()
        pos.setX(pos.x() + self._bbox.right() / 2)
        pos.setY(pos.y() + self._bbox.bottom() / 2)
        return pos

    def connect_to(self, socket: 'GNodeSocket') -> GEdge:
        """
        Connects this GNodeSocket to another GNodeSocket.
        :param other_socket: Other GNodeSocket to connect this socket to.
        :return: the GEdge that was created between the sockets, or the old GEdge if there already exists a connection.
        """
        edge = self._socket.connect_to(socket.get_backend_socket())

        # Only emit change event for input sockets, as nothing really changed for the output socket (at least not for the node as a whole)
        if self.type() == SocketType.INPUT:
            self.connection_changed.emit(self, edge)
        elif socket.type() == SocketType.INPUT:
            socket.connection_changed.emit(socket, edge)

        return GEdge.from_edge(edge)

    def get_connected_edges(self) -> IndexedSet:
        return self._connected_g_edges

    def add_connecting_edge(self, edge: GEdge):
        self._connected_g_edges.add(edge)

    def remove_connected_edge(self, gedge: GEdge):
        assert gedge in self._connected_g_edges

        self._connected_g_edges.remove(gedge)

    def label(self) -> str:
        return self._socket.label()

    def __eq__(self, other):
        if isinstance(other, GNodeSocket):
            return self._socket.__eq__(other.get_backend_socket())

        return False

    def __hash__(self):
        return self._socket.__hash__()

    def __str__(self):
        return self._socket.__str__()

    # -------- Event Handling ---------
    # ----------------------------------
    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        #self._circle_brush = self._circle_hover_brush
        self._border_brush = self._border_connected_brush
        self.update()
        event.accept()

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        #self._circle_brush = self._circle_connected_brush if self.connected else self._circle_disconnected_brush
        if not self.is_connected():
            self._border_brush = self._border_disconnected_brush
            self.update()

        event.accept()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        # Needs to be reimplemented to be able to catch mouseMoveEvent, but it does not need to do anything
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() == Qt.LeftButton:
            if not self._moving_edge:
                if self.type() == self.INPUT:
                    edge = GEdge(destination=self)
                else:
                    edge = GEdge(source=self)
                self._current_edge = edge
                self._moving_edge = True
                self.edge_started.emit(edge)
            else:
                self._current_edge.set_tip_pos(event.scenePos())

            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self._current_edge:  # Prevent release without dragging
            self.edge_released.emit(self._current_edge)
            self._current_edge = None
            self._moving_edge = False

    def itemChange(self, change, value):
        if change == self.ItemScenePositionHasChanged:

            for edge in self._connected_g_edges:
                edge.update_edge()

        return super().itemChange(change, value)
