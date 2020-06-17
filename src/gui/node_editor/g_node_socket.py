import logging
import uuid

import typing
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QColor, QBrush, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLinearLayout, QGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent
from node_graph.data_type import DataType
from node_graph.node_socket import NodeSocket, SocketType

from src.gui.node_editor.g_edge import GEdge

_logger = logging.getLogger(__name__)


class GNodeSocket(QGraphicsWidget):
    edge_started = pyqtSignal(GEdge)
    edge_released = pyqtSignal(GEdge)
    connection_changed = pyqtSignal(object, GEdge)  # Socket, Edge
    position_changed = pyqtSignal(QPointF)

    INPUT = SocketType.INPUT
    OUTPUT = SocketType.OUTPUT

    def __init__(self, parent_node: 'GShaderNode', socket: NodeSocket):
        super().__init__(parent=parent_node)
        self._socket = socket
        self._parent_g_node = parent_node
        self._connected_g_edges = set()

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
        self._connected = False
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

    def parent_node(self) -> 'GShaderNode':
        return self._parent_g_node

    def get_size(self):
        """Returns a tuple with the width,height of this socket."""
        return self._bbox.right(), self._bbox.bottom()

    def get_connected_nodes(self):

    def get_backend_socket(self) -> NodeSocket:
        return self._socket

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

    def add_connected_edge(self, edge: GEdge):
        self._connected_g_edges.add(edge)
        self._socket.connect_to()

        # Only emit change event for input sockets, as nothing really changed for the output socket (at least not for the node as a whole)
        if self.type() == SocketType.INPUT:
            self.connection_changed.emit(self, edge)

    def remove_connected_edge(self, edge: GEdge):
        assert edge in self._connected_edges

        self._connected_edges.remove(edge)
        if edge.out_socket == self:
            self._connected_sockets.remove(edge.in_socket)
        else:
            self._connected_sockets.remove(edge.out_socket)

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
                edge = GEdge()
                if self.type() == self.INPUT:
                    edge.set_destination_socket(self)
                else:
                    edge.set_source_socket(self)
                self._current_edge = edge
                self._moving_edge = True
                self.edge_started.emit(edge)
            else:
                self._current_edge.update_edge()

            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        self.edge_released.emit(self._current_edge)

        if self._current_edge.out_socket and self._current_edge.in_socket:
            self.add_connected_edge(self._current_edge)

        self._current_edge = None
        self._moving_edge = False

    def itemChange(self, change, value):
        if change == self.ItemScenePositionHasChanged:

            for edge in self._connected_g_edges:
                if self.type() == SocketType.INPUT:
                    edge.dest_pos = self.get_scene_pos()
                else:
                    edge.src_pos = self.get_scene_pos()

        return super().itemChange(change, value)
