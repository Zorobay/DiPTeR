import logging
import uuid

from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QColor, QBrush, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLinearLayout, QGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent

from src.gui.node_editor.edge import Edge

_logger = logging.getLogger(__name__)


class Socket(QGraphicsWidget):
    edge_started = pyqtSignal(Edge)
    edge_released = pyqtSignal(Edge)
    position_changed = pyqtSignal(QPointF)

    SOCKET_INPUT = 0
    SOCKET_OUTPUT = 1

    def __init__(self, parent_node, socket_type: int):
        super().__init__(parent_node)
        self._parent_node = parent_node
        self._socket_type = socket_type
        self._id = uuid.uuid4()
        self._connected_edges = set()

        # Define socket properties
        self._circle_connected_brush = QColor(255, 130, 0, 255) if socket_type == self.SOCKET_INPUT else QColor(130, 255, 0, 255)
        self._circle_disconnected_brush = QColor(102, 50, 0, 255) if socket_type == self.SOCKET_INPUT else QColor(50, 102, 0, 255)
        self._circle_hover_brush = QColor(170, 130, 0, 255) if socket_type == self.SOCKET_INPUT else QColor(130, 170, 0, 255)
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

    def __eq__(self, other):
        if isinstance(other, Socket):
            return other.id == self.id

        return False

    @property
    def id(self):
        return self._id

    @property
    def socket_type(self):
        return self._socket_type

    @property
    def connected(self):
        return len(self._connected_edges) > 0

    def get_size(self):
        """Returns a tuple with the width,height of this socket."""
        return self._bbox.right(), self._bbox.bottom()

    def boundingRect(self):
        return self._bbox

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(self._border_brush)
        painter.setBrush(self._circle_brush)
        painter.drawEllipse(self._bbox)

    def get_socket_scene_center(self) -> QPointF:
        pos = self.scenePos()
        pos.setX(pos.x() + self._bbox.right() / 2)
        pos.setY(pos.ty() + self._bbox.bottom() / 2)
        return pos

    def add_connected_edge(self, edge: Edge):
        self._connected_edges.add(edge)
        _logger.debug("Connected edge ({}) to socket ({}) of node {}".format(edge.id, self.id, self._parent_node.title))

    # -------- Event Handling ---------
    # ----------------------------------
    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        #self._circle_brush = self._circle_hover_brush
        self._border_brush = self._border_connected_brush
        self.update()
        event.accept()

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        #self._circle_brush = self._circle_connected_brush if self.connected else self._circle_disconnected_brush
        if not self.connected:
            self._border_brush = self._border_disconnected_brush
            self.update()

        event.accept()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        # Needs to be reimplemented to be able to catch mouseMoveEvent, but it does not need to do anything
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() == Qt.LeftButton:
            if not self._moving_edge:
                start_pos: QPointF = self.get_socket_scene_center()
                edge = Edge(start_pos)
                if self.socket_type == self.SOCKET_INPUT:
                    edge.in_socket_id = self.id
                else:
                    edge.out_socket_id = self.id
                self._current_edge = edge
                self._moving_edge = True
                self.edge_started.emit(edge)
            else:
                if self.socket_type == self.SOCKET_INPUT:
                    self._current_edge.out_pos = event.scenePos()
                else:
                    self._current_edge.in_pos = event.scenePos()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        self.edge_released.emit(self._current_edge)

        if self._current_edge.out_socket_id and self._current_edge.in_socket_id:
            self.add_connected_edge(self._current_edge)

        self._current_edge = None
        self._moving_edge = False

    def itemChange(self, change, value):
        if change == self.ItemScenePositionHasChanged:

            for edge in self._connected_edges:
                if self._socket_type == self.SOCKET_INPUT:
                    edge.in_pos = self.get_socket_scene_center()
                else:
                    print("output socket moved")
                    edge.out_pos = self.get_socket_scene_center()

        return super().itemChange(change, value)
