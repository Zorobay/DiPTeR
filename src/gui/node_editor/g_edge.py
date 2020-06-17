import uuid

from PyQt5.QtCore import QPoint, QRectF, Qt, QPointF
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLineItem, QGraphicsItem
from node_graph.edge import Edge


class GEdge(QGraphicsLineItem):

    def __init__(self, start_pos):
        super().__init__()

        self._id = uuid.uuid4()
        self.out_socket: 'Socket' = None
        self.in_socket: 'Socket' = None
        self._out_pos = start_pos
        self._in_pos = start_pos

        self._edge_pen = QPen(QColor(200, 130, 0, 255), 3)
        self._init_item()

    def _init_item(self):
        self.setPen(self._edge_pen)

    @property
    def id(self):
         return self._id

    @property
    def uuid(self):
        return self._id

    @property
    def out_pos(self):
        return self._out_pos

    @out_pos.setter
    def out_pos(self, value: QPoint):
        self._out_pos = value
        self._update_line()

    @property
    def in_pos(self):
        return self._in_pos

    @in_pos.setter
    def in_pos(self, value: QPoint):
        self._in_pos = value
        self._update_line()

    def _update_line(self):
        self.setLine(self.out_pos.x(), self.out_pos.y(), self.in_pos.x(), self.in_pos.y())

