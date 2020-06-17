import uuid

import typing
from PyQt5.QtCore import QPoint, QRectF, Qt, QPointF
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLineItem, QGraphicsItem
from node_graph.edge import Edge


class GEdge(QGraphicsLineItem):

    def __init__(self):
        super().__init__()

        self._src_socket = None
        self._dest_socket = None

        self._edge_pen = QPen(QColor(200, 130, 0, 255), 3)
        self._init_item()

    def _init_item(self):
        self.setPen(self._edge_pen)

    def set_source_socket(self, socket: 'GNodeSocket'):
        """Sets the socket that is to be the source of this Edge."""
        self._src_socket = socket
        self.update_edge()

    def set_destination_socket(self, socket: 'GNodeSocket'):
        """Sets the socket that is to be the destination of this Edge."""
        self._dest_socket = socket
        self.update_edge()

    def src_pos(self) -> typing.Union[QPointF, None]:
        if self._src_socket:
            return self._src_socket.get_scene_pos()
        return None

    def dest_pos(self) -> typing.Union[QPointF, None]:
        if self._dest_socket:
            return self._dest_socket.get_scene_pos()
        return None

    def update_edge(self):
        if self.src_pos() and self.dest_pos():
            dest, source = self.src_pos(), self.dest_pos()
            self.setLine(source.x(), source.y(), dest.x(), dest.y())

