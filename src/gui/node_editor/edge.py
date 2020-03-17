import uuid

from PyQt5.QtCore import QPoint, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsWidget


class Edge(QGraphicsWidget):

    def __init__(self, *args):
        super().__init__(*args)

        self._id = uuid.uuid4()
        self._start_pos = None
        self._end_pos = None

        self._edge_pen = QPen(QColor(200, 130, 0, 255), 2)

    def boundingRect(self):
        return QRectF(QPoint(0, 0), self._end_pos)

    @property
    def uuid(self):
        return self._id

    @property
    def start_pos(self):
        return self._start_pos

    @start_pos.setter
    def start_pos(self, value: QPoint):
        self._start_pos = value

    @property
    def end_pos(self):
        return self._end_pos

    @end_pos.setter
    def end_pos(self, value: QPoint):
        self._end_pos = value
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(self._edge_pen)
        painter.drawLine(QPoint(0, 0), self._end_pos)

        print("start: {} -> end: {}".format(self.start_pos, self.end_pos))
