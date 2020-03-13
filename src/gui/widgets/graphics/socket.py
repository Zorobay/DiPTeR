from PyQt5.QtCore import Qt, QRectF, QPoint
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLinearLayout

from src.gui.widgets.graphics.edge import Edge


class Socket(QGraphicsWidget):

    def __init__(self, *args):
        super().__init__(*args)

        # Define socket properties
        self._circle_active_brush = QColor(255, 130, 0, 255)
        self._circle_inactive_brush = QColor(102, 50, 0, 255)
        self._circle_hover_brush = QColor(170, 130, 0, 255)
        self._border_active_brush = QBrush(QColor(255, 255, 255, 255))
        self._border_inactive_brush = Qt.black
        self._circle_brush = self._circle_inactive_brush
        self._border_brush = self._border_inactive_brush
        self._bbox = QRectF(0, 0, 10, 10)

        self._layout = QGraphicsLinearLayout(Qt.Horizontal)
        self.setLayout(self._layout)
        self._current_edge = None

    def get_width_height(self):
        """Returns a tuple with the width,height of this socket."""
        return self._bbox[2], self._bbox[3]

    def boundingRect(self):
        return self._bbox

    def paint(self, painter, option, widget=None):
        painter.setPen(self._border_brush)
        painter.setBrush(self._circle_brush)
        painter.drawEllipse(self._bbox)

    def hoverEnterEvent(self, hover_event):
        print("hover")
        self._circle_brush = self._circle_hover_brush
        self._border_brush = self._border_inactive_brush

    def mousePressEvent(self, mouse_event):
        # Needs to be reimplemented to be able to catch mouseMoveEvent
        pass
    #     if mouse_event.buttons() == Qt.LeftButton:
    #         pos = mouse_event.scenePos()
    #         edge = Edge()
    #         edge.start_pos = QPoint(self.mapFromScene(pos).toPoint())
    #         self._current_edge = edge

    def mouseMoveEvent(self, mouse_event):
        if mouse_event.buttons() == Qt.LeftButton:
            if not self._current_edge:
                self._current_edge = Edge()
                self._layout.addItem(self._current_edge)

            pos = self.mapFromScene(mouse_event.scenePos())
            self._current_edge.end_pos = QPoint(pos.toPoint())
