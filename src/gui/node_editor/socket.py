from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QPainter
from PyQt5.QtWidgets import QGraphicsWidget, QGraphicsLinearLayout, QGraphicsItem


class Socket(QGraphicsWidget):
    edge_started = pyqtSignal(object)

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
        self._current_edge = None
        self._layout = QGraphicsLinearLayout(Qt.Horizontal)

        self.setLayout(self._layout)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)

    def get_width_height(self):
        """Returns a tuple with the width,height of this socket."""
        return self._bbox[2], self._bbox[3]

    def boundingRect(self):
        return self._bbox

    def paint(self, painter:QPainter, option, widget=None):
        painter.setPen(self._border_brush)
        painter.setBrush(self._circle_brush)
        painter.drawEllipse(self._bbox)

    def hoverEnterEvent(self, hover_event):
        print("hover")
        self._circle_brush = self._circle_hover_brush
        self._border_brush = self._border_inactive_brush

    def mousePressEvent(self, mouse_event):
        # Needs to be reimplemented to be able to catch mouseMoveEvent, but it does not need to do anything
        pass

    def mouseMoveEvent(self, mouse_event):
        if mouse_event.buttons() == Qt.LeftButton:
            self.edge_started.emit(self)
