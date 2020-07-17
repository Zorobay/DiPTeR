import logging

from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QGraphicsScene

_logger = logging.getLogger(__name__)


class NodeScene(QGraphicsScene):
    """Class that keeps track of all nodes present on the scene, as well as their relation to each other."""

    def __init__(self):
        super().__init__()

        self.is_drawing_edge = False
        self._current_edge = None

        self._init_scene()

    def _init_scene(self):
        self.setBackgroundBrush(QBrush(QColor(140, 140, 140, 255)))

    # def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
    #     if event.buttons() == Qt.LeftButton:
    #         print("Dragging on Scene")
    #         event.accept()
    #
    # def mousePressEvent(self, QMouseEvent):
    #     super().mousePressEvent(QMouseEvent)
    #
    # def mouseMoveEvent(self, mouse_event):
    #     if mouse_event.buttons() == Qt.LeftButton:
    #         if not self._current_edge:
    #             self._current_edge = Edge()
    #             self._current_edge.start_pos = mouse_event.scenePos()
    #             self.addItem(self._current_edge)
    #         if self._current_edge:
    #             self._current_edge.end_pos = mouse_event.scenePos()
    #
    # def mouseReleaseEvent(self, mouse_event):
    #     if self._current_edge:
    #         self.removeItem(self._current_edge)
    #         self._current_edge = None
