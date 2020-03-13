from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QGraphicsScene


class NodeScene(QGraphicsScene):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initScene()

    def initScene(self):
        self.setBackgroundBrush(QBrush(QColor(140, 140, 140, 255)))
