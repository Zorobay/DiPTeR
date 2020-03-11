from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainterPath, QBrush, QFont, QColor, QPalette
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsWidget, QGridLayout, QGraphicsGridLayout, QTextEdit, QLabel, \
    QLineEdit, QSpacerItem, QSizePolicy

TYPE_FLOAT = "type_float"
TYPE_RGB = "type_rgb"


class Node(QGraphicsWidget):
    def __init__(self, scene, title: str = "", parent=None):
        super().__init__(parent)

        # define Node properties
        self._scene = scene
        self._width = 140
        self._height = 200
        self._rounding = 5
        self._padding = 8
        self._bg_color = QColor(80, 80, 100, 200)
        self._title_color = Qt.white
        self._title_font = QFont("Corbel", 8)
        self._title_font.setBold(True)
        self._title = title
        self._title_item = None

        # Define socket properties
        self._socket_label_font = QFont("Corbel", 7)
        self._socket_label_palette = QPalette()
        self._socket_label_palette.setColor(QPalette.Background, QColor(0,0,0,0))
        self._socket_label_palette.setColor(QPalette.Foreground, self._title_color)

        # Define layout
        self._layout = QGraphicsGridLayout()
        self._layout.setContentsMargins(*[self._padding]*4)
        self._layout.setHorizontalSpacing(2.0)
        self._layout.setRowSpacing(0, 16)
        self.setLayout(self._layout)

        self.add_input(TYPE_FLOAT, "Factor", (0,2.5))

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # Initialize title
        self._create_title()
        self.title = self._title

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self._title_item.setPlainText(self._title)

    def _create_title(self):
        self._title_item = QGraphicsTextItem(self)
        self._title_item.setDefaultTextColor(self._title_color)
        self._title_item.setFont(self._title_font)
        self._title_item.setPos(self._padding, 0)
        self._title_item.setTextWidth(self._width - self._padding)

    def boundingRect(self) -> QtCore.QRectF:
        return QRectF(0, 0, self._width, self._height).normalized()

    def paint(self, painter, option, widget=None):
        path_bg = QPainterPath()
        path_bg.addRoundedRect(0, 0, self._width, self._height, self._rounding, 1)
        painter.setPen(Qt.NoPen)  # Disables the border
        painter.setBrush(QBrush(self._bg_color))

        painter.drawPath(path_bg)

    def add_input(self, input_type: str, input_name: str, input_range=(0, 1)):
        if input_type == TYPE_FLOAT:
            label_widget = QLabel(input_name)
            labe_item = self._scene.addWidget(label_widget)
            labe_item.setPalette(self._socket_label_palette)

            line_edit_widget = QLineEdit()
            line_edit_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            line_edit_widget.setMaximumWidth(self._width - label_widget.width() - 2*self._padding)
            line_edit_widget.setValidator(QtGui.QDoubleValidator(bottom=input_range[0], top=input_range[1]))
            line_edit_item = self._scene.addWidget(line_edit_widget)

            self._layout.addItem(labe_item, 1, 0)
            self._layout.addItem(line_edit_item, 1, 1)
