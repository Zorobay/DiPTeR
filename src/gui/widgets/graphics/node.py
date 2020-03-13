import typing

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainterPath, QBrush, QFont, QColor, QPalette, QPainter
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsWidget, QSizePolicy, QGraphicsLinearLayout

from src.gui.widgets.graphics.graphics_grid_layout import GraphicsGridLayout
from src.gui.widgets.graphics.node_scene import NodeScene
from src.gui.widgets.graphics.socket import Socket
from src.gui.widgets.input.double_line_edit import DoubleLineEdit
from src.gui.widgets.input.input_module import InputModule
from src.opengl.shader_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_RGB, Type
from src.shaders.shader_super import Shader


class Node(QGraphicsWidget):
    def __init__(self, scene: NodeScene, title: str = "", parent=None):
        super().__init__(parent)

        # define data properties
        self._input_sockets = []

        # define Node properties
        self._scene = scene
        self._width = 250
        self._height = 200
        self._rounding = 5
        self._padding = 8
        self._bg_color = QColor(80, 80, 100, 200)
        self._title_color = Qt.white
        self._title_font = QFont("Corbel", 11)
        self._title_font.setBold(True)
        self._title = title
        self._title_item = QGraphicsTextItem(self)

        # Define input properties
        self._input_label_font = QFont("Corbel", 8)
        self._input_label_palette = QPalette()
        self._input_label_palette.setColor(QPalette.Background, QColor(0, 0, 0, 0))
        self._input_label_palette.setColor(QPalette.Foreground, self._title_color)

        # Define layout
        self._master_layout = GraphicsGridLayout()
        self._socket_layout = QGraphicsLinearLayout(Qt.Vertical)
        self._input_layout = QGraphicsLinearLayout(Qt.Vertical)

        # Set flags to enable the widget to be moved and selected
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # Initialize the widget
        self._init_title()
        self._init_layout()

    def _init_title(self):
        self._title_item.setDefaultTextColor(self._title_color)
        self._title_item.setFont(self._title_font)
        self._title_item.setPos(self._padding, 0)
        self._title_item.setTextWidth(self._width - self._padding)
        self.title = self._title

    def _init_layout(self):
        self._master_layout.setContentsMargins(-5, 4, 4, 4)
        self._master_layout.setRowSpacing(0, self._title_font.pointSize() + 10)  # Add empty space for first row so that title is visible
        self._master_layout.setHorizontalSpacing(2.0)

        self._socket_layout.setSpacing(10.0)
        self._input_layout.setSpacing(10.0)

        self._master_layout.addItem(self._socket_layout, 1, 0)
        self._master_layout.addItem(self._input_layout, 1, 1)
        self.setLayout(self._master_layout)

    @classmethod
    def from_shader(cls, scene: NodeScene, shader: Shader):
        node = cls(scene, shader.get_name())
        for p in shader.get_parameters():
            node.add_input_from_uniform(p)
        return node

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self._title_item.setPlainText(self._title)


    def boundingRect(self) -> QtCore.QRectF:
        return QRectF(0, 0, self._width, self._height).normalized()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(Qt.NoPen)  # Disables the border
        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(0,0,self._width, self._height, self._rounding, 1)

    def add_input_from_uniform(self, uniform: typing.Tuple[str, Type]):
        name = uniform[0]
        type_ = uniform[1]
        self.add_input(type_, name)

    def add_input(self, type_: Type, input_name: str, input_range=(0, 1)):
        # Create a socket
        socket = Socket()
        self._socket_layout.addItem(Socket())

        if type_.internal_type == INTERNAL_TYPE_FLOAT:
            # Create an input widget
            line_edit_widget = DoubleLineEdit(min_=input_range[0], max_=input_range[1])
            line_edit_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

            module = InputModule(input_name, line_edit_widget)
            module.set_label_palette(self._input_label_palette)
            module_item = self._scene.addWidget(module)

            # Add all components to the layout
            self._input_layout.addItem(module_item)

        if type_.internal_type == INTERNAL_TYPE_RGB:
            pass
