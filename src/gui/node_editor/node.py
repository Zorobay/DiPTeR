import abc
import typing
import uuid

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QFont, QColor, QPalette, QPainter
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsWidget, QSpacerItem, QLabel
from glumpy.gloo import Program

from src.gui.widgets.color_input import ColorInput
from src.gui.widgets.float_input import FloatInput
from src.gui.widgets.input_module import InputModule
from src.gui.node_editor.layouts import GraphicsGridLayout
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.gui.widgets.output_module import OutputModule
from src.opengl.shader_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_ARRAY_RGB, INTERNAL_TYPE_ARRAY_RGBA
from src.shaders.shader_super import Shader


class NodeGraphics(QGraphicsWidget):
    """This abstract class defines the look and feel of a Node. Specialized classes can subclass this instead of the Node class.
    """
    edge_started = pyqtSignal(uuid.UUID, object)

    @abc.abstractmethod
    def __init__(self, scene: NodeScene, title: str, parent=None):
        super().__init__(parent)

        self._scene = scene
        self._title = title
        self._id = uuid.uuid4()

        # define Node properties
        self._input_index = 2
        self._width = 250
        self._height = 50
        self._rounding = 5
        self._padding = 8
        self._bg_color = QColor(80, 80, 100, 200)
        self._title_color = Qt.white
        self._title_font = QFont("Corbel", 11)
        self._title_font.setBold(True)
        self._title_item = QGraphicsTextItem(self)

        # Define layout
        self._master_layout = GraphicsGridLayout()

        # Define widgets properties
        self._input_label_font = QFont("Corbel", 8)
        self._input_label_palette = QPalette()
        self._input_label_palette.setColor(QPalette.Background, QColor(0, 0, 0, 0))
        self._input_label_palette.setColor(QPalette.Foreground, self._title_color)

        # Set flags to enable the widget to be moved and selected
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

        self._init_title()
        self._init_layout()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self._title_item.setPlainText(self._title)

    @property
    def id(self):
        return self._id

    def _init_layout(self):
        self._master_layout.setContentsMargins(-5, 4, 4, 4)
        self._master_layout.setRowSpacing(0, self._title_font.pointSize() + 12)  # Add empty space for first row so that title is visible
        self._master_layout.setHorizontalSpacing(2.0)
        self._master_layout.setVerticalSpacing(0.0)

        self.setLayout(self._master_layout)

    def _init_title(self):
        self._title_item.setDefaultTextColor(self._title_color)
        self._title_item.setFont(self._title_font)
        self._title_item.setPos(self._padding, 0)
        self._title_item.setTextWidth(self._width - self._padding)
        self.title = self._title

    def boundingRect(self) -> QtCore.QRectF:
        return QRectF(0, 0, self._width, self._height).normalized()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(Qt.NoPen)  # Disables the border
        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(0, 0, self._width, self._height, self._rounding, 1)


class Node(NodeGraphics):
    input_changed = pyqtSignal(uuid.UUID)

    def __init__(self, scene: NodeScene, title: str, shader: Shader, parent=None):
        super().__init__(scene, title, parent)

        # define data properties
        self._input_modules = {}
        self._shader = shader
        self._program = shader.get_program()

        # Initialize the widget
        self._init_widget()

    def _init_widget(self):
        for nf, nu, t, ra, de in self._shader.get_inputs():
            self.add_input(nf, nu, t, ra, de)

        for nf, t in self._shader.get_outputs():
            self.add_output(nf, t)

    def get_program(self) -> Program:
        return self._program

    def _set_input(self, uniform_var: str, value: typing.Any, internal_type: str, input_id: uuid.UUID):
        """Event is called when any of this node's widgets is changed"""
        self._shader.set_input_by_uniform(uniform_var, value)
        self.input_changed.emit(self.id)

    def add_input(self, input_name: str, uniform_var: str, internal_type: str, input_range: (float, float), default_value: typing.Any):
        socket = Socket()
        socket.edge_started.connect(lambda s: self.edge_started.emit(self.id, s))

        if internal_type == INTERNAL_TYPE_FLOAT:
            # Create an widgets widget
            input_widget = FloatInput(internal_type, min_=input_range[0], max_=input_range[1])
        elif internal_type == INTERNAL_TYPE_ARRAY_RGB or INTERNAL_TYPE_ARRAY_RGBA:
            input_widget = ColorInput(internal_type)
        else:
            raise TypeError("Internal type {} is not yet supported!".format(internal_type))

        # Create an widgets module and add to this node
        module = InputModule(input_name, internal_type, uniform_var, input_widget)
        module.input_changed.connect(self._set_input)
        module.set_label_palette(self._input_label_palette)
        module.set_default_value(default_value)

        module_item = self._scene.addWidget(module)
        self._master_layout.addItem(socket, self._input_index, 0)
        self._master_layout.addItem(module_item, self._input_index, 1)
        self._master_layout.setRowAlignment(self._input_index, Qt.AlignBottom)
        self._input_index += 1
        self._height = self._input_index * 40

    def add_output(self, output_name: str, internal_type: str):
        socket = Socket()
        output_module = OutputModule(output_name)
        output_module.set_label_palette(self._input_label_palette)
        module_item = self._scene.addWidget(output_module)

        self._master_layout.addItem(module_item, 1, 1)
        self._master_layout.addItem(socket, 1, 2)
        #self._output_socket_layout.addItem(socket)
