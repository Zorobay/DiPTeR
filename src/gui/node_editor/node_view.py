import typing

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QWheelEvent, QMouseEvent
from PyQt5.QtWidgets import QGraphicsView, QMenu, QGraphicsSceneMouseEvent

from src.gui.node_editor.material import MaterialSelector
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.output_node import OutputNode
from src.misc import string_funcs, array_funcs
from src.shaders.brick_shader import BrickShader
from src.shaders.color_shader import ColorShader
from src.shaders.shader_super import Shader
from tests.stuff_for_testing.shaders.test_lines_shader import TestLinesShader

SHADERS_TO_CONTEXT_MENU = [BrickShader, ColorShader, TestLinesShader]


class NodeView(QGraphicsView):

    def __init__(self, material_selector: MaterialSelector, scene: NodeScene):
        super().__init__(scene)

        # define NodeView properties
        self._pan_last_pos = None

        self._material_selector = material_selector
        self._node_scene = scene
        self._add_node_menu = AddNodeMenu()

        self._init_view()

    def _init_view(self):
        self.setDragMode(self.NoDrag)
        self._node_scene.addItem(OutputNode(self._node_scene))

    def contextMenuEvent(self, cm_event):
        pos = cm_event.pos()
        self._spawn_add_node_menu(self.mapToGlobal(pos))

    def keyPressEvent(self, key_event):
        key = key_event.key()
        modifiers = key_event.modifiers()

        if modifiers == Qt.ControlModifier and key == Qt.Key_A:
            mouse_pos = QtGui.QCursor().pos()
            self._spawn_add_node_menu(mouse_pos)
        else:
            super().keyPressEvent(key_event)

    def _spawn_add_node_menu(self, pos: QPoint):
        shader: typing.Type[Shader] = self._add_node_menu.execute(pos)
        if shader is not None:
            self._material_selector.add_node(shader)

    def wheelEvent(self, wheel_event: QWheelEvent):
        scroll_steps = wheel_event.angleDelta().y() / 8 / 15  # Get actual number of steps (default is 15 deg/step)

        if scroll_steps > 0:
            for sc in range(int(scroll_steps)):
                self.scale(0.9, 0.9)
        elif scroll_steps < 0:
            for sc in range(int(abs(scroll_steps))):
                self.scale(1.1, 1.1)

    def mousePressEvent(self, event: QMouseEvent):
        # Handle start of panning
        if event.button() == Qt.MiddleButton:
            self._pan_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        # Handle end of panning
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() == Qt.MiddleButton:
            delta: QPointF = event.pos() - self._pan_last_pos
            new_x = self.horizontalScrollBar().value() - delta.x()
            new_y = self.verticalScrollBar().value() - delta.y()
            self.horizontalScrollBar().setValue(new_x)
            self.verticalScrollBar().setValue(new_y)
            self._pan_last_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)


class AddNodeMenu(QMenu):

    def __init__(self, *args):
        super().__init__(*args)

        self._action_shader_map = []
        self._add_shader_actions()

    def _add_shader_actions(self):
        for shader in SHADERS_TO_CONTEXT_MENU:
            words = string_funcs.split_on_upper_case(shader.__name__)
            action = self.addAction(" ".join(words))
            self._action_shader_map.append((action, shader))

    def execute(self, *args) -> typing.Type[Shader]:
        action = self.exec_(*args)
        if action is None:
            return None

        action_index = array_funcs.index_of(self._action_shader_map, lambda t: t[0] == action)
        return self._action_shader_map[action_index][1]
