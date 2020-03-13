import inspect

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QGraphicsView, QMenu

import src.shaders as shader_module
from src.gui.widgets.graphics.node import Node
from src.misc import string_funcs, array_funcs
from src.shaders.shader_super import Shader


class NodeView(QGraphicsView):

    def __init__(self, *args):
        super().__init__(*args)

        self._add_node_menu = AddNodeMenu()

    def contextMenuEvent(self, cm_event):
        pos = cm_event.pos()
        self._spawn_add_node_menu(self.mapToGlobal(pos))

    def keyPressEvent(self, key_event):
        key = key_event.key()
        modifiers = key_event.modifiers()

        if modifiers == Qt.ControlModifier and key == Qt.Key_A:
            mouse_pos = QtGui.QCursor().pos()
            self._spawn_add_node_menu(mouse_pos)

    def _spawn_add_node_menu(self, pos: QPoint):
        shader = self._add_node_menu.execute(pos)
        shader = shader()
        self.scene().addItem(Node.from_shader(self.scene(), shader))


class AddNodeMenu(QMenu):

    def __init__(self, *args):
        super().__init__(*args)

        self._action_shader_map = []
        self._add_shader_actions()

    def _add_shader_actions(self):
        for name, obj in inspect.getmembers(shader_module):
            if inspect.ismodule(obj):
                for name, cls in inspect.getmembers(obj):
                    if inspect.isclass(cls) and issubclass(cls, Shader) and not cls == Shader:
                        words = string_funcs.split_on_upper_case(name)
                        action = self.addAction(" ".join(words))
                        self._action_shader_map.append((action, cls))

    def execute(self, *args) -> Shader:
        action = self.exec_(*args)
        action_index = array_funcs.index_of(self._action_shader_map, lambda t: t[0] == action)
        return self._action_shader_map[action_index][1]
