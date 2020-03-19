import typing

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QGraphicsView, QMenu

from src.gui.node_editor.material import Material
from src.misc import string_funcs, array_funcs
from src.shaders.brick_shader import BrickShader
from src.shaders.color_shader import ColorShader
from src.shaders.shader_super import Shader
from tests.stuff_for_testing.test_shaders.test_lines_shader import TestLinesShader

SHADERS_TO_CONTEXT_MENU = [BrickShader, ColorShader, TestLinesShader]


class NodeView(QGraphicsView):

    def __init__(self, material:Material, *args):
        super().__init__(*args)

        self._material = material
        self._node_scene = self.scene()
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
        else:
            super().keyPressEvent(key_event)

    def _spawn_add_node_menu(self, pos: QPoint):
        shader: typing.Type[Shader] = self._add_node_menu.execute(pos)
        if shader is not None:
            self._material.new_node(shader)


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
