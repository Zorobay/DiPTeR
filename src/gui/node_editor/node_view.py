import typing

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QWheelEvent, QMouseEvent
from PyQt5.QtWidgets import QGraphicsView, QMenu, QGraphicsSceneMouseEvent, QMessageBox

from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.g_edge import GEdge
from src.gui.node_editor.material import Material
from src.misc import string_funcs, array_funcs
from src.shaders.brick_shader import BrickShader
from src.shaders.checker_shader import CheckerShader
from src.shaders.cloud_shader import CloudShader
from src.shaders.color_shader import ColorShader
from src.shaders.frag_coord_shader import FragmentCoordinatesShader
from src.shaders.gradient_shader import GradientShader
from src.shaders.hsv_shader import HSVShader
from src.shaders.mix_shader import MixShader
from src.shaders.rgb_shader import RGBShader
from src.shaders.shader_super import FunctionShader
from src.shaders.tile_shader import TileShader
from src.shaders.vector_math_shader import VectorMathShader
from tests.stuff_for_testing.shaders.test_box_shader import TestBoxShader

SHADERS_TO_CONTEXT_MENU = [TileShader, GradientShader, CloudShader, CheckerShader, HSVShader, ColorShader, RGBShader, BrickShader, MixShader]

SHADERS_MISC_MENU = [FragmentCoordinatesShader, TestBoxShader]
SHADERS_MATH_MENU = [VectorMathShader]


class NodeView(QGraphicsView):

    def __init__(self, cc: ControlCenter):
        super().__init__(None)

        self.cc = cc

        # define NodeView properties
        self._pan_last_pos = None
        self.is_drawing_edge = False
        self._current_edge = None

        self._node_scene = None
        self._add_node_menu = AddNodeMenu()

        self._init_view()

    def _init_view(self):
        self.setDragMode(self.NoDrag)
        self.cc.active_material_changed.connect(self._set_scene)

    def _set_scene(self, material: Material):
        assert material.node_scene == self.cc.active_scene

        self.setScene(self.cc.active_scene)

    def _spawn_add_node_menu(self, pos: QPoint):
        shader: typing.Type[FunctionShader] = self._add_node_menu.execute(pos)
        if shader is not None:
            res = self.cc.new_node(shader)
            if not res:
                msg = QMessageBox(self)
                msg.setWindowTitle("No Material Selected")
                msg.setText("Please select or create a new material before adding nodes.")
                msg.setIcon(QMessageBox.Information)
                msg.exec_()

    def draw_edge(self, edge: GEdge):
        scene = self.cc.active_scene
        if scene:
            scene.addItem(edge)
            self._current_edge = edge
            self.is_drawing_edge = True

    def cancel_edge(self, edge: GEdge):
        scene = self.cc.active_scene
        if edge == self._current_edge and scene:
            scene.removeItem(edge)
            self._current_edge = None
            self.is_drawing_edge = False

    def addWidget(self, *args, **kwargs):
        self.scene().addWidget(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        self.scene().addItem(*args, **kwargs)

    # ------------------------------------
    # ------ Event handling -------------
    # ------------------------------------
    def contextMenuEvent(self, cm_event):
        pos = cm_event.pos()
        self._spawn_add_node_menu(self.mapToGlobal(pos))

    def keyPressEvent(self, key_event):
        key = key_event.key()
        modifiers = key_event.modifiers()

        if modifiers == Qt.ControlModifier and key == Qt.Key_A:
            mouse_pos = QtGui.QCursor().pos()
            self._spawn_add_node_menu(mouse_pos)
        elif key == Qt.Key_Delete:
            mat = self.cc.active_material
            if mat:
                nodes = mat.get_nodes()
                delete = []

                for id_ in nodes:
                    n = nodes[id_]
                    if n.isSelected() and n.isDeletable():
                        delete.append(id_)

                for id_ in delete:
                    self.cc.delete_node(id_)

        else:
            super().keyPressEvent(key_event)

    def wheelEvent(self, wheel_event: QWheelEvent):
        scroll_steps = wheel_event.angleDelta().y() / 8 / 15  # Get actual number of steps (default is 15 deg/step)

        if scroll_steps > 0:
            for sc in range(int(scroll_steps)):
                self.scale(0.9, 0.9)
        elif scroll_steps < 0:
            for sc in range(int(abs(scroll_steps))):
                self.scale(1.1, 1.1)

    def mousePressEvent(self, event: QMouseEvent):
        # Unselect all nodes
        mat = self.cc.active_material
        if mat:
            nodes = mat.get_nodes()
            for n in nodes:
                nodes[n].setSelected(False)

        # Handle start of panning
        if event.button() == Qt.MiddleButton:
            self._pan_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

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

        self._misc_menu = self.addMenu("Misc")
        self._math_menu = self.addMenu("Math")
        self._action_shader_map = []
        self._add_shader_actions()

    def _add_shader_actions(self):
        for shader in SHADERS_TO_CONTEXT_MENU:
            self._add_shader(shader, self)

        for shader in SHADERS_MISC_MENU:
            self._add_shader(shader, self._misc_menu)

        for shader in SHADERS_MATH_MENU:
            self._add_shader(shader, self._math_menu)

    def _add_shader(self, shader, menu: QMenu):
        words = string_funcs.split_on_upper_case(shader.__name__)
        action = menu.addAction(" ".join(words))
        self._action_shader_map.append((action, shader))

    def execute(self, *args) -> typing.Type[FunctionShader]:
        action = self.exec_(*args)
        if action is None:
            return None

        action_index = array_funcs.index_of(self._action_shader_map, lambda t: t[0] == action)
        return self._action_shader_map[action_index][1]
