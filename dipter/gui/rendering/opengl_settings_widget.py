import types

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from dipter.gui.texture_matching.texture_matcher import TextureMatcher
from dipter.gui.node_editor.control_center import ControlCenter
from dipter.gui.rendering.opengl_widget import OpenGLWidget
from dipter.opengl import object_vertices


def find_module_func_names(module):
    func_names = []
    for a in dir(module):
        attr = getattr(module, a)
        if isinstance(attr, types.FunctionType):
            func_names.append(a)

    return func_names


def vertex_func_name_to_label(vertex_func_names: list):
    """Processes a list of function names from the object_vertices module into formatted labels."""
    labels = []
    for f in vertex_func_names:
        if f.startswith("get_"):
            f = f[4:]

        split = f.split("_")
        labels.append(" ".join([s[0].upper() + s[1:] for s in split]))

    return labels


class OpenGLSettingsWidget(QWidget):
    # Define defaults
    OBJECT_VERTICES_DEFAULT = 0

    def __init__(self, cc: ControlCenter, open_gl_widget: OpenGLWidget, parent=None):
        super().__init__(parent)

        self.cc = cc
        self._open_gl_widget = open_gl_widget

        # Define gui component variables
        self._layout = QVBoxLayout()
        self._object_list = QComboBox()
        self._reset_view_button = QPushButton("Reset Scene View")
        self._match_texture_button = QPushButton("Match Texture")
        self._randomize_button = QPushButton("Randomize Parameters")
        self._matcher = None

        # Define data variables
        self._object_vertices_func_names = find_module_func_names(object_vertices)
        self._loaded_texture = None

        self._init_widget()
        self._send_defaults_to_gl()

    def _init_widget(self):
        # Setup 3d object chooser
        self._object_list.addItems(vertex_func_name_to_label(self._object_vertices_func_names))
        self._object_list.currentIndexChanged.connect(self._handle_object_vertices_list_change)
        self._object_list.setCurrentIndex(0)

        scene_object_layout = QHBoxLayout()
        scene_object_layout.addWidget(QLabel("Scene object:"))
        scene_object_layout.addWidget(self._object_list)
        self._layout.addLayout(scene_object_layout)

        # Setup reset scene view button
        self._reset_view_button.clicked.connect(self._reset_view)
        self._layout.addWidget(self._reset_view_button)

        # Setup Match Texture Button
        self._match_texture_button.clicked.connect(self._match_texture)
        self._layout.addWidget(self._match_texture_button)

        self._randomize_button.clicked.connect(self._randomize_shader_model)
        self._layout.addWidget(self._randomize_button)

        self._layout.setAlignment(Qt.AlignTop)
        self.setLayout(self._layout)
        self.show()

    def _reset_view(self):
        self._open_gl_widget.reset_view()

    def _send_defaults_to_gl(self):
        self._handle_object_vertices_list_change(self.OBJECT_VERTICES_DEFAULT)

    def _handle_object_vertices_list_change(self, index):
        func_name = self._object_vertices_func_names[index]
        func = getattr(object_vertices, func_name)
        V, I = func()
        self._open_gl_widget.set_vertices(V, I)

    def _match_texture(self):
        mat = self.cc.active_material
        if mat:
            out_node = mat.get_material_output_node()
            if out_node and out_node.can_render():  # Check that the material output node exists
                self._matcher = TextureMatcher(self.cc, out_node)
                self._matcher.show()
            else:
                # Not ready to render, show dialog to user...
                dialog = QMessageBox(QMessageBox.Warning, "Warning",
                                     "Can not start Texture Matcher with only a Material Ouput Node. "
                                     "Please connect the output node to a shader and try again.", buttons=QMessageBox.Ok, parent=self)
                dialog.exec()

    def _randomize_shader_model(self):
        for node in self.cc.active_material.get_nodes().values():
            node.randomize_input()
