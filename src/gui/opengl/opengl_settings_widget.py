from PyQt5.QtWidgets import QWidget, QGridLayout, QComboBox, QLabel

from src.gui.opengl.opengl_widget import OpenGLWidget
from src.opengl import object_vertices
import types


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

    def __init__(self, open_gl_widget: OpenGLWidget, parent=None):
        super().__init__(parent)
        self._open_gl_widget = open_gl_widget

        # Define gui component variables
        self._layout = QGridLayout()
        self._object_list = None

        # Define data variables
        self._object_vertices_func_names = find_module_func_names(object_vertices)

        self._init_widget()
        self._send_defaults_to_gl()

    def _init_widget(self):
        self._object_list = QComboBox()
        self._object_list.addItems(vertex_func_name_to_label(self._object_vertices_func_names))
        self._object_list.currentIndexChanged.connect(self._handle_object_vertices_list_change)
        self._layout.addWidget(QLabel("Scene object:"), 0, 0)
        self._layout.addWidget(self._object_list, 0, 1)

        self.setLayout(self._layout)
        self.show()

    def _send_defaults_to_gl(self):
        self._handle_object_vertices_list_change(self.OBJECT_VERTICES_DEFAULT)

    def _handle_object_vertices_list_change(self, index):
        func_name = self._object_vertices_func_names[index]
        func = getattr(object_vertices, func_name)
        V, I = func()
        self._open_gl_widget.set_vertices(V,I)
