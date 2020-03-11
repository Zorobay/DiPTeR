from PyQt5.QtWidgets import QWidget, QGridLayout, QComboBox, QLabel
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

    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = QGridLayout()
        self._object_list = None

        self.setupUI()

    def setupUI(self):
        self._object_list = QComboBox()
        self._object_list.addItems(vertex_func_name_to_label(find_module_func_names(object_vertices)))
        self._layout.addWidget(QLabel("Scene object:"),0,0)
        self._layout.addWidget(self._object_list,0,1)

        self.setLayout(self._layout)
        self.show()
