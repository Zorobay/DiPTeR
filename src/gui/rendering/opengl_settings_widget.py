import types

import autograd.numpy as anp
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

from src.gui.node_editor.control_center import Material, ControlCenter
from src.gui.node_editor.texture_matcher import TextureMatcher
from src.gui.rendering.opengl_widget import OpenGLWidget
from src.opengl import object_vertices
from src.shaders.brick_shader import BrickShader


def find_module_func_names(module):
    func_names = []
    for a in dir(module):
        attr = getattr(module, a)
        if isinstance(attr, types.FunctionType):
            func_names.append(a)

    return func_names


def vertex_func_name_to_label(vertex_func_names: list):
    """Processes a list of primary_function names from the object_vertices module into formatted labels."""
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
            shader = mat.shader
            if shader:
                self._matcher = TextureMatcher(shader)
                self._matcher.show()

    # def _load_texture(self):
    #     filename, _ = QFileDialog.getOpenFileName(self, "Open Texture", filter="Image Files (*.png *.jpg)")
    #     if filename:
    #         self._texture_label.setText(Path(filename).stem)
    #         self._loaded_texture = Image.open(filename)
    #
    # def _start_gradient_descent(self):
    #     if self._loaded_texture:
    #         W, H = 50, 50
    #         truth = np.asarray(self._loaded_texture.resize((W, H)), dtype=np.float32) / 255.
    #         shader = self._open_gl_widget.material._shader
    #         f = shader.shade
    #         loss_grad = grad(loss, list(range(1 + len(shader.get_parameters_list()))))
    #
    #         max_iter = 100
    #         early_stopping_thresh = 0.02
    #         lr_decay = 0.99
    #         lr = 0.1
    #         params = shader.get_parameters_list()
    #         loss_hist = []
    #
    #         for i in range(max_iter):
    #             _, gradient = loss_grad(truth, *params)
    #             params -= lr * np.array(gradient)
    #             new_loss = loss(truth, *params)
    #             loss_hist.append(new_loss)
    #             shader.set_input_by_uniform("color", params[0])
    #
    #             print("{}. new loss: {:.5f}, lr: {:.5f}, Params: {}".format(i, new_loss, lr, params))
    #             if new_loss <= early_stopping_thresh:
    #                 break
    #             lr = lr * lr_decay


def loss(truth: np.ndarray, *args) -> float:
    width = truth.shape[0]
    height = truth.shape[1]
    x_res = 1.0 / width
    y_res = 1.0 / height
    x_pos = anp.linspace(0, 1.0, width, endpoint=False) + (x_res / 2.0)
    y_pos = anp.linspace(0, 1.0, height, endpoint=False) + (y_res / 2.0)
    loss_sum = 0
    sh = BrickShader()

    for x in range(width):
        for y in range(height):
            vert_pos = anp.array((x_pos[x], y_pos[y], 0.))
            val = sh.shade(vert_pos, *args)
            loss_sum += anp.sum(anp.abs(truth[y, x, :] - val))

    return loss_sum / (width * height * 4)
