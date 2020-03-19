import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.shaders.brick_shader import BrickShader


class RenderingArea(QWidget):

    def __init__(self, *args):
        super().__init__(*args)

        self._figure = Figure(figsize=(6,6))
        self._axis = self._figure.add_subplot(111)
        self._figure_canvas = FigureCanvas(self._figure)
        self._layout = QVBoxLayout()

        self._init_widget()

    def _init_widget(self):
        # Test generation of data
        W, H = 100, 100
        sh = BrickShader()
        def_vals = [t[-1] for t in sh.get_inputs()]
        img = np.zeros((W, H, 3))
        for y in range(H):
            for x in range(W):
                vert_pos = np.array((x / H, y / H, 0.0))
                val = sh.shade(vert_pos, *def_vals)
                img[y, x, :] = val[:3]

        self._layout.addWidget(self._figure_canvas)
        self._axis.imshow(img, vmin=0., vmax=1.)
        self._figure.gca().invert_yaxis()
        self._axis.set_ylim(0, H)
        self._axis.set_xlim(0, W)

        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.setLayout(self._layout)
