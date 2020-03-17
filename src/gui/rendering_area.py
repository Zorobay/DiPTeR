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
        img = np.zeros((W, H, 3))
        for r in range(H):
            for c in range(W):
                vert_pos = np.array((r / H, c / H, 0.0))
                val = sh.shade(vert_pos, 0.9, 10, 2,
                               np.array((0.5, 0.4, 0.1, 1.0)), np.array((0.1, 0.9, 0.1, 1.0)))
                #val = np.array((1.0,1.0,1.0))
                img[r, c, :] = val[:3]

        self._layout.addWidget(self._figure_canvas)
        self._axis.imshow(img)

        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.setLayout(self._layout)
