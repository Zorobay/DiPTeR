import typing

import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from dipter.gui.widgets.node_input.io_module import Input
from dipter.gui.widgets.node_input.line_input import FloatInput
from dipter.node_graph.data_type import DataType


class ArrayInput(QWidget, Input):

    def __init__(self, size: int, min_: float, max_: float, dtype: DataType):
        super().__init__(dtype=dtype)

        self._size = size
        self._min = min_
        self._max = max_
        self._inputs = []
        self._layout = QVBoxLayout()

        self._init()

    def _init(self):
        for i in range(self._size):
            inp = FloatInput(self._min, self._max, self._dtype)
            inp.input_changed.connect(self._fire_input_changed)
            self._layout.addWidget(inp)
            self._inputs.append(inp)

        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

    def _fire_input_changed(self):
        self.input_changed.emit()

    def get_value(self) -> typing.Any:
        return np.array((self._x.get_value(), self._y.get_value(), self._z.get_value()))

    def get_gl_value(self) -> typing.Any:
        out = np.zeros(self._size)
        for i, inp in enumerate(self._inputs):
            out[i] = inp.get_gl_value()

        return out

    def set_value(self, value: typing.Any):
        for i, d in enumerate(value):
            self._inputs[i].set_value(d)
