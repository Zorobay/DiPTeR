import typing

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QPushButton, QColorDialog
from node_graph.data_type import DataType

from src.gui.widgets.io_module import Input


class ColorInput(QPushButton, Input):

    def __init__(self, dtype):
        super().__init__(dtype=dtype)
        self._color_dialog = QColorDialog()
        self._palette = QPalette()
        self._init_widget()

    def set_value(self, default_value: list):
        color = QColor.fromRgbF(*default_value)
        self._set_display_color(color)
        self._color_dialog.setCurrentColor(color)

    def get_gl_value(self) -> np.ndarray:
        if self._dtype == DataType.Vec3_RGB:
            return np.array((self._color.getRgbF()[:3]), dtype=np.float32)
        else:
            raise TypeError("Internal type {} not supported for ColorInput!".format(self._dtype))

    def get_value(self) -> typing.Any:
        return self._color

    def _init_widget(self):
        self._color_dialog.currentColorChanged.connect(self._current_color_changed)
        self._color_dialog.setOption(QColorDialog.ShowAlphaChannel, True)
        self.clicked.connect(self._open_color_dialog)

    @pyqtSlot(QColor)
    def _current_color_changed(self, color: QColor):
        self._set_display_color(color)
        self.input_changed.emit()

    def _open_color_dialog(self):
        self._color_dialog.open()

    def _set_display_color(self, color: QColor):
        self._palette.setColor(QPalette.Button, color)
        self.setAutoFillBackground(True)
        self.setFlat(True)
        self.setPalette(self._palette)
        self.update()
        self._color = color
