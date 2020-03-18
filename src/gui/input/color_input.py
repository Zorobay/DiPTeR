import typing

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QPushButton, QColorDialog

from src.opengl.shader_types import INTERNAL_TYPE_RGB, INTERNAL_TYPE_RGBA
from src.gui.input.input_module import Input


class ColorInput(QPushButton, Input):

    def __init__(self, internal_type):
        super().__init__(internal_type=internal_type)
        self._color_dialog = QColorDialog()
        self._palette = QPalette()
        self._init_widget()

    def set_default_value(self, default_value: list):
        color = QColor.fromRgbF(*default_value)
        self._set_display_color(color)
        self._color_dialog.setCurrentColor(color)

    def get_gl_value(self) -> np.ndarray:
        if self._internal_type == INTERNAL_TYPE_RGB:
            return np.array((self._color.getRgbF()[:3]), dtype=np.float32)
        elif self._internal_type == INTERNAL_TYPE_RGBA:
            return np.array((self._color.getRgbF()), dtype=np.float32)
        else:
            raise TypeError("Internal type {} not supported for ColorInput!".format(self._internal_type))

    def get_value(self) -> typing.Any:
        return self._color

    def _init_widget(self):
        self._color_dialog.currentColorChanged.connect(self._current_color_changed)
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
