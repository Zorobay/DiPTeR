import typing
import uuid
from abc import abstractmethod

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy, QVBoxLayout


class Input:
    # To be called by subclasses when widgets changes!
    input_changed = pyqtSignal()

    def __init__(self, dtype: str):
        self._dtype = dtype

    @abstractmethod
    def get_value(self) -> typing.Any:
        """Return the value of this widgets widget."""
        raise NotImplementedError("Input subclass need to implement this method!")

    @abstractmethod
    def get_gl_value(self) -> typing.Any:
        """Return the value of this widgets widget converted to an OpenGL compatible type."""
        raise NotImplementedError("Input subclass need to implement this method!")

    @abstractmethod
    def set_default_value(self, default_value: typing.Any):
        """Set the default value of this widgets."""
        raise NotImplementedError("Input subclass need to implement this method!")

    def get_dtype(self) -> str:
        return self._dtype


class SocketModule(QWidget):
    input_changed = pyqtSignal()

    def __init__(self, label: str, input_widget: Input):
        super().__init__()
        self.label = label
        self.widget = input_widget
        self._id = uuid.uuid4()
        self._label_widget = QLabel(self.label)

        self._layout = QHBoxLayout()
        self._palette = QPalette()

        self.init_widget()

    def init_widget(self):
        self._palette.setColor(QPalette.Background, QColor(0, 0, 0, 0))
        self.setPalette(self._palette)

        self.widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.widget.input_changed.connect(self._input_changed)
        self._layout.addWidget(self.widget)

        self._layout.addWidget(self._label_widget)

        self.setLayout(self._layout)

    @pyqtSlot(name="_input_changed")
    def _input_changed(self):
        self.input_changed.emit()

    @property
    def id(self) -> uuid.UUID:
        return self._id

    def set_label_palette(self, palette: QPalette):
        self._label_widget.setPalette(palette)

    def set_default_value(self, default_value: typing.Any):
        self.widget.set_default_value(default_value)

    def get_gl_value(self):
        return self.widget.get_gl_value()


class OutputModule(QWidget):

    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self._label_widget = QLabel(self.label)

        self._layout = QVBoxLayout()
        self._palette = QPalette()

        self.init_widget()

    def init_widget(self):
        self._palette.setColor(QPalette.Background, QColor(0, 0, 0, 0))
        self.setPalette(self._palette)

        self._layout.setAlignment(Qt.AlignRight)
        self._layout.addWidget(self._label_widget)

        self.setLayout(self._layout)

    def set_label_palette(self, palette: QPalette):
        self._label_widget.setPalette(palette)
