import typing
import uuid
from abc import abstractmethod

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy


class Input:
    # To be called by subclasses when input changes!
    input_changed = pyqtSignal()

    def __init__(self, internal_type: str):
        self._internal_type = internal_type

    @abstractmethod
    def get_value(self) -> typing.Any:
        """Return the value of this input widget."""
        raise NotImplementedError("Input subclass need to implement this method!")

    @abstractmethod
    def get_gl_value(self) -> typing.Any:
        """Return the value of this input widget converted to an OpenGL compatible type."""
        raise NotImplementedError("Input subclass need to implement this method!")

    @abstractmethod
    def set_default_value(self, default_value: typing.Any):
        """Set the default value of this input."""
        raise NotImplementedError("Input subclass need to implement this method!")


class InputModule(QWidget):
    input_changed = pyqtSignal(str, object, str, object)  # uniform variable name, input value, internal type, input module ID

    def __init__(self, label: str, internal_type: str, uniform_var: str, widget: Input):
        super().__init__()
        self.label = label
        self._internal_type = internal_type
        self._uniform_var = uniform_var
        self.widget = widget
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
        self._layout.addWidget(self._label_widget)
        self._layout.addWidget(self.widget)

        self.setLayout(self._layout)

    @pyqtSlot(name="_input_changed")
    def _input_changed(self):
        self.input_changed.emit(self.uniform_var, self.widget.get_gl_value(), self._internal_type, self.id)

    @property
    def uniform_var(self) -> str:
        """The name of the uniform variable in the GLSL code that this input is connected to."""
        return self._uniform_var

    @property
    def id(self) -> uuid.UUID:
        return self._id

    def set_label_palette(self, palette: QPalette):
        self._label_widget.setPalette(palette)

    def set_default_value(self, default_value: typing.Any):
        self.widget.set_default_value(default_value)

