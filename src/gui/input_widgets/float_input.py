import re
import typing

import numpy as np
from PyQt5.QtGui import QDoubleValidator, QValidator
from PyQt5.QtWidgets import QLineEdit

from src.gui.input_widgets.input_module import Input

reg_float = re.compile(r"(-?)(\d+)[\.,]?(\d*)", flags=re.UNICODE)


class FloatValidator(QDoubleValidator):
    NOT_FLOAT = 0
    TOO_LARGE = 1
    TOO_SMALL = 2

    def __init__(self, *args, **kwargs):
        """
        Creates a new DoubleValidator with fixup (attempts to correct an invalid string)
        """
        super().__init__(*args, **kwargs)
        self._reason_for_invalid = None

    def validate(self, input_: str, cursor_pos: int) -> (int, str, int):
        try:
            input_ = input_.strip()

            if input_ == "":
                return QValidator.Intermediate, input_, cursor_pos

            if input_ == "-" or input_ == "+":
                return QValidator.Intermediate, input_, cursor_pos

            as_float = float(input_)

            top_valid = as_float <= self.top()
            bottom_valid = as_float >= self.bottom()

            if top_valid:
                if bottom_valid:
                    self._reason_for_invalid = None
                    return QValidator.Acceptable, input_, cursor_pos
                else:
                    self._reason_for_invalid = FloatValidator.TOO_SMALL
                    return QValidator.Intermediate, input_, cursor_pos
            else:
                self._reason_for_invalid = FloatValidator.TOO_LARGE
                return QValidator.Invalid, input_, cursor_pos

        except ValueError:
            self._reason_for_invalid = FloatValidator.NOT_FLOAT
            return QValidator.Invalid, input_, cursor_pos

    def fixup(self, input_: str) -> str:
        """Attempts to return a corrected version of the invalid input_widgets string."""

        corrected = input_

        if self._reason_for_invalid == FloatValidator.NOT_FLOAT:
            corrected = corrected.replace(',', '.')  # replace commas with full stops
            corrected = corrected.replace('\D', '')  # Remove any non-digits
        elif self._reason_for_invalid == FloatValidator.TOO_SMALL:
            corrected = str(self.bottom())
        elif self._reason_for_invalid == FloatValidator.TOO_LARGE:
            corrected = str(self.top())

        return corrected


class FloatInput(QLineEdit, Input):

    def __init__(self, internal_type, min_: float, max_: float, ):
        assert min_ <= max_, "Minimum value must be less than or equal to maximum value!"
        super().__init__(internal_type=internal_type)

        self.min_ = min_
        self.max_ = max_
        self._init_widget()

    def set_default_value(self, default_value: typing.Any):
        assert isinstance(default_value, (float, np.float32)), "Incompatible type of default value for FloatInput!"

        self.setText(str(default_value))

    def get_gl_value(self) -> typing.Any:
        text = self.text().strip()
        if text == "+" or text == "-" or text == "":
            return np.float32(0.0)

        return np.float32(float(self.text()))

    def get_value(self) -> str:
        return self.text()

    def _init_widget(self):
        self.setValidator(FloatValidator(bottom=self.min_, top=self.max_))
        self.editingFinished.connect(lambda: self.input_changed.emit())

    def setText(self, p_str):
        super().setText(p_str)
        self.input_changed.emit()

