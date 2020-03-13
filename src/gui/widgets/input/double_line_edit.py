import re

from PyQt5.QtGui import QDoubleValidator, QValidator
from PyQt5.QtWidgets import QLineEdit

reg_float = re.compile(r"(-?)(\d+)[\.,]?(\d*)", flags=re.UNICODE)


class DoubleValidator(QDoubleValidator):
    NOT_FLOAT = 0
    TOO_LARGE = 1
    TOO_SMALL = 2

    def __init__(self, *args, **kwargs):
        """
        Creates a new DoubleValidator with fixup (attempts to correct an invalid string)
        """
        super().__init__(*args, **kwargs)
        self._reason_for_invalid = None

    def validate(self, input_: str, cursor_pos: int) -> int:
        try:
            as_float = float(input_)

            top_valid = as_float <= self.top()
            bottom_valid = as_float >= self.bottom()

            if top_valid:
                if bottom_valid:
                    self._reason_for_invalid = None
                    return QValidator.Acceptable
                else:
                    self._reason_for_invalid = DoubleValidator.TOO_SMALL
                    return QValidator.Intermediate
            else:
                self._reason_for_invalid = DoubleValidator.TOO_LARGE
                return QValidator.Invalid

        except ValueError:
            self._reason_for_invalid = DoubleValidator.NOT_FLOAT
            return QValidator.Invalid

    def fixup(self, input_: str) -> str:
        """Attempts to return a corrected version of the invalid input string."""

        corrected = input_

        if self._reason_for_invalid == DoubleValidator.NOT_FLOAT:
            corrected = corrected.replace(',', '.')  # replace commas with full stops
            corrected = corrected.replace('\D', '')  # Remove any non-digits
        elif self._reason_for_invalid == DoubleValidator.TOO_SMALL:
            corrected = str(self.bottom())
        elif self._reason_for_invalid == DoubleValidator.TOO_LARGE:
            corrected = str(self.top())

        return corrected


class DoubleLineEdit(QLineEdit):

    def __init__(self, min_: float = 0.0, max_: float = 1.0, *args):
        assert min_ <= max_, "Minimum value must be less than or equal to maximum value!"
        super().__init__(*args)

        self.min_ = min_
        self.max_ = max_
        self._init_widget()

    def _init_widget(self):
        self.setValidator(DoubleValidator(bottom=self.min_, top=self.max_))
