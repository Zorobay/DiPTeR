import typing
from abc import abstractmethod

import numpy as np
import torch
from PyQt5.QtGui import QValidator, QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QLineEdit

from dipter.gui.widgets.node_input.io_module import Input
from dipter.node_graph.data_type import DataType

Number = typing.Union[int, float]

INT_MIN = np.iinfo(np.int32).min
INT_MAX = np.iinfo(np.int32).max
FLOAT_MIN = np.finfo(np.float32).min
FLOAT_MAX = np.finfo(np.float32).max


class NumberValidator:
    MALFORMED_NUMBER = 0
    TOO_LARGE = 1
    TOO_SMALL = 2

    def __init__(self, bottom, top):
        """Subclasses must pass a function used to convert the string to be validated into the correct data type."""
        self._reason_for_invalid = None
        self.bottom = bottom
        self.top = top

    def validate(self, input_: str, cursor_pos: int) -> (int, str, int):
        input_ = input_.strip()

        if input_ == "" or input_ == "-" or input_ == "+":
            return QValidator.Intermediate, input_, cursor_pos

        if input_ == "-" or input_ == "+":
            return QValidator.Intermediate, input_, cursor_pos

        try:
            as_number = self.validate_intermediate(input_)

            ltt = as_number <= self.top
            gtb = as_number >= self.bottom

            if ltt:
                if gtb:
                    self._reason_for_invalid = None
                    return QValidator.Acceptable, input_, cursor_pos
                else:
                    self._reason_for_invalid = self.TOO_SMALL
                    return QValidator.Intermediate, input_, cursor_pos
            else:
                self._reason_for_invalid = self.TOO_LARGE
                return QValidator.Invalid, input_, cursor_pos

        except ValueError:
            self._reason_for_invalid = self.MALFORMED_NUMBER
            return QValidator.Invalid, input_, cursor_pos

    @abstractmethod
    def validate_intermediate(self, intermediate_input: str) -> Number:
        """Validate the intermediate input. This is called after the string has been cleaned and checked if its empty (including '+' or '-').

        This function should raise a ValueError if the string does not follow the formatting of the wanted number type. This includes converting a
        valid float with decimals to an integer (as integers should not contain decimals). This enables fixup() to clean the string properly.

        :return: The string input converted to a number.
        """
        pass


class FloatValidator(NumberValidator, QDoubleValidator):

    def __init__(self, bottom: float, top: float, decimals: int):
        """
        Creates a new DoubleValidator with fixup (attempts to correct an invalid string)
        """
        NumberValidator.__init__(self, bottom, top)
        QDoubleValidator.__init__(self, bottom, top, decimals)

    def validate_intermediate(self, intermediate_input: str) -> Number:
        # TODO Check for number of decimals to display!
        return float(intermediate_input)

    def fixup(self, input_: str) -> str:
        """Attempts to return a corrected version of the invalid widgets string."""

        corrected = input_

        if self._reason_for_invalid == self.MALFORMED_NUMBER:
            corrected = corrected.replace(',', '.')  # replace commas with full stops
            corrected = corrected.replace('\D', '')  # Remove any non-digits
        elif self._reason_for_invalid == self.TOO_SMALL:
            corrected = str(self._bottom())
        elif self._reason_for_invalid == self.TOO_LARGE:
            corrected = str(self._top())

        return corrected


class IntValidator(NumberValidator, QIntValidator):

    def __init__(self, bottom: int, top: int):
        """
        Creates a new IntValidator with fixup (attempts to correct an invalid string)
        """
        NumberValidator.__init__(self, bottom, top)
        QIntValidator.__init__(self, bottom, top)

    def validate_intermediate(self, intermediate_input: str) -> Number:
        if ',' in intermediate_input or '.' in intermediate_input:
            raise ValueError("Decimal symbol in integer input is invalid!")

        return int(intermediate_input)

    def fixup(self, input_: str) -> str:
        """Attempts to return a corrected version of the invalid widgets string."""

        corrected = input_

        if self._reason_for_invalid == self.MALFORMED_NUMBER:
            corrected = corrected.replace(',', '.')  # replace commas with full stops
            corrected = corrected.replace('\D', '')  # Remove any non-digits
            try:
                corrected = int(corrected)  # Attempt to remove decimals
            except ValueError:
                pass
        elif self._reason_for_invalid == self.TOO_SMALL:
            corrected = str(self.bottom())
        elif self._reason_for_invalid == self.TOO_LARGE:
            corrected = str(self.top())

        return corrected


class MultipleNumberValidator(QValidator):

    def __init__(self, validate: typing.Callable, fixup: typing.Callable):
        super().__init__()
        self.func_validate = validate
        self.func_fixup = fixup

    def validate(self, input_: str, cursor_pos: int) -> (int, str, int):
        numbers = input_.split(",")
        total_res = self.Acceptable
        for num in numbers:
            res, _, _ = self.func_validate(num, cursor_pos)

            if res == self.Intermediate:
                total_res = self.Intermediate
            elif res == self.Invalid:
                return self.Invalid, input_, cursor_pos

        return total_res, input_, cursor_pos

    def fixup(self, input_: str) -> str:
        numbers = input_.split(",")
        for i, num in enumerate(numbers):
            corrected = self.func_fixup(num)
            numbers[i] = corrected

        return ", ".join(numbers)


class MultipleFloatValidator(MultipleNumberValidator):
    def __init__(self, bottom: float, top: float, decimals: int):
        validator = FloatValidator(bottom, top, decimals)
        super().__init__(validate=validator.validate, fixup=validator.fixup)


class MultipleIntValidator(IntValidator):

    def __init__(self, bottom: int, top: int):
        super().__init__(bottom, top)

    def validate(self, input_: str, cursor_pos: int) -> (int, str, int):
        numbers = input_.split(",")
        total_res = self.Acceptable
        for num in numbers:
            res, _, _ = super().validate(num, cursor_pos)

            if res == self.Intermediate:
                total_res = self.Intermediate
            elif res == self.Invalid:
                return self.Invalid, input_, cursor_pos

        return total_res, input_, cursor_pos

    def fixup(self, input_: str) -> str:
        numbers = input_.split(",")
        for i, num in enumerate(numbers):
            corrected = super().fixup(num)
            numbers[i] = corrected

        return ", ".join(numbers)


class LineInput(QLineEdit, Input):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._old_text = ""
        self.editingFinished.connect(self._handle_change)
        self.textChanged.connect(self._handle_change)
        self.setMaximumWidth(80)

    def _handle_change(self):
        if self.text() != self._old_text:
            self.input_changed.emit()
            self._old_text = self.text()

    def get_value(self) -> str:
        return self.text()


class FloatInput(LineInput):

    def __init__(self, min_: float = FLOAT_MIN, max_: float = FLOAT_MAX, dtype=DataType.Float):
        assert min_ <= max_, "Minimum value must be less than or equal to maximum value!"
        super().__init__(dtype=dtype)

        self.min_ = min_
        self.max_ = max_
        self.setValidator(FloatValidator(bottom=self.min_, top=self.max_, decimals=5))

    def set_value(self, value: typing.Any):
        assert isinstance(value, (float, int, np.int, np.float32, torch.FloatTensor, torch.cuda.FloatTensor)), \
            "Incompatible type of default value for FloatInput!"

        self.setText(str(float(value)))

    def get_gl_value(self) -> typing.Any:
        text = self.text().strip()
        if text == "+" or text == "-" or text == "":
            return np.float32(0.0)

        return np.float32(float(self.text()))


class IntInput(LineInput):

    def __init__(self, min_: int = INT_MIN, max_: int = INT_MAX, dtype=DataType.Int):
        assert min_ <= max_, "Minimum value must be less than or equal to maximum value!"
        super().__init__(dtype=dtype)

        self.min_ = min_
        self.max_ = max_
        self.setValidator(IntValidator(bottom=self.min_, top=self.max_))

    def set_value(self, value: typing.Any):
        if isinstance(value, (float, np.float)):
            value = int(value)
        elif isinstance(value, torch.Tensor):
            value = int(value.detach().cpu().numpy())

        assert isinstance(value, (int, np.int)), "Incompatible type of default value for FloatInput!"

        self.setText(str(value))

    def get_gl_value(self) -> int:
        text = self.text().strip()
        if text == "+" or text == "-" or text == "":
            return np.int(0)

        return int(self.text())


class MultipleIntInput(LineInput):

    def __init__(self, min_: int = INT_MIN, max_: int = INT_MAX, dtype=DataType.Int):
        assert min_ <= max_, "Minimum value must be less than or equal to maximum value!"
        super().__init__(dtype=dtype)

        self.min_ = min_
        self.max_ = max_
        self.setValidator(MultipleIntValidator(bottom=self.min_, top=self.max_))

    def set_value(self, values: typing.Iterable):
        list_values = []
        for value in values:
            if isinstance(value, (float, np.float)):
                value = int(value)
            elif isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().numpy())

            list_values.append(value)
            assert isinstance(value, (int, np.int)), "Incompatible type of default value for FloatInput!"

        self.setText(", ".join([str(i) for i in list_values]))

    def get_gl_value(self) -> typing.List[int]:
        text = self.text().strip()
        if len(text) == 0:
            return []
        nums = text.split(",")

        return [int(num) for num in nums]


class MultipleFloatInput(LineInput):

    def __init__(self, min_: float = FLOAT_MIN, max_: float = FLOAT_MAX, dtype=DataType.Int):
        assert min_ <= max_, "Minimum value must be less than or equal to maximum value!"
        super().__init__(dtype=dtype)

        self.min_ = min_
        self.max_ = max_
        self.setValidator(MultipleFloatValidator(bottom=self.min_, top=self.max_, decimals=5))

    def set_value(self, values: typing.Iterable):
        list_values = []
        for value in values:
            if isinstance(value, (float, np.float)):
                value = float(value)
            elif isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().numpy())

            list_values.append(value)
            assert isinstance(value, (float, np.float32)), "Incompatible type of default value for FloatInput!"

        self.setText(", ".join([str(i) for i in list_values]))

    def get_gl_value(self) -> typing.List[float]:
        text = self.text().strip()
        if len(text) == 0:
            return []
        nums = text.split(",")

        return [float(num) for num in nums]


class StringInput(LineInput, Input):

    def __init__(self):
        super().__init__(dtype=DataType.String)

    def get_gl_value(self) -> typing.Any:
        return self.get_value()

    def set_value(self, value: typing.Any):
        self.setText(value)
