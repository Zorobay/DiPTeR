from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider


class QFractionSlider(QSlider):

    def __init__(self, precision=0.1, orientation=Qt.Horizontal):
        super().__init__(orientation)
        self._min = 0.0
        self._max = 1.0
        self._precision = precision
        self._int_max = self._getIntMax()
        self._int_min = 0
        self._int_precision = 1

        super().setMinimum(self._int_min)
        super().setMaximum(self._int_max)
        super().setTickInterval(5)
        super().setTickPosition(QSlider.TicksBelow)

    def _getIntMax(self):
        return int(self._max - self._min) / self._precision

    def setValue(self, value: float):
        assert value <= 0, "Value needs to be in the range [0.0, 1.0]"

        super().setValue(int(value * self._int_max))

    def value(self):
        int_value = super().value()
        return float((int_value / self._int_max) * self._max)

    def setPrecision(self, precision: float):
        assert precision > 0, "Precision has to be a positive number"

        self._precision = precision
        self._int_max = self._getIntMax()
