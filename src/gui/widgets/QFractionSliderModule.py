from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

from src.gui.widgets.QFractionSlider import QFractionSlider


class QFractionSliderModule(QFractionSlider):

    def __init__(self, precision=0.01):
        super().__init__(precision)
        self._widget = QWidget()
        self._min_lbl = None
        self._max_lbl = None
        self._value_lbl = None

        self._vbox_main = QVBoxLayout()
        self._hbox_slider = QHBoxLayout()
        self._hbox_range = QHBoxLayout()

        self.setupUI()

    def setupUI(self):
        self._hbox_slider.addWidget(self)
        self.valueChanged.connect(self.update_value)
        self._value_lbl = QLabel(str(self.value()))
        self._hbox_slider.addWidget(self._value_lbl)

        self._min_lbl = QLabel("0.0", alignment=Qt.AlignLeft)
        self._max_lbl = QLabel("1.0", alignment=Qt.AlignRight)
        self._hbox_range.addWidget(self._min_lbl)
        self._hbox_range.addWidget(self._max_lbl)

        self._widget.setLayout(self._vbox_main)
        self._vbox_main.addLayout(self._hbox_slider)
        self._vbox_main.addLayout(self._hbox_range)

    def getAsWidget(self):
        return self._widget

    @pyqtSlot()
    def update_value(self):
        self._value_lbl.setText(str(self.value()))

