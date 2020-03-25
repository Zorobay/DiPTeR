from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy, QVBoxLayout


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
