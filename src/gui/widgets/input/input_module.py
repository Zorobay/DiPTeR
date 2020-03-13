from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy


class InputModule(QWidget):

    def __init__(self, label: str, widget: QWidget):
        super().__init__()
        self.label = label
        self._label_widget = QLabel(self.label)
        self.widget = widget

        self._layout = QHBoxLayout()
        self._palette = QPalette()

        self.init_widget()

    def init_widget(self):
        self._palette.setColor(QPalette.Background, QColor(0, 0, 0, 0))
        self.setPalette(self._palette)

        self.widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._layout.addWidget(self._label_widget)
        self._layout.addWidget(self.widget)

        self.setLayout(self._layout)

    def set_label_palette(self, palette: QPalette):
        self._label_widget.setPalette(palette)

