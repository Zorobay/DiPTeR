from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel


class LabelledInput(QWidget):

    def __init__(self, label, widget: QWidget):
        super().__init__()
        self.label = label
        self.widget = widget

        self.layout = QHBoxLayout()
        self.layout.addWidget(QLabel(self.label))
        self.layout.addWidget(self.widget)
        self.setLayout(self.layout)
