from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QCheckBox, QApplication, QMainWindow, QTableWidget, QTableWidgetItem


class CheckboxItem(QWidget):

    state_changed = pyqtSignal(QWidget)

    def __init__(self, label:str, content=None, parent=None):
        super().__init__(parent)
        self.label = label
        self.content = content
        self._checkbox = QCheckBox(text=self.label, parent=self)

        self._layout = QHBoxLayout()
        self._init()

    def _init(self):
        self._checkbox.stateChanged.connect(self._state_changed)
        self._layout.addWidget(self._checkbox)

        self.setLayout(self._layout)

    def get_state(self) -> int:
        return self._checkbox.checkState()

    def set_state(self, state):
        self._checkbox.setCheckState(state)

    def set_checkable(self, checkable:bool):
        self._checkbox.setCheckable(checkable)

    def _state_changed(self, state:int):
        self.state_changed.emit(self)
