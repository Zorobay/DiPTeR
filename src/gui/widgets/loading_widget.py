from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QLabel, QDialog, QVBoxLayout, QWidget

GIF_DIR = Path.cwd() / "res" / "gif"


class LoadingModal(QWidget):

    def __init__(self, parent, message: str):
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self._message = QLabel(message)
        self._gif_label = QLabel()
        self._movie = QMovie()

        self.setWindowFlag(Qt.Tool, True)
        self._init_modal()

    def _init_modal(self):
        self._movie.setFileName(str((GIF_DIR / "loading.gif").resolve()))
        self._gif_label.setMovie(self._movie)
        # self._gif_label.show()
        self._movie.start()
        self._layout.addWidget(self._message)
        self._layout.addWidget(self._gif_label)
        self.setLayout(self._layout)
