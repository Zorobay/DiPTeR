import typing

from PyQt5.QtWidgets import QWidget

from src.gui.widgets.input_module import Input


class ShaderInput(QWidget, Input):

    def __init__(self, *args):
        super().__init__(*args)

        self.setFixedSize(0, 0)
        self.setVisible(False)

    def get_value(self) -> typing.Any:
        pass

    def get_gl_value(self) -> typing.Any:
        pass

    def set_default_value(self, default_value: typing.Any):
        pass