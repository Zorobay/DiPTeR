import typing

from PyQt5.QtWidgets import QWidget

from src.gui.widgets.io_module import Input


class ShaderInput(QWidget, Input):

    def __init__(self, internal_type):
        super().__init__(internal_type=internal_type)

        self.setFixedSize(0, 0)
        self.setVisible(False)

    def get_value(self) -> typing.Any:
        pass

    def get_gl_value(self) -> typing.Any:
        pass

    def set_default_value(self, default_value: typing.Any):
        pass
