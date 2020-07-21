import typing

from PyQt5.QtWidgets import QWidget

from dipter.gui.widgets.node_input.io_module import Input


class ShaderInput(QWidget, Input):

    def __init__(self, dtype):
        super().__init__(dtype=dtype)

        self.setFixedSize(0, 0)
        self.setVisible(False)

    def get_value(self) -> typing.Any:
        pass

    def get_gl_value(self) -> typing.Any:
        pass

    def set_value(self, value: typing.Any):
        pass
