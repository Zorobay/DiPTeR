import numbers
import typing

import torch
from PyQt5.QtWidgets import QComboBox
from node_graph.data_type import DataType
from dipter.gui.widgets.node_input.io_module import Input


class IntChoiceInput(QComboBox, Input):

    def __init__(self, values: typing.List[str], dtype=DataType.Int_Choice):
        super().__init__(dtype=dtype)
        self.addItems(values)
        self.currentIndexChanged.connect(self._index_changed)

    def _index_changed(self, index:int):
        self.input_changed.emit()

    def get_value(self) -> typing.Any:
        return self.currentIndex()

    def get_gl_value(self) -> typing.Any:
        return self.currentIndex()

    def set_value(self, value: int):
        if isinstance(value, numbers.Number):
            value = int(value)
        elif isinstance(value, torch.IntTensor):
            value = int(value.detach().numpy())

        self.setCurrentIndex(value)
