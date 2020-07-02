from PyQt5.QtCore import Qt, QVariant
from PyQt5.QtWidgets import QListWidgetItem
from node_graph.parameter import Parameter


class ListWidgetItem(QListWidgetItem):

    def __init__(self, text: str, param: Parameter, index=-1):
        super().__init__(text, None, type=QListWidgetItem.Type)

        self.param = param
        self.index = index
        self.setFlags(self.flags() | Qt.ItemIsUserCheckable ^ Qt.ItemIsSelectable)
        self.setCheckState(Qt.Unchecked)

    def _enable_flag(self, flag):
        self.setFlags(self.flags() | flag)

    def _disable_flag(self, flag):
        self.setFlags(self.flags() ^ flag)

    def remove_checkbox(self):
        self.setData(Qt.CheckStateRole, QVariant())

    def set_checkable(self, checkable: bool):
        if checkable:
            self._enable_flag(Qt.ItemIsUserCheckable)
        else:
            self._disable_flag(Qt.ItemIsUserCheckable)

    def set_enabled(self, enabled: bool):
        if enabled:
            self._enable_flag(Qt.ItemIsEnabled)
        else:
            self._disable_flag(Qt.ItemIsEnabled)
