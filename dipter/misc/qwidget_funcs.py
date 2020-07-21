from PyQt5.QtWidgets import QLayout


def clear_layout(qlayout: QLayout):
    """Removes all children of a layout."""
    for i in reversed(range(qlayout.count())):
        item = qlayout.itemAt(i)
        qlayout.removeItem(item)
        item.widget().deleteLater()
