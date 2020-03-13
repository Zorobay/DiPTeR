from PyQt5.QtWidgets import QGraphicsGridLayout


class GraphicsGridLayout(QGraphicsGridLayout):

    def __init__(self, *args):
        super().__init__(*args)
        self._margins = (0, 0, 0, 0)

    def setContentsMargins(self, left: float, top: float, right: float, bottom: float):
        self._margins = left, top, right, bottom

        # Refresh the layout
        self.invalidate()
        self.activate()

    def getContentsMargins(self):
        return self._margins
