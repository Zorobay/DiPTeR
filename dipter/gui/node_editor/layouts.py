from PyQt5.QtWidgets import QGraphicsGridLayout, QGraphicsLinearLayout, QGraphicsLayout


class NegativeContentMargins(QGraphicsLayout):

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


class GraphicsGridLayout(QGraphicsGridLayout, NegativeContentMargins):

    def __init__(self, *args):
        super().__init__(*args)


class GraphicsLinearLayout(QGraphicsLinearLayout, NegativeContentMargins):

    def __init__(self, *args):
        super().__init__(*args)
