import pyqtgraph as pg
import torch
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout


class ImagePlotter(pg.GraphicsLayoutWidget):

    def __init__(self, title=""):
        super().__init__()

        self._plotItem = pg.PlotItem(title=title)
        self._viewBox = self._plotItem.getViewBox()
        self._imgItem = pg.ImageItem()
        self._cant_render_text = pg.TextItem(text="Unable to render image.", color="r", anchor=(0.5, 0.5))
        self._font = QFont()
        self._layout = QHBoxLayout()

        self._image_width = 1
        self._image_height = 1

        self._init()

    def _init(self):
        self._font.setPointSize(20)
        self._cant_render_text.setFont(self._font)

        self._viewBox.addItem(self._imgItem)

        self._viewBox.setAspectLocked(True)
        self._viewBox.disableAutoRange()
        self._imgItem.setLevels(0, 1.0)

        self.enableMouse(False)
        self.setCentralItem(self._plotItem)

    def set_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img is not None:
            self._image_width = img.shape[0]
            self._image_height = img.shape[1]
            self._plotItem.removeItem(self._cant_render_text)
            self._imgItem.setImage(img, autoLevels=False, levels=(0, 1.))
            self.auto_range()
        else:
            self._plotItem.removeItem(self._cant_render_text)
            self._plotItem.addItem(self._cant_render_text)
            self._cant_render_text.setPos(self._image_width / 2, self._image_height / 2)

    def clear(self):
        self._imgItem.clear()
        self._plotItem.clear()
        self._imgItem._updateView()

    def auto_range(self):
        self.set_x_range(0, self._image_width)
        self.set_y_range(0, self._image_height)

    def set_x_range(self, start, end):
        self._viewBox.setXRange(start, end)

    def set_y_range(self, start, end):
        self._viewBox.setYRange(start, end)

    def hide_axes(self):
        self._plotItem.hideAxis("left")
        self._plotItem.hideAxis("bottom")

    def show_axes(self):
        self._plotItem.showAxis("left")
        self._plotItem.showAxis("bottom")

    def set_user_input(self, enable):
        self._plotItem.setMenuEnabled(enable)
        self._viewBox.setMenuEnabled(enable)
        self._plotItem.setMouseEnabled(x=enable,y=enable)
        self._viewBox.setMouseEnabled(x=enable,y=enable)

        self.enableMouse(enable)
