import logging
import typing

import numpy as np
import torch
from PIL.Image import Image
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QWidget, QGridLayout, QSplitter, QTableWidget, QPushButton, QMessageBox, QProgressDialog, QVBoxLayout
from gui.node_editor.g_shader_node import GMaterialOutputNode
from gui.widgets.checkbox_item import CheckboxItem
from gui.widgets.io_module import Module
from gui.widgets.line_input import IntInput, FloatInput
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from misc import image_funcs
from misc.fifo_queue import FIFOQueue
from node_graph.parameter import Parameter
from optimization.gradient_descent import GradientDescentSettings, GradientDescent

_logger = logging.getLogger(__name__)


class PlotWidget3D(QWidget):

    def __init__(self):
        super().__init__()
        self._figure = Figure(figsize=(5, 5))
        self._canvas = FigureCanvas(self._figure)
        self._navbar = NavigationToolbar(self._canvas, self)
        self._layout = QVBoxLayout()
        self._layout.addWidget(self._navbar)
        self._layout.addWidget(self._canvas)
        self.setLayout(self._layout)

    def get_axis(self):
        return self._figure.add_subplot(111, projection="3d")

    def get_canvas(self):
        return self._canvas


class LossVisualizer(QWidget):

    def __init__(self, mat_output_node: GMaterialOutputNode):
        super().__init__(parent=None)
        self._mat_out_node = mat_output_node

        # Declare widgets
        self._layout = QGridLayout()
        self._splitter = QSplitter(Qt.Horizontal)
        self._plot = PlotWidget3D()
        self._canvas = self._plot.get_canvas()
        self._fig_ax = self._plot.get_axis()
        self._table_widget = QTableWidget(self)
        self._p1_res = IntInput(0, 100)
        self._p2_res = IntInput(0, 100)
        self._plot_button = QPushButton("Plot Loss")
        self._start_gd_button = QPushButton("Start Gradient Descent")

        # Declare data
        self._item_queue = FIFOQueue(maxsize=2)
        self._bg_brush_selected = QBrush(QColor("#8bf9b0"))
        self._bg_brush_default = QBrush(QColor("#ffffff"))
        self._settings = {}
        self._target_image = None
        self._target_matrix = None
        self._out_node = None
        self._thread = None
        self._gd = None
        self._p1 = None
        self._p2 = None
        self._plot_last_x = None
        self._plot_last_y = None

        self._init()
        self._list_parameters()

    def _init(self):
        self.setWindowTitle("Loss Visualizer")

        # Setup table widget
        self._table_widget.setColumnCount(3)
        self._table_widget.setHorizontalHeaderLabels(["Parameter", "Min", "Max"])
        self._table_widget.setColumnWidth(1, 50)
        self._table_widget.setColumnWidth(2, 50)

        # Setup plot
        self._fig_ax.set_title("Loss Surface")
        self._fig_ax.set_xlabel("Parameter ?")
        self._fig_ax.set_ylabel("Parameter ?")
        self._fig_ax.set_zlabel("Loss Value")

        # Setup resolution input
        self._p1_res.set_value(20)
        self._p2_res.set_value(20)
        p1_module = Module("Param 1 Res.", self._p1_res)
        p2_module = Module("Param 2 Res.", self._p2_res)

        # Setup buttons
        self._plot_button.clicked.connect(self._plot_loss)
        self._start_gd_button.clicked.connect(self._start_gd)
        self._start_gd_button.setEnabled(False)

        # Add widgets to layout
        self._splitter.addWidget(self._table_widget)
        self._splitter.addWidget(self._plot)
        self._layout.addWidget(self._splitter, 0, 0, 1, 5)
        self._layout.addWidget(p1_module, 1, 1)
        self._layout.addWidget(p2_module, 1, 2)
        self._layout.addWidget(self._plot_button, 1, 3)
        self._layout.addWidget(self._start_gd_button, 1, 4)

        self.setLayout(self._layout)

    def _list_parameters(self):
        _, param_dict = self._mat_out_node.get_backend_node().render(10, 10, retain_graph=True)
        row = 0
        self._table_widget.setRowCount(len(param_dict))

        for key in param_dict:
            param = param_dict[key]
            range = param.get_range()
            min_item = FloatInput(range[0], range[1])
            min_item.set_value(range[0])
            max_item = FloatInput(range[0], range[1])
            max_item.set_value(range[1])
            item = CheckboxItem(key, content={"param": param, "index": -1})
            item.state_changed.connect(self._item_state_changed)
            self._table_widget.setCellWidget(row, 0, item)
            self._table_widget.setCellWidget(row, 1, min_item)
            self._table_widget.setCellWidget(row, 2, max_item)

            row += 1

            if param.is_vector():
                item.set_checkable(False)
                item.setEnabled(False)

                for i in range(param.shape()[1]):
                    self._table_widget.insertRow(row)
                    min_item = FloatInput(range[0], range[1])
                    min_item.set_value(range[0])
                    max_item = FloatInput(range[0], range[1])
                    max_item.set_value(range[1])
                    sub_item = CheckboxItem("  [{}]".format(i), content={"param": param, "index": i})
                    sub_item.state_changed.connect(self._item_state_changed)
                    self._table_widget.setCellWidget(row, 0, sub_item)
                    self._table_widget.setCellWidget(row, 1, min_item)
                    self._table_widget.setCellWidget(row, 2, max_item)
                    row += 1

        self._table_widget.resizeColumnToContents(0)
        self._table_widget.resizeRowsToContents()

    def _checked_items(self) -> typing.List[typing.Tuple[QWidget, QWidget, QWidget]]:
        checked = []

        for i in range(self._table_widget.rowCount()):
            item = self._table_widget.cellWidget(i, 0)
            min_item = self._table_widget.cellWidget(i, 1)
            max_item = self._table_widget.cellWidget(i, 2)
            if item.get_state() == Qt.Checked:
                checked.append((item, min_item, max_item))

        return checked

    def _item_state_changed(self, item: CheckboxItem):
        if item.get_state() == Qt.Checked:
            if self._item_queue.is_full():
                first_item = self._item_queue.pop()
                first_item.set_state(Qt.Unchecked)

            self._item_queue.put(item)
        elif item.get_state() == Qt.Unchecked:
            if item in self._item_queue:
                self._item_queue.remove(item)

    def _plot_loss(self):
        self._fig_ax.clear()

        W, H = self._settings.render_width, self._settings.render_height
        R1, R2 = self._p1_res.get_gl_value(), self._p2_res.get_gl_value()
        progress_dialog = QProgressDialog("Calculating loss surface...", "Cancel", 0, R1 - 1, self)
        progress_dialog.setWindowTitle("Calculating")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(1)

        self._target_matrix = image_funcs.image_to_tensor(self._target_image, (W, H))
        loss_surface = np.empty((R1, R2))
        loss_f = self._settings.loss_func
        checked_items = self._checked_items()

        item1 = checked_items[0][0]
        item1_min = checked_items[0][1].get_gl_value()
        item1_max = checked_items[0][2].get_gl_value()
        self._fig_ax.set_xlabel(item1.label)
        self._p1: Parameter = item1.content["param"]
        self._p1.save_value()
        p1_values = torch.from_numpy(np.linspace(item1_min, item1_max, num=R1, endpoint=True))

        item2 = checked_items[1][0]
        item2_min = checked_items[1][1].get_gl_value()
        item2_max = checked_items[1][2].get_gl_value()
        self._fig_ax.set_ylabel(item2.label)
        self._p2: Parameter = item2.content["param"]
        self._p2.save_value()
        p2_values = torch.from_numpy(np.linspace(item2_min, item2_max, num=R2, endpoint=True))

        min_loss = np.finfo(np.float32).max
        min_loss_p1 = None
        min_loss_p2 = None

        for i in range(R1):
            self._p1.set_value(p1_values[i], index=item1.content["index"])
            progress_dialog.setValue(i)

            if progress_dialog.wasCanceled():
                return

            for j in range(R2):
                self._p2.set_value(p2_values[j], index=item2.content["index"])
                r, _ = self._mat_out_node.get_backend_node().render(W, H, retain_graph=True)
                loss = loss_f(r, self._target_matrix).detach().numpy()
                if loss < min_loss:
                    min_loss = loss
                    min_loss_p1 = self._p1.tensor().detach().numpy()
                    min_loss_p2 = self._p2.tensor().detach().numpy()

                loss_surface[i, j] = loss

            _logger.info("{:.2f}% complete...".format((i + 1) / R1 * 100))

        P1, P2 = torch.meshgrid([p1_values, p2_values])
        self._p1.restore_value()
        self._p2.restore_value()

        self._fig_ax.plot_surface(P1, P2, loss_surface, cmap=plt.cm.viridis)
        self._fig_ax.set_zlim(bottom=0)

        # Add min value marker
        self._fig_ax.plot([min_loss_p1], [min_loss_p2], [min_loss], marker='+', color="#ff00ff")
        self._fig_ax.text(min_loss_p1, min_loss_p2, min_loss*1.1, "Minimum Loss = {:.4f}".format(float(min_loss)), color='#ff00ff')

        self._canvas.draw()
        self._start_gd_button.setEnabled(True)

    def _start_gd(self):
        self._gd = GradientDescent(self._target_image, self._out_node, self._settings)
        self._thread = QThread()
        self._gd.iteration_done.connect(self._gd_callback)
        self._gd.moveToThread(self._thread)
        self._thread.started.connect(self._gd.run)
        self._gd.finished.connect(self._finish_gradient_descent)

        _logger.debug("Started Gradient Descent Thread...")

        self._thread.start()

    def _gd_callback(self, info: dict):
        loss = info["loss"]
        x = self._p1.tensor().detach().numpy()
        y = self._p2.tensor().detach().numpy()
        self._fig_ax.plot([x],[y], zs=[loss], color='#ff0000', marker='o', linestyle='-', markersize=8)
        self._canvas.draw()
        _logger.debug("Loss: {}, P1: {}, P2: {}".format(loss, self._plot_last_x, self._plot_last_y))

    def _finish_gradient_descent(self):
        if self._thread.isRunning():
            _logger.info("Stopping Gradient Descent Thread...")
            self._gd.stop()
            self._thread.quit()
            self._gd.restore_params()

    def open(self, settings: GradientDescentSettings, target: Image, mat_out_node: GMaterialOutputNode):
        if target is None:
            msg = QMessageBox(QMessageBox.Warning, "Need to set Target Texture!", "Can not open loss visualizer because target texture is not set.")
            msg.exec()
        else:
            self._settings = settings
            self._target_image = target
            self._out_node = mat_out_node
            super().show()
