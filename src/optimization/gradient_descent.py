import time
import typing

import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from src.gui.node_editor.g_shader_node import GMaterialOutputNode
from src.misc import image_funcs
from src.optimization.losses import Loss
from torch.optim.optimizer import Optimizer


class GradientDescentSettings:

    def __init__(self):
        self._loss_func = None
        self._render_width = 200
        self._render_height = 200
        self._max_iter = 100
        self._early_stopping_thresh = 0.01
        self._learning_rate = 0.01
        self._decay = 0.99

    def set_loss_func(self, func: Loss):
        self._loss_func = func

    def get_loss_func(self) -> Loss:
        return self._loss_func

    def set_render_width(self, width: int):
        self._render_width = width

    def get_render_width(self) -> int:
        return self._render_width

    def set_render_height(self, height: int):
        self._render_height = height

    def get_render_height(self) -> int:
        return self._render_height

    def set_max_iter(self, iter: int):
        self._max_iter = iter

    def get_max_iter(self) -> int:
        return self._max_iter

    def set_early_stopping_thresh(self, thresh: float):
        self._early_stopping_thresh = thresh

    def get_early_stopping_thresh(self) -> float:
        return self._early_stopping_thresh

    def set_learning_rate(self, rate: float):
        self._learning_rate = rate

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def set_decay(self, decay: float):
        self._decay = decay

    def get_decay(self) -> float:
        return self._decay


class GradientDescent(QObject):
    first_render_done = pyqtSignal(dict)  # parameter values dict
    iteration_done = pyqtSignal(dict)
    finished = pyqtSignal(dict, np.ndarray)

    def __init__(self, image_to_match: torch.Tensor, out_node: GMaterialOutputNode, settings: GradientDescentSettings,
                 optimizer: Optimizer = torch.optim.Adam):
        super().__init__()
        self.out_node = out_node
        self.settings = settings
        self.optimizer = optimizer
        self.width, self.height = self.settings.get_render_width(), self.settings.get_render_height()
        self.lr = self.settings.get_learning_rate()
        self.decay = self.settings.get_decay()
        self.max_iter = self.settings.get_max_iter()
        self.early_stopping_thresh = self.settings.get_early_stopping_thresh()
        self.target = image_to_match
        self.f = None
        self.loss_func = self.settings.get_loss_func()
        self._stop = False

    def stop(self):
        self._stop = True

    @pyqtSlot(name='run')
    def run(self):
        self.target = image_funcs.image_to_tensor(self.target, (self.settings.get_render_width(), self.settings.get_render_height()))
        params, loss_hist = self._run_gd(lr=self.lr, max_iter=self.max_iter, early_stopping_thresh=self.early_stopping_thresh)
        self.finished.emit(params, loss_hist)

    def _run_gd(self, lr=0.01, max_iter=150, early_stopping_thresh=0.01) -> typing.Tuple[list, np.ndarray]:
        _, args_dict = self.out_node.render(self.width, self.height, retain_graph=True)
        args_list = [args_dict[k] for k in args_dict]

        for p in args_list:
            p.requires_grad = True

        optimizer = self.optimizer(args_list, lr=lr)
        loss_hist = np.empty(max_iter, dtype=np.float32)

        for i in range(max_iter):
            if self._stop:
                return args_dict, loss_hist

            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                start = time.time()
                render, _ = self.out_node.render(self.width, self.height, retain_graph=True)
                loss = self.loss_func(render, self.target)
                new_loss_np = float(loss.detach())
                loss_hist[i] = new_loss_np
                props = {'iter': i, 'loss': new_loss_np, 'loss_hist': loss_hist[:i + 1], 'learning_rate': lr, 'params': args_dict,
                         'iter_time': 0.0, 'render': render}

                # We need to break here, otherwise the parameters will change when we call optimizer.step()
                if loss <= early_stopping_thresh:
                    self.iteration_done.emit(props)
                    break

                loss.backward(retain_graph=False, create_graph=False)

                optimizer.step()

            props['iter_time'] = time.time() - start
            self.iteration_done.emit(props)

        return args_dict, loss_hist
