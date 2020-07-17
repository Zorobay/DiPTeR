import time
import typing

import numpy as np
import torch
from PIL.Image import Image
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from dipter.gui.node_editor.g_shader_node import GMaterialOutputNode
from dipter.misc import image_funcs
from dipter.optimization.losses import Loss
from torch.optim.optimizer import Optimizer


class GradientDescentSettings:

    def __init__(self):
        self.loss_func = None
        self.optimizer = None
        self.render_width = 200
        self.render_height = 200
        self.max_iter = 100
        self.early_stopping_thresh = 0.01
        self.learning_rate = 0.01
        self.decay = 0.99


class GradientDescent(QObject):
    first_render_done = pyqtSignal(dict)  # parameter values dict
    iteration_done = pyqtSignal(dict)
    finished = pyqtSignal(dict, np.ndarray)

    def __init__(self, image_to_match: Image, out_node: GMaterialOutputNode, settings: GradientDescentSettings):
        super().__init__()
        self.out_node = out_node
        self.settings = settings
        self.target = image_to_match
        self._stop = False
        self._last_params = None

    def stop(self):
        self._stop = True

    @pyqtSlot(name='run')
    def run(self):
        self.target = image_funcs.image_to_tensor(self.target, (self.settings.render_width, self.settings.render_height))
        params, loss_hist = self._run_gd()
        self._last_params = params
        self.finished.emit(params, loss_hist)

    def run_without_callback(self) -> typing.Tuple[dict, np.ndarray]:
        self.target = image_funcs.image_to_tensor(self.target, (self.settings.render_width, self.settings.render_height))
        params, loss_hist = self._run_gd()
        self._last_params = params
        return params, loss_hist

    def restore_params(self):
        for key in self._last_params:
            self._last_params[key].restore_value()

    def _run_gd(self) -> typing.Tuple[dict, np.ndarray]:
        lr = self.settings.learning_rate
        max_iter = self.settings.max_iter
        early_stopping_thresh = self.settings.early_stopping_thresh
        loss_func = self.settings.loss_func
        width, height = self.settings.render_width, self.settings.render_height

        _, params_dict = self.out_node.render(width, height, retain_graph=False)
        for _, p in params_dict.items():
            p.save_value()
        args_list = [params_dict[k].tensor() for k in params_dict]  # Convert to list of tensors

        for p in args_list:
            p.requires_grad = True

        loss_hist = np.empty(max_iter, dtype=np.float32)
        optimizer = self.settings.optimizer(args_list, lr=lr)

        i = 0
        while i < max_iter:
            if self._stop:
                return params_dict, loss_hist

            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                start = time.time()
                render, _ = self.out_node.render(width, height, retain_graph=True)
                loss = loss_func(render, self.target)
                new_loss_np = loss.detach().clone().numpy()
                loss_hist[i] = new_loss_np
                props = {'iter': i, 'loss': new_loss_np, 'loss_hist': loss_hist[:i + 1], 'learning_rate': lr,
                         'params': {k: params_dict[k].get_value() for k in params_dict}, 'iter_time': 0.0, 'render': render}

                # We need to break here, otherwise the parameters will change when we call optimizer.step()
                if loss <= early_stopping_thresh:
                    self.iteration_done.emit(props)
                    break

                loss.backward(retain_graph=False, create_graph=False)

                optimizer.step()

            props['iter_time'] = time.time() - start
            self.iteration_done.emit(props)

            i += 1

        return params_dict, loss_hist[0:i+1]
