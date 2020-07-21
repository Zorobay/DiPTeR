import abc
import typing

import torch
from torch import Tensor
from torch.nn import Module, MSELoss
from torchvision import models, transforms as T

mse_loss = MSELoss(reduction="mean")
sse_loss = MSELoss(reduction="sum")


class Loss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, target: Tensor):
        assert target.shape == x.shape
        assert target.shape[2] == 3 and x.shape[2] == 3, "Images need to be on the format [W,H,3]"
        return self._loss(x, target)

    @abc.abstractmethod
    def _loss(self, x: Tensor, target: Tensor):
        pass


class NeuralLoss(Loss):

    def __init__(self, layers: typing.Iterable[int] = None, layer_weights: typing.Iterable[float] = None):
        super().__init__()
        if layers is None or layers == []:
            layers = [0]
        if layer_weights is None or layer_weights == []:
            layer_weights = [1]

        self._layer_indices = layers
        self._layer_weights = torch.tensor(layer_weights)[0:len(layers)]

        self.vgg = models.vgg19(pretrained=True, progress=True)
        self.modulelist = list(self.vgg.features.modules())

        for i, mod in enumerate(self.modulelist):
            if hasattr(mod, "inplace"):
                mod.inplace = False

            if isinstance(mod, torch.nn.MaxPool2d):
                self.modulelist[i] = torch.nn.AvgPool2d(kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding, ceil_mode=mod.ceil_mode)

        self._normalize = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __str__(self):
        return "Neural Loss (\n\tlayers:\n {}, \n\tweights: {}\n)".format(self._layer_indices, self._layer_weights)

    def _preprocess(self, x: Tensor):
        # Swap axes to get image on CxHxW form, which is required for Models in PyTorch, then Normalize to comply with VGG19 and add batch dimension
        return self._normalize(x.permute(2, 1, 0)).unsqueeze(0)

    def _gram_matrix(self, activation: Tensor, N: int, M: int):
        feature_map = activation.view(N, M)
        G = torch.mm(feature_map, feature_map.t())
        return G

    def _loss(self, x: Tensor, target: Tensor):
        assert x.shape == target.shape
        assert list(x.shape) == [224, 224, 3]

        x_ = self._preprocess(x)
        target_ = self._preprocess(target)
        E = []

        for i, layer in enumerate(self.modulelist[1:]):
            x_ = layer(x_)
            target_ = layer(target_)
            if i > self._layer_indices[-1]:
                break
            if i in self._layer_indices:
                N = x_.shape[1]
                M = x_.shape[2] * x_.shape[3]
                G1 = self._gram_matrix(x_, N, M)
                G2 = self._gram_matrix(target_, N, M)

                E.append(sse_loss(G1, G2) * (1 / (4. * (N ** 2) * (M ** 2))))

        L_tot = torch.sum(torch.stack(E) * self._layer_weights)
        return L_tot


class XSELoss(Loss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss_func = MSELoss(reduction=reduction)

    def _loss(self, x: Tensor, target: Tensor):
        return self.loss_func(x, target)


class SquaredBinLoss(Loss):

    def __init__(self, bin_size: int = 10):
        super().__init__()
        self.size = bin_size

    def _loss(self, x: Tensor, target: Tensor):
        input_bins = x.unfold(1, self.size, self.size).unfold(0, self.size, self.size).transpose(2, -1)
        input_bins = input_bins.reshape(input_bins.shape[0] * input_bins.shape[1], self.size, self.size, 3)
        target_bins = target.unfold(1, self.size, self.size).unfold(0, self.size, self.size).transpose(2, -1)
        target_bins = target_bins.reshape(target_bins.shape[0] * target_bins.shape[1], self.size, self.size, 3)
        input_means = torch.mean(input_bins, dim=(1, 2))
        target_means = torch.mean(target_bins, dim=(1, 2))
        return mse_loss(input_means, target_means)


class VerticalBinLoss(Loss):

    def __init__(self, bin_size: int = 20):
        super().__init__()
        self.bin_size = bin_size

    def _loss(self, x: Tensor, target: Tensor):
        assert target.shape[2] % self.bin_size == 0, "The number of columns in the image can not be evenly divided into bins of size {}!".format(
            self.bin_size)
        bins_truth = torch.stack(target.split(self.bin_size, dim=-1))
        bins_x = torch.stack(target.split(self.bin_size, dim=-1))
        std_truth, mean_truth = torch.std_mean(bins_truth, dim=(1, 2, 3))
        std_x, mean_x = torch.std_mean(bins_x, dim=(1, 2, 3))
        return mse_loss(mean_x, mean_truth)
