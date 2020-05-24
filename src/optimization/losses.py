import abc

import torch
from torch import Tensor
from torch.nn import Module


class Loss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        assert target.shape == input.shape
        assert target.shape[2] == 3 and input.shape[2] == 3, "Images need to be on the format [W,H,3]"
        return self._loss(target, input)

    @abc.abstractmethod
    def _loss(self, truth: Tensor, x: Tensor):
        pass


class MSELoss(Loss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self._mse = torch.nn.MSELoss(reduction=reduction)

    def _loss(self, input: Tensor, target: Tensor):
        return self._mse(input, target)


class SquaredBinLoss(Loss):

    def __init__(self):
        super().__init__()
        self._mse = torch.nn.MSELoss(reduction='mean')

    def _loss(self, input: Tensor, target: Tensor):
        size = 10
        input_bins = torch.stack(torch.stack(input.split(size, dim=1)).split(size, dim=-1))
        target_bins = torch.stack(torch.stack(target.split(size, dim=1)).split(size, dim=-1))
        input_means = torch.mean(input_bins, dim=(2, 3, 4))
        target_mean = torch.mean(target_bins, dim=(2, 3, 4))
        return self._mse(input_means, target_mean)


class VerticalBinLoss(Loss):

    def __init__(self, bin_size: int = 20):
        super().__init__()
        self.bin_size = bin_size
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def _loss(self, input: Tensor, target: Tensor):
        assert target.shape[2] % self.bin_size == 0, "The number of columns in the image can not be evenly divided into bins of size {}!".format(
            self.bin_size)
        bins_truth = torch.stack(target.split(self.bin_size, dim=-1))
        bins_x = torch.stack(input.split(self.bin_size, dim=-1))
        std_truth, mean_truth = torch.std_mean(bins_truth, dim=(1, 2, 3))
        std_x, mean_x = torch.std_mean(bins_x, dim=(1, 2, 3))
        return self.mse_loss(mean_x, mean_truth)
