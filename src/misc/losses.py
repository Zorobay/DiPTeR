import torch
from PIL import Image
from torchvision import transforms

mse = torch.nn.MSELoss(reduction='mean')


def squared_bin_loss(render, truth):
    size = 10
    r_bins = torch.stack(torch.stack(render.split(size, dim=1)).split(size, dim=-1))
    t_bins = torch.stack(torch.stack(truth.split(size, dim=1)).split(size, dim=-1))
    r_mean = torch.mean(r_bins, dim=(2, 3, 4))
    t_mean = torch.mean(t_bins, dim=(2, 3, 4))
    return mse(r_mean, t_mean)

def mean_squared_error(render, truth):
    return mse(render, truth)


class VerticalBinLoss(torch.nn.Module):

    def __init__(self, bin_size: int = 20):
        super().__init__()
        self.bin_size = bin_size
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, truth, x):
        assert truth.shape == x.shape
        assert truth.shape[0] == 3 and x.shape[0] == 3, "Images need to be on the format [3,W,H]"
        assert truth.shape[2] % self.bin_size == 0, "The number of columns in the image can not be evenly divided into bins of size {}!".format(
            self.bin_size)
        bins_truth = torch.stack(truth.split(self.bin_size, dim=-1))
        bins_x = torch.stack(x.split(self.bin_size, dim=-1))
        std_truth, mean_truth = torch.std_mean(bins_truth, dim=(1, 2, 3))
        std_x, mean_x = torch.std_mean(bins_x, dim=(1, 2, 3))
        return self.mse_loss(mean_x, mean_truth)


if __name__ == '__main__':
    size = (100, 100)
    tr = transforms.ToTensor()
    truth = tr(Image.open("default.png").resize(size).convert("RGB"))
    x = tr(Image.open("mortar_scale=0.6.png").resize(size).convert("RGB"))
    loss = VerticalBinLoss(5)

    print("Loss: {}".format(loss(truth, x)))
