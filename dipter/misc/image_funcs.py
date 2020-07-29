import typing

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def image_to_tensor(image: Image, size: typing.Tuple[int, int] = None) -> torch.Tensor:
    """
    Converts a pillow image (that is **column major**) to a pytorch Tensor that is also **column major** on the format WxHxC
    :param image: Pillow Image object
    :param size: if supplied, the image will be resized to this size.
    :return: a torch Tensor on the format WxHxC
    """
    image = image.convert("RGB")

    if size:
        size = np.asarray(size, dtype=np.int32)
        image = image.resize(size)

    t = transforms.Compose([
        transforms.ToTensor()
    ])

    tensor = t(image).float().flip(1).transpose(0,-1)

    return tensor
