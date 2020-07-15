import torch


def floor_to_int(x: torch.Tensor) -> torch.Tensor:
    """Floors the value of a Tensor and returns it as an IntegerTensor."""
    return torch.floor(x).int()
