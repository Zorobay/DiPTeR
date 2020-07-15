import torch


def mod289(x):
    return x - torch.floor(x * (1.0 / 289.0)) * 289.0


def permute(x):
    return mod289(((x * 34.0) + 1.0) * x)


def taylorInvSqrt(r):
    return 1.79284291400159 - 0.85373472095314 * r


def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
