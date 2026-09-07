import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    n, c, h, w = x.shape
    mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=[0, 2, 3], keepdim=True)
    std = torch.sqrt(var + eps)

    gamma = gamma.view(1, c, 1, 1)
    beta = beta.view(1, c, 1, 1)

    x_hat = (x - mean) / std
    out = x_hat * gamma + beta

    return out
