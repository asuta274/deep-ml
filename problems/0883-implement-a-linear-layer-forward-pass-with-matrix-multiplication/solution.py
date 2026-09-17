import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: implement y = x W^T + b using PyTorch ops
    W = W.transpose(0, 1)
    print(x.shape)
    print(W.shape)
    result = torch.matmul(x, W) + b
    print(result)
    return result
