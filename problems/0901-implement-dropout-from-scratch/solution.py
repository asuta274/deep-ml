import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if not training or p == 0:
        return x

    mask = torch.rand_like(x) >= p
    return x * mask / (1 - p)
