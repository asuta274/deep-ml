import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    def f(x):
        return x ** 2 + 3 * x + 2
    
    x = torch.tensor(x_value, dtype=torch.bfloat16, requires_grad=True)
    y = f(x)
    y.backward()

    return x.grad.item()
