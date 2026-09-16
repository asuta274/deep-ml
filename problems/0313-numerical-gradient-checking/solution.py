import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    n = len(x)
    numerical_grad, relative_error = [], 0
    for i in range(n):
        x_e_left = np.copy(x)
        x_e_left[i] += epsilon
        x_e_right = np.copy(x)
        x_e_right[i] -= epsilon
        y = (f(x_e_left) - f(x_e_right))/ (2 * epsilon)
        numerical_grad.append(y)
    
    numerical_grad = np.array(numerical_grad)
    delta = numerical_grad - analytical_grad
    delta_norm = np.linalg.norm(delta)
    numerical_grad_norm = np.linalg.norm(numerical_grad)
    analytical_grad_norm = np.linalg.norm(analytical_grad)
    if numerical_grad_norm + analytical_grad_norm == 0:
        relative_error = 0
    else:
        relative_error = delta_norm / (numerical_grad_norm + analytical_grad_norm)

    return numerical_grad, relative_error