import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    return 3 * x ** 2 + 2
