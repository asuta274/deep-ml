import numpy as np
from scipy.special import softmax

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_model, d_k = W_Q.shape

    if x_new.ndim == 1:
        x_new = x_new[np.newaxis, :]

    q = np.dot(x_new, W_Q)
    k = np.dot(x_new, W_K)
    v = np.dot(x_new, W_V)
    
    if cache:
        K_cache, V_cache = cache
        K_cache = np.vstack([K_cache, k])
        V_cache = np.vstack([V_cache, v])
    else:
        K_cache, V_cache = k, v

    k = K_cache
    v = V_cache

    A = np.dot(q, np.transpose(k)) / np.sqrt(d_k)
    A = softmax(A)
    A = np.dot(A, v)
    return A[0], (K_cache, V_cache)
