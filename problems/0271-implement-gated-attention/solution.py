import numpy as np

def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    d_model, d_k = W_q.shape
    
    q = np.dot(X, W_q)
    k = np.dot(X, W_k)
    v = np.dot(X, W_v)
    
    k_t = np.transpose(k)
    A = np.dot(q, k_t)
    A = A / np.sqrt(d_k)
    from scipy.special import softmax
    A = softmax(A, axis=1)
    A = np.dot(A, v)

    G = np.dot(X, W_g)
    G = 1 / (1 + np.exp(-G))
    return np.multiply(G, A)