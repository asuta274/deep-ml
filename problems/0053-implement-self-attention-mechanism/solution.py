import numpy as np
from scipy.special import softmax

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    seq_len, d_k = Q.shape
    
    K_T = np.transpose(K)
    A = np.dot(Q, K_T)
    A = A / np.sqrt(d_k)
    A = softmax(A, axis=1)
    A = np.dot(A, V)
    return A
    
