import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    predicted_probs = np.clip(predicted_probs, a_min=epsilon, a_max=1-epsilon)
    log2_predict = -np.log(predicted_probs)
    label_log2_predict = np.multiply(true_labels, log2_predict)
    cross_entropy_loss = np.sum(label_log2_predict) / true_labels.shape[0]
    return cross_entropy_loss