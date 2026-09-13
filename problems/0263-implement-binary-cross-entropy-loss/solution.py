def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here
	import numpy as np
	y_hat = np.array(y_true)
	y = np.array(y_pred)
	y = np.clip(y, a_min = epsilon, a_max = 1-epsilon)
	bce = -np.sum(np.multiply(y_hat, np.log(y)) + np.multiply(1-y_hat, np.log(1-y)))/len(y)
	return bce