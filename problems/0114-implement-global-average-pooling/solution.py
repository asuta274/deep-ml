import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	w, h, c = x.shape
	x = x.reshape(w * h, c)
	result = np.average(x, axis=0)
	return result