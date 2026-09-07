import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	updated_weights, updated_bias, mse_values = initial_weights, initial_bias, []
	n = len(labels)
	for epoch in range(epochs): 
		z = np.dot(features, updated_weights) + updated_bias
		sigma = 1 / (1 + np.exp(-z))
		mse = np.sum((sigma - labels) ** 2) / n
		mse_values.append(round(mse, 4))

		delta = (sigma - labels) * (sigma * (1 - sigma)) 
		dw = (2 / n) * np.dot(features.T, delta)
		db = (2 / n) * np.sum(delta)
		
		updated_weights = updated_weights - learning_rate * dw
		updated_bias = updated_bias - learning_rate * db

	return updated_weights, updated_bias, mse_values