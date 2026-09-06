import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	#features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
	#([0.4626, 0.4134, 0.6682], 0.3349)
	f_m = len(features)
	f_n = len(features[0])

	# calculate y = features dot weights + bias
	y = []
	for row in range(f_m):
		value = sum([features[row][idx] * weights[idx] for idx in range(f_n)])
		y.append(value + bias)

	y = [round(1/(1+math.exp(-value)), 4) for value in y]
	mse = sum([round((y[idx] -  labels[idx]) ** 2 /len(y), 4) for idx in range(len(y))])
	return y, mse