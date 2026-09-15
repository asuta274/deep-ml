import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	assert n == 10
	for i in range(n):
		consumption_data[i] -= 10 * math.sin(2 * PI * (i+1) / 10)

	x = np.arange(1, 1 + n)
	y = np.array(consumption_data)

	a = (n * np.sum(np.multiply(x, y)) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
	b = (np.sum(y) - a * np.sum(x)) / n

	x_15 = 15
	y_15 = a * x_15 + b

	result = (y_15 + 10 * math.sin(2 * PI * x_15 / 10))
	result = math.ceil(result * 1.05)

	return result