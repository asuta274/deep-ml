import math

def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	# Your code here
	return round(x * (1 / (1 + math.exp(-x))), 4)