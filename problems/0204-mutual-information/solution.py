import numpy as np
import math

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	# Your code here
	p = np.array(joint_prob)
	X, Y = p.shape
	I_xy = 0
	p_x = np.sum(p, axis=1)
	p_y = np.sum(p, axis=0)

	for x in range(X):
		for y in range(Y):
			p_xy = p[x][y]
			if p_xy == 0:
				# convention that 0 * log 0 = 0
				continue
			I_xy += p_xy * math.log(p_xy/(p_x[x]*p_y[y]))

	return I_xy