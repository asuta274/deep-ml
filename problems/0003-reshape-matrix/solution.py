import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	# a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
	# [[1, 2], [3, 4], [5, 6], [7, 8]]

	m = len(a)
	assert m > 0
	n = len(a[0])
	assert n > 0
	m2, n2 = new_shape
	if m * n != m2 * n2:
		return []

	flat_list = [item for row in a for item in row]

	result = []
	for r in range(m2):
		start_idx = r * n2
		end_idx = start_idx + n2
		result.append(flat_list[start_idx:end_idx])

	return result