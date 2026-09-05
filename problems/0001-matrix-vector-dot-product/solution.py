def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	# a = [[1, 2], [2, 4]], b = [1, 2]
	# [5, 10]
	c = len(a)
	assert c > 0
	r = len(a[0])
	assert r > 0
	if len(b) != r:
		return -1

	result = [0] * c
	for col in range(c):
		result[col] = sum(e_a * e_b for e_a, e_b in zip(a[col], b))

	return result