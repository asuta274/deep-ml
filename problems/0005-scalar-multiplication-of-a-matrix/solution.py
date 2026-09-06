def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	n_row = len(matrix)
	n_col = len(matrix[0])
	for row in range(n_row):
		for col in range(n_col):
			matrix[row][col] *= scalar
	
	return matrix