def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	n_row = len(matrix)
	n_col = len(matrix[0])
	if mode == 'row':
		return [sum(row) / len(row) for row in matrix]
	elif mode == 'column':
		result = []
		for col_idx in range(n_col):
			total = sum([matrix[row_idx][col_idx] for row_idx in range(n_row)])
			result.append(total/n_row)
		return result
	else:
		raise Exception("Mode not supported")
	
	return None