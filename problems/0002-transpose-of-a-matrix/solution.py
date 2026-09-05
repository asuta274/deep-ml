def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    n_row = len(a)
    assert n_row > 0
    n_col = len(a[0])
    assert n_col > 0

    result = [[0] * n_row for _ in range(n_col)]

    #a = [[1, 2, 3], [4, 5, 6]]
    #[[1, 4], [2, 5], [3, 6]]
    for r in range(n_row):
        for c in range(n_col):
            result[c][r] = a[r][c]

    return result 