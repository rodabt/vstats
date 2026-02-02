import linalg

fn test__shape() {
	n_rows, n_cols := linalg.shape([[1, 2, 3], [4, 5, 6]])
	assert n_rows == 2
	assert n_cols == 3
}

fn test__identity() {
	assert linalg.identity_matrix[int](3) == [[1,0,0], [0, 1, 0], [0, 0, 1]]
}
