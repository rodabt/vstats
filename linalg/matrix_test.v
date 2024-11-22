module linalg

fn test__shape() {
	n_rows, n_cols := shape([[f64(1), 2, 3], [f64(4), 5, 6]])
	assert n_rows == 2
	assert n_cols == 3
}

fn test__identity() {
	assert identity_matrix(3) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0]]
}
