module linalg

fn test__add() {
	assert add([f64(1), 2, 3], [f64(4), 5, 6]) == [f64(5), 7, 9]
}

fn test__subtract() {
	assert subtract([f64(1), 2, 3], [f64(4), 5, 6]) == [f64(-3), -3, -3]
}

fn test__vector_sum() {
	assert vector_sum([[f64(1), 2], [f64(3), 4], [f64(5), 6]]) == [f64(9), 12]
}

fn test__flatten() {
	assert flatten[int]([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
	assert flatten[f64]([[1.0, 2, 3], [4.0, 5, 6]]) == [1.0, 2, 3, 4, 5, 6]
}

fn test__flatten_uneven() {
	assert flatten[int]([[1, 2, 3], [4, 0]]) == [1, 2, 3, 4, 0]
}

fn test__scalar_multiply() {
	assert scalar_multiply(f64(3), [f64(1), 6, 7, 8]) == [f64(3), 18, 21, 24]
}

fn test__vector_mean() {
	assert vector_mean([[f64(1), 2], [f64(3), 4], [f64(5), 6]]) == [f64(3), 4]
}

fn test__dot() {
	assert dot([f64(1), 2, 3], [f64(4), 5, 6]) == f64(32)
}

fn test__sum_of_squares() {
	assert sum_of_squares([f64(1), 2, 3]) == f64(14)
}

fn test__magnitude() {
	assert magnitude([f64(3), 4]) == f64(5)
}

fn test__arange() {
	assert arange(10) == [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test__reshape() {
	v := arange(9)
	matrix := [
		[0.0, 1, 2],
		[3.0, 4, 5],
		[6.0, 7, 8],
	]
	assert reshape[f64](v, 3, 3) == matrix
	unit_array := [1.0, 0, 0, 0.0, 1, 0, 0.0, 0, 1]
	assert reshape[f64](unit_array, 3, 3) == identity_matrix(3)
}
