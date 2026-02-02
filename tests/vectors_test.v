import linalg

fn test__add() {
	assert linalg.add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
	assert linalg.add([1.0, 2, 3], [4.0, 5, 6]) == [5.0, 7, 9]
}

fn test__subtract() {
	assert linalg.subtract([1, 2, 3], [4, 5, 6]) == [-3, -3, -3]
	assert linalg.subtract([1.0, 2, 3], [4.0, 5, 6]) == [-3.0, -3, -3]
}

fn test__vector_sum() {
	assert linalg.vector_sum([[1, 2], [3, 4], [5, 6]]) == [9, 12]
	assert linalg.vector_sum([[1.0, 2], [3.0, 4], [5.0, 6]]) == [9.0, 12]
}

fn test__flatten() {
	assert linalg.flatten([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
	assert linalg.flatten([[1.0, 2, 3], [4.0, 5, 6]]) == [1.0, 2, 3, 4.0, 5, 6]
}

fn test__flatten_uneven() {
	assert linalg.flatten([[1.0, 2, 3], [4.0, 0]]) == [1.0, 2, 3, 4, 0]
}

fn test__scalar_multiply() {
	assert linalg.scalar_multiply(3, [1, 6, 7, 8]) == [3, 18, 21, 24]
}

fn test__vector_mean() {
	assert linalg.vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
}

fn test__dot() {
	assert linalg.dot([1, 2, 3], [4, 5, 6]) == 32
}

fn test__sum_of_squares() {
	assert linalg.sum_of_squares([1, 2, 3]) == 14
}

fn test__magnitude() {
	assert linalg.magnitude([3, 4]) == 5
}

fn test__arange() {
	assert linalg.arange[int](10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test__reshape() {
	v := linalg.arange[int](9)
	matrix := [
		[0, 1, 2],
		[3, 4, 5],
		[6, 7, 8],
	]
	assert linalg.reshape(v, 3, 3) == matrix
	unit_array := [1.0, 0, 0, 0.0, 1, 0, 0.0, 0, 1]
	assert linalg.reshape[f64](unit_array, 3, 3) ==  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
