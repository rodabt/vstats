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
