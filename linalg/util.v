module linalg

import math

// assert_type: assert the type to be numerics (HS eq Num type)
fn assert_type[T](t T) {
	match typeof(t).name {
		'int', '[]int', '[][]int' {}
		'i64', '[]i64', '[][]i64' {}
		'f32', '[]f32', '[][]f32' {}
		'f64', '[]f64', '[][]f64' {}
		else {
			eprintln('Type not supported')
			exit(1)
		}
	}
}

// sum: sum all elements in an array of type T
fn sum[T](arr []T) T {
	mut sum := T(0)
	assert_type[T](arr[0])
	for element in arr {
		sum += element
	}
	return sum
}

// rotation in z
pub fn rotation_z(alpha f64) [][]f64 {
	cos := math.cos(alpha)
	sin := math.sin(alpha)
	return [
		[cos, -sin, 0],
		[sin, cos, 0],
		[0.0, 0, 1],
	]
}

// rotation in y
pub fn rotation_y(beta f64) [][]f64 {
	cos := math.cos(beta)
	sin := math.sin(beta)
	return [
		[cos, 0, sin],
		[0.0, 1, 0],
		[-sin, 0, cos],
	]
}

// rotation in x
pub fn rotation_x(gamma f64) [][]f64 {
	cos := math.cos(gamma)
	sin := math.sin(gamma)
	return [
		[1.0, 0, 0],
		[0.0, cos, -sin],
		[0.0, sin, cos],
	]
}

// rotation matrix in x y z
pub fn rotation(alpha f64, beta f64, gamma f64) [][]f64 {
	zz := rotation_z(alpha)
	yy := rotation_y(beta)
	xx := rotation_x(gamma)
	return matmul(matmul(zz, yy), xx)
}

// arange a vector form 0 to index
@[inline]
pub fn arange(end int) []f64 {
	return []f64{len: end, init: index}
}

// reshape a vector to a matrix
pub fn reshape[T](v []T, rows int, columns int) [][]T {
	assert rows * columns == v.len, 'Require all elements'
	mut matrix := [][]T{len: rows, init: []T{len: columns}}
	for i in 0 .. rows {
		for j in 0 .. columns {
			matrix[i][j] = v[i * rows + j]
		}
	}
	return matrix
}
