module linalg
import math

// count the number of elements in a matrix
fn number_of_elements[T](m [][]T) int {
	mut elements := 0
	for i in 0 .. m.len {
		for _ in 0 .. m[i].len {
			elements++
		}
	}
	return elements
}

// flatten a matrix to an array.
pub fn flatten[T](m [][]T) []T {
	num_elements := number_of_elements(m)

	mut list := []T{len: num_elements}
	mut done := 0
	for i in 0 .. m.len {
		v := m[i]
		for j in 0 .. v.len {
			list[done + j] = m[i][j] // If square, we could have done = i * m[0].len instead.
		}
		done += m[i].len
	}
	return list
}

// Returns a tuple of the number of rows and columns of a matrix
pub fn shape[T](a [][]T) (int, int) {
	num_rows := a.len
	num_cols := if a.len > 0 { a[0].len } else { 0 }
	return num_rows, num_cols
}

// Returns the i-th row of a matrix as a vector
pub fn get_row[T](a [][]T, i int) []T {
	return a[i]
}

// Returns the j-th column of a matrix as a vector
pub fn get_column[T](a [][]T, j int) []T {
	rows, cols := shape(a)
	mut vector := []T{len: cols}
	for i in 0 .. rows {
		vector[i] = a[i][j]
	}
	return vector
}

// Creates a matrix according to a formula of row and column position
@[heap]
pub fn make_matrix[T](num_rows int, num_cols int, op fn (int, int) T) [][]T {
	mut res := [][]T{len: num_rows, init: []T{len: num_cols}}
	for i in 0 .. num_rows {
		for j in 0 .. num_cols {
			res[i][j] = op(i, j)
		}
	}
	return res
}

// Creates an identity matrix of size `n`
@[heap]
pub fn identity_matrix[T](n int) [][]T {
	return make_matrix(n, n, fn[T](i int, j int) T {
		return if i == j { 1 } else { 0 }
	})
}

// Multiplies two matrixes `a` and `b`
@[heap]
pub fn matmul[T](a [][]T, b [][]T) [][]T {
	a_nr, a_nc := shape(a)
	b_nr, b_nc := shape(b)
	assert a_nc == b_nr, 'wrong matrix dimentions'
	mut res := [][]T{len: a_nr, init: []T{len: b_nc}}
	for i in 0 .. a_nr {
		for j in 0 .. b_nc {
			row := get_row(a, i)
			col := get_column(b, j)
			res[i][j] = dot(row, col)
		}
	}
	return res
}

// sum: sum all elements in an array
fn sum[T](arr []T) T {
	mut sum := T(0)
	for element in arr {
		sum += element
	}
	return sum
}

// rotation in z
@[heap]
pub fn rotation_z[T](alpha T) [][]T {
	cos := math.cos(alpha)
	sin := math.sin(alpha)
	return [
		[cos, -sin, 0],
		[sin, cos, 0],
		[0.0, 0, 1],
	]
}

// rotation in y`
@[heap]
pub fn rotation_y[T](beta T) [][]T {
	cos := math.cos(beta)
	sin := math.sin(beta)
	return [
		[cos, 0, sin],
		[0.0, 1, 0],
		[-sin, 0, cos],
	]
}

// rotation in x`
@[heap]
pub fn rotation_x[T](gamma T) [][]T {
	cos := math.cos(gamma)
	sin := math.sin(gamma)
	return [
		[1.0, 0, 0],
		[0.0, cos, -sin],
		[0.0, sin, cos],
	]
}

// rotation matrix in x y z`
@[heap]
pub fn rotation[T](alpha T, beta T, gamma T) [][]T {
	zz := rotation_z(alpha)
	yy := rotation_y(beta)
	xx := rotation_x(gamma)
	return matmul(matmul(zz, yy), xx)
}

// arange a vector form 0 to index
@[inline]
pub fn arange[T](end int) []T {
	return []T{len: end, init: T(index)}
}

// reshape a vector to a matrix
@[heap]
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