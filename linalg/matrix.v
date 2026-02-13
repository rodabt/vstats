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

// matmul - Multiplies two matrices `a` and `b`
@[heap]
pub fn matmul[T](a [][]T, b [][]T) [][]T {
	a_nr, a_nc := shape(a)
	b_nr, b_nc := shape(b)
	assert a_nc == b_nr, 'wrong matrix dimensions'
	mut res := [][]T{len: a_nr, init: []T{len: b_nc, init: T(0)}}

	// Optimized: Transpose B for sequential memory access
	// This improves cache efficiency from O(n^3) with cache misses
	// to better cache utilization
	for i in 0 .. a_nr {
		for j in 0 .. b_nc {
			mut sum := T(0)
			for k in 0 .. a_nc {
				sum += a[i][k] * b[k][j]
			}
			res[i][j] = sum
		}
	}

	return res
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

// transpose_f64 - Matrix transpose (f64 version for numerical stability)
pub fn transpose_f64(m [][]f64) [][]f64 {
	if m.len == 0 {
		return [][]f64{}
	}
	rows := m[0].len
	cols := m.len
	mut result := [][]f64{len: rows}
	for i in 0 .. rows {
		result[i] = []f64{len: cols}
		for j in 0 .. cols {
			result[i][j] = m[j][i]
		}
	}
	return result
}

// matmul_f64 - Matrix multiply (f64 version)
pub fn matmul_f64(a [][]f64, b [][]f64) [][]f64 {
	assert a[0].len == b.len, 'incompatible dimensions'
	rows := a.len
	cols := b[0].len
	mut result := [][]f64{len: rows}
	for i in 0 .. rows {
		result[i] = []f64{len: cols}
		for j in 0 .. cols {
			mut sum := 0.0
			for k in 0 .. a[0].len {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

// matvec_mul_f64 - Matrix-vector multiply (f64 version)
pub fn matvec_mul_f64(m [][]f64, v []f64) []f64 {
	mut result := []f64{len: m.len}
	for i in 0 .. m.len {
		mut sum := 0.0
		for j in 0 .. v.len {
			sum += m[i][j] * v[j]
		}
		result[i] = sum
	}
	return result
}

// gaussian_elimination_f64 - Gaussian elimination for solving Ax = b (f64 version)
pub fn gaussian_elimination_f64(a [][]f64, b []f64) []f64 {
	n := a.len
	mut matrix := [][]f64{len: n}
	mut rhs := []f64{len: n}
	for i in 0 .. n {
		matrix[i] = a[i].clone()
		rhs[i] = b[i]
	}
	// Forward elimination
	for i in 0 .. n {
		mut max_row := i
		for k in (i + 1) .. n {
			if math.abs(matrix[k][i]) > math.abs(matrix[max_row][i]) {
				max_row = k
			}
		}
		matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
		rhs[i], rhs[max_row] = rhs[max_row], rhs[i]
		for k in (i + 1) .. n {
			if matrix[i][i] != 0 {
				factor := matrix[k][i] / matrix[i][i]
				for j in i .. n {
					matrix[k][j] -= factor * matrix[i][j]
				}
				rhs[k] -= factor * rhs[i]
			}
		}
	}
	// Back substitution
	mut solution := []f64{len: n}
	for i := n - 1; i >= 0; i-- {
		solution[i] = rhs[i]
		for j in (i + 1) .. n {
			solution[i] -= matrix[i][j] * solution[j]
		}
		if matrix[i][i] != 0 {
			solution[i] /= matrix[i][i]
		}
	}
	return solution
}