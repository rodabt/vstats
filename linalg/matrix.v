module linalg

import arrays

// Returns a tuple of the number of rows and columns of a matrix
pub fn shape(a [][]f64) (int, int) {
	num_rows := a.len
	num_cols := if a.len > 0 { a[0].len } else { 0 }
	return num_rows, num_cols
}

// Returns the i-th row of a matrix as a vector 
pub fn get_row(a [][]f64, i int) []f64 {
	return a[i]
}

// Returns the j-th column of a matrix as a vector
pub fn get_column(a [][]f64, j int) []f64 {
	return arrays.map_indexed[[]f64, f64](a, fn [j] (idx int, elem []f64) f64 {
		return elem[j]
	})
}

// Creates a matrix according to a formula of row and column position
pub fn make_matrix(num_rows int, num_cols int, op fn (int, int) f64) [][]f64 {
	mut res := [][]f64{len: num_rows, init: []f64{len: num_cols}}
	for i in 0..num_rows {
		for j in 0..num_cols {
			res[i][j] = op(i, j)
		}
	}
	return res
}

// Creates an identity matrix of size `n`
pub fn identity_matrix(n int) [][]f64 {
	return make_matrix(n, n, fn (i int, j int) f64 {
		return if i == j { f64(1) } else { f64(0) }
	})
}

// Multiplies two matrixes `a` and `b` 
pub fn matmul(a [][]f64, b [][]f64) [][]f64 {
	a_nr, a_nc := shape(a)
	b_nr, b_nc := shape(b)
	assert a_nc == b_nr, "wrong matrix dimentions"
	mut res := [][]f64{len: a_nr, init: []f64{len: b_nc}}
	for i in 0..a_nr {
		for j in 0..b_nc {
			row := get_row(a, i)
			col := get_column(b, j)
			res[i][j] = dot(row,col)
		}
	}
	return res
}