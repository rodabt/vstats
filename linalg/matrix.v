module linalg

import arrays
import math

/*
Maybe do something like this for matrices?
pub struct Matrix[T] {
	rows    int
	columns int
	data    []T
}

pub fn (m Matrix[T]) str() string {
	mut str := ''
	for i in 0 .. m.rows {
		for j in 0 .. m.columns {
			str += m.data[i * m.rows + j].str()
			if j < m.columns - 2 {
				str += ' '
			}
		}
		if i < m.rows - 2 {
			str += '\n'
		}
	}
	return str
}

pub fn (m Matrix[T]) shape() (int, int) {
	return m.rows, m.columns
}
*/

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
	// return arrays.map_indexed[[]f64, f64](a, fn [j] (idx int, elem []f64) f64 {
	// 	return elem[j]
	// })
	rows, cols := shape[T](a)
	mut vector := []T{len: cols}
	for i in 0 .. rows {
		vector[i] = a[i][j]
	}
	return vector
}

// Creates a matrix according to a formula of row and column position
pub fn make_matrix[T](num_rows int, num_cols int, op fn (int, int) T) [][]T {
	mut res := [][]f64{len: num_rows, init: []f64{len: num_cols}}
	for i in 0 .. num_rows {
		for j in 0 .. num_cols {
			res[i][j] = op(i, j)
		}
	}
	return res
}

// Creates an identity matrix of size `n`
pub fn identity_matrix(n int) [][]f64 {
	return make_matrix[f64](n, n, fn (i int, j int) f64 {
		return if i == j { f64(1) } else { f64(0) }
	})
}

// Multiplies two matrixes `a` and `b`
pub fn matmul[T](a [][]T, b [][]T) [][]T {
	a_nr, a_nc := shape(a)
	b_nr, b_nc := shape(b)
	assert a_nc == b_nr, 'wrong matrix dimentions'
	mut res := [][]T{len: a_nr, init: []T{len: b_nc}}
	for i in 0 .. a_nr {
		for j in 0 .. b_nc {
			row := get_row[T](a, i)
			col := get_column[T](b, j)
			res[i][j] = dot[T](row, col)
		}
	}
	return res
}
