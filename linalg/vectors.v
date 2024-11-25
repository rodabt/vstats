module linalg

import math
import arrays

/*
Maybe do something like this for vectors?
pub struct Vector[T] {
	row bool // Row or column vector, then we could have v.dot(v) which only works if they are different types,
	// e.g Einstein notation v^i v_i instead v_i v_i which is not technically correct. also makes it easy to deal with tensors if we add higher dimensions.
	elements []T
}
*/

// Adds two vectors `a` and `b`: `(a + b)`
pub fn add[T](v []T, w []T) []T {
	assert v.len == w.len, 'vectors must be the same length'
	assert_type(v[0])
	return []T{len: v.len, init: T(v[index] + w[index])}
}

// Subtracts two vectors `a` and `b`: `(a - b)`
pub fn subtract[T](v []T, w []T) []T {
	assert v.len == w.len, 'vectors must be the same length'
	assert_type(v[0])
	return []T{len: v.len, init: T(v[index] - w[index])}
}

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
	num_elements := number_of_elements[T](m)
	assert_type(m[0][0])

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

// Sums a list of vectors
// Example: `vector_sum([[f64(1),2],[3,4]]) => [4.0, 6.0]`
pub fn vector_sum[T](vector_list [][]T) []T {
	assert vector_list.len > 0, 'no vectors provided'
	assert_type[T](vector_list[0][0])
	num_elements := vector_list[0].len
	assert vector_list.all(it.len == num_elements), 'vectors must be the same length'
	mut res := []T{len: vector_list[0].len}
	for i in 0 .. vector_list.len {
		for j in 0 .. vector_list[i].len { // should be fine
			res[j] += vector_list[i][j]
		}
	}
	return res
}

// Multiplies an scalar value `c` to each element of a vector `v`
pub fn scalar_multiply[T, U](c T, v []U) []U {
	// return v.map(it * c)
	assert_type[T](c)
	assert_type[U](v[0])
	return []U{len: v.len, init: U(v[index] * c)}
}

// 1/n sum_j (v[j])
pub fn vector_mean[T](vector_list [][]T) []T {
	n := f64(vector_list.len)
	assert_type[T](vector_list[0][0])
	return scalar_multiply(1.0 / n, vector_sum(vector_list))
}

// 1/n sum_j (v[j])
// Dot product of two vectors, v and w of math type T
pub fn dot[T](v []T, w []T) T {
	assert v.len == w.len, 'vectors must be the same length'
	assert_type(v[0])
	return sum[T]([]T{len: v.len, init: T(v[index] * w[index])})
}

// [1,2,3]^2 = [1^2, 2^2, 3^2]
pub fn sum_of_squares[T](v []T) T {
	assert_type[T](v[0])
	return dot[T](v, v)
}

// || [3,4] || = 5
pub fn magnitude[T](v []T) T {
	assert_type[T](v[0])
	return math.sqrt(sum_of_squares[T](v))
}

// sqrt[(v1-w1)^2 + (v2-w2)^2...]
pub fn squared_distance[T](v []T, w []T) T {
	assert_type[T](v[0])
	return sum_of_squares[T](subtract[T](v, w))
}

// dist(v,w)
pub fn distance[T](v []T, w []T) T {
	assert_type[T](v[0])
	return magnitude(subtract(v, w))
}
