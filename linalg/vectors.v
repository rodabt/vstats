module linalg

import math

// Adds two vectors `a` and `b`: `(a + b)`
pub fn add[T](v []T, w []T) []T {
	assert v.len == w.len, 'vectors must be the same length'
	return []T{len: v.len, init: T(v[index] + w[index])}
}

// Subtracts two vectors `a` and `b`: `(a - b)`
pub fn subtract[T](v []T, w []T) []T {
	assert v.len == w.len, 'vectors must be the same length'
	return []T{len: v.len, init: T(v[index] - w[index])}
}

// Sums a list of vectors
// Example: `vector_sum([[f64(1),2],[3,4]]) => [4.0, 6.0]`
pub fn vector_sum[T](vector_list [][]T) []T {
	assert vector_list.len > 0, 'no vectors provided'
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
pub fn scalar_multiply[T](c f64, v []T) []T {
	// return v.map(it * c)
	return []T{len: v.len, init: T(v[index] * c)}
}

// 1/n sum_j (v[j])
pub fn vector_mean[T](vector_list [][]T) []T {
	n := T(vector_list.len)
	return scalar_multiply[T](1.0 / f64(n), vector_sum(vector_list))
}

// Sum all elements in an array
pub fn sum[T](arr []T) T {
	mut sum := T(0)
	for element in arr {
		sum += element
	}
	return sum
}

// 1/n sum_j (v[j])
// Dot product of two vectors, v and w of math type f64
pub fn dot[T](v []T, w []T) T {
	assert v.len == w.len, 'vectors must be the same length'
	return sum([]T{len: v.len, init: T(v[index] * w[index])})
}

// [1,2,3]^2 = [1^2, 2^2, 3^2]
pub fn sum_of_squares[T](v []T) T {
	return dot(v, v)
}

// || [3,4] || = 5
pub fn magnitude[T](v []T) T {
	return T(math.sqrt(sum_of_squares(v)))
}

// sqrt[(v1-w1)^2 + (v2-w2)^2...]
pub fn squared_distance[T](v []T, w []T) T {
	return sum_of_squares(subtract(v, w))
}

// dist(v,w)
pub fn distance[T](v []T, w []T) T {
	return magnitude(subtract(v, w))
}
