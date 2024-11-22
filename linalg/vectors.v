module linalg

import math
import arrays

// Adds two vectors `a` and `b`: `(a + b)`
pub fn add(v []f64, w []f64) []f64 {
	assert v.len == w.len, 'vectors must be the same length'
	mut res := []f64{len: v.len}

	arrays.each_indexed[f64](v, fn [mut res, w] (i int, e f64) {
		res[i] = e + w[i]
	})

	return res
}

// Subtracts two vectors `a` and `b`: `(a - b)`
pub fn subtract(v []f64, w []f64) []f64 {
	assert v.len == w.len, 'vectors must be the same length'
	mut res := []f64{len: v.len}

	arrays.each_indexed[f64](v, fn [mut res, w] (i int, e f64) {
		res[i] = e - w[i]
	})

	return res
}

// Sums a list of vectors
// Example: `vector_sum([[f64(1),2],[3,4]]) => [4.0, 6.0]`
pub fn vector_sum(vector_list [][]f64) []f64 {
	assert vector_list.len > 0, 'no vectors provided'

	num_elements := vector_list[0].len
	assert vector_list.all(it.len == num_elements), 'vectors must be the same length'

	res := arrays.reduce(vector_list, fn (v1 []f64, v2 []f64) []f64 {
		return add(v1, v2)
	}) or { []f64{} }
	return res
}

// Multiplies an scalar value `c` to each element of a vector `v`
pub fn scalar_multiply(c f64, v []f64) []f64 {
	return v.map(it * c)
}

// 1/n sum_j (v[j])
pub fn vector_mean(vector_list [][]f64) []f64 {
	n := f64(vector_list.len)
	return scalar_multiply(1.0 / n, vector_sum(vector_list))
}

// 1/n sum_j (v[j])
pub fn dot(v []f64, w []f64) f64 {
	assert v.len == w.len, 'vectors must be the same length'
	res := arrays.map_indexed[f64, f64](v, fn [w] (idx int, elem f64) f64 {
		return elem * w[idx]
	})
	return arrays.reduce(res, fn (x f64, y f64) f64 {
		return x + y
	}) or { f64(0) }
}

// [1,2,3]^2 = [1^2, 2^2, 3^2]
pub fn sum_of_squares(v []f64) f64 {
	return dot(v, v)
}

// || [3,4] || = 5
pub fn magnitude(v []f64) f64 {
	return math.sqrt(sum_of_squares(v))
}

// sqrt[(v1-w1)^2 + (v2-w2)^2...]
pub fn squared_distance(v []f64, w []f64) f64 {
	return sum_of_squares(subtract(v, w))
}

// dist(v,w)
pub fn distance(v []f64, w []f64) f64 {
	return magnitude(subtract(v, w))
}
