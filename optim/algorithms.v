module optim

import linalg

pub fn difference_quotient(f fn (f64) f64, x f64, h f64) f64 {
	assert h != 0,"h must be possitive"
	return (f(x+h)-f(x))/h
}

pub fn partial_difference_quotient(f fn([]f64) f64, v []f64, i int, h f64) f64 {
	assert h != 0, "h must be possitive"
	mut w := []f64{}
	for j, v_j in v {
		w << if j == i { h + v_j } else { v_j } 
	} 
	return (f(w) - f(v))/h
}

pub fn gradient(f fn([]f64) f64, v []f64, h f64) []f64 {
	mut g := []f64{}
	for i, _ in v {
		g << partial_difference_quotient(f, v, i, h)
	}
	return g
}

pub fn gradient_step(v []f64, gradient_vector []f64, step_size f64) []f64 {
	assert v.len == gradient_vector.len, "vector and gradient lengths should be equal"
	step := linalg.scalar_multiply(step_size, gradient_vector)
	return linalg.add(v, step)
}

pub fn sum_of_squares_gradient(v []f64) []f64 {
	return v.map(2*it)
}