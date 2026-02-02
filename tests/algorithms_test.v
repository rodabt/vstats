import optim
import linalg
import math

fn test__difference_quotient() {
	// f(x) = x^2, derivative at x=2 should be ~4
	f := fn (x f64) f64 { return x * x }
	result := optim.difference_quotient(f, 2.0, 0.001)
	assert math.abs(result - 4.0) < 0.01, "derivative of x^2 at x=2 should be ~4"
}

fn test__difference_quotient_sine() {
	// f(x) = sin(x), derivative at 0 should be ~1
	f := fn (x f64) f64 { return math.sin(x) }
	result := optim.difference_quotient(f, 0.0, 0.001)
	assert math.abs(result - 1.0) < 0.01, "derivative of sin(x) at x=0 should be ~1"
}

fn test__partial_difference_quotient() {
	// f(x, y) = x^2 + y^2
	f := fn (v []f64) f64 { return v[0]*v[0] + v[1]*v[1] }
	v := [2.0, 3.0]
	
	// Partial derivative w.r.t. x at (2,3) should be ~4
	partial_x := optim.partial_difference_quotient(f, v, 0, 0.001)
	assert math.abs(partial_x - 4.0) < 0.01, "partial w.r.t. x should be ~4"
	
	// Partial derivative w.r.t. y at (2,3) should be ~6
	partial_y := optim.partial_difference_quotient(f, v, 1, 0.001)
	assert math.abs(partial_y - 6.0) < 0.01, "partial w.r.t. y should be ~6"
}

fn test__gradient() {
	// f(x, y) = x^2 + y^2
	f := fn (v []f64) f64 { return v[0]*v[0] + v[1]*v[1] }
	v := [2.0, 3.0]
	
	grad := optim.gradient(f, v, 0.001)
	assert grad.len == 2
	assert math.abs(grad[0] - 4.0) < 0.02, "gradient x component should be ~4"
	assert math.abs(grad[1] - 6.0) < 0.02, "gradient y component should be ~6"
}

fn test__gradient_step() {
	v := [1.0, 2.0, 3.0]
	grad := [0.1, 0.2, 0.3]
	step_size := 0.5
	
	result := optim.gradient_step(v, grad, step_size)
	expected := [1.05, 2.1, 3.15]
	
	for i := 0; i < result.len; i++ {
		assert math.abs(result[i] - expected[i]) < 0.01, "result[${i}] = ${result[i]}, expected ${expected[i]}"
	}
}

fn test__gradient_step_descent() {
	// Test that gradient step actually moves in negative direction
	v := [5.0, 5.0]
	grad := [1.0, 1.0]
	step_size := -0.1  // negative for descent
	
	result := optim.gradient_step(v, grad, step_size)
	
	// Should move closer to origin
	orig_dist := linalg.magnitude(v)
	new_dist := linalg.magnitude(result)
	assert new_dist < orig_dist, "gradient descent should move closer to minimum"
}

fn test__sum_of_squares_gradient() {
	v := [1.0, 2.0, 3.0]
	grad := optim.sum_of_squares_gradient(v)
	
	expected := [2.0, 4.0, 6.0]
	for i := 0; i < grad.len; i++ {
		assert math.abs(grad[i] - expected[i]) < 0.001
	}
}

fn test__sum_of_squares_gradient_zero() {
	v := [0.0, 0.0, 0.0]
	grad := optim.sum_of_squares_gradient(v)
	
	expected := [0.0, 0.0, 0.0]
	for i := 0; i < grad.len; i++ {
		assert math.abs(grad[i] - expected[i]) < 0.001
	}
}
