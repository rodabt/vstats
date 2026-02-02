module utils

import math

// Sigmoid activation function
// Converts any numeric type to f64 for stable computation
pub fn sigmoid[T](x T) T {
	x_f64 := f64(x)
	result := 1.0 / (1.0 + math.exp(-x_f64))
	return T(result)
}

// Sigmoid Derivative: sig'(x) = sig(x) * (1 - sig(x))
pub fn sigmoid_derivative[T](x T) T {
	sig := sigmoid(x)
	return sig * (T(1) - sig)
}