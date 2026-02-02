module nn

import math
import rand

pub struct DenseLayer {
pub mut:
	weights     [][]f64
	bias        []f64
	input_size  int
	output_size int
}

pub struct ActivationLayer {
pub mut:
	activation fn(f64) f64 @[required]
	derivative fn(f64) f64 @[required]
}

pub struct BatchNormLayer {
pub mut:
	gamma []f64
	beta  []f64
	mean  []f64
	variance []f64
	epsilon f64
}

// Initialize Dense Layer with random weights
pub fn dense_layer(input_size int, output_size int) DenseLayer {
	mut weights := [][]f64{len: input_size}
	for i in 0 .. input_size {
		weights[i] = []f64{len: output_size}
		for j in 0 .. output_size {
			// Xavier initialization
			weights[i][j] = (rand.f64() - 0.5) / math.sqrt(f64(input_size))
		}
	}
	
	mut bias := []f64{len: output_size}
	for j in 0 .. output_size {
		bias[j] = 0.01
	}
	
	return DenseLayer{
		weights: weights
		bias: bias
		input_size: input_size
		output_size: output_size
	}
}

// Forward pass through dense layer
pub fn (layer DenseLayer) forward(input []f64) []f64 {
	assert input.len == layer.input_size, "input size mismatch"
	
	mut output := []f64{len: layer.output_size}
	for j in 0 .. layer.output_size {
		mut sum := layer.bias[j]
		for i in 0 .. layer.input_size {
			sum += input[i] * layer.weights[i][j]
		}
		output[j] = sum
	}
	
	return output
}

// Backward pass through dense layer (returns only gradient, layer updates in-place)
pub fn (mut layer DenseLayer) backward(grad_output []f64, input []f64, learning_rate f64) []f64 {
	assert grad_output.len == layer.output_size, "output gradient size mismatch"
	
	// Compute input gradient
	mut grad_input := []f64{len: layer.input_size}
	for i in 0 .. layer.input_size {
		mut sum := 0.0
		for j in 0 .. layer.output_size {
			sum += grad_output[j] * layer.weights[i][j]
		}
		grad_input[i] = sum
	}
	
	// Update weights
	for i in 0 .. layer.input_size {
		for j in 0 .. layer.output_size {
			layer.weights[i][j] -= learning_rate * grad_output[j] * input[i]
		}
	}
	
	// Update bias
	for j in 0 .. layer.output_size {
		layer.bias[j] -= learning_rate * grad_output[j]
	}
	
	return grad_input
}

// ReLU Activation Function
pub fn relu(x f64) f64 {
	return if x > 0 { x } else { 0 }
}

// ReLU Derivative
pub fn relu_derivative(x f64) f64 {
	return if x > 0 { 1 } else { 0 }
}

// Sigmoid Activation Function
pub fn sigmoid(x f64) f64 {
	return 1.0 / (1.0 + math.exp(-x))
}

// Sigmoid Derivative
pub fn sigmoid_derivative(x f64) f64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// Tanh Activation Function
pub fn tanh(x f64) f64 {
	return math.tanh(x)
}

// Tanh Derivative
pub fn tanh_derivative(x f64) f64 {
	t := math.tanh(x)
	return 1 - t * t
}

// Softmax activation (for output layer)
pub fn softmax(x []f64) []f64 {
	// Subtract max for numerical stability
	mut max_val := x[0]
	for v in x {
		if v > max_val {
			max_val = v
		}
	}
	
	mut exp_vals := []f64{len: x.len}
	mut sum := 0.0
	for i, v in x {
		exp_vals[i] = math.exp(v - max_val)
		sum += exp_vals[i]
	}
	
	return exp_vals.map(it / sum)
}

// Create activation layer
pub fn activation_layer(activation_fn string) ActivationLayer {
	return match activation_fn {
		'relu' { ActivationLayer{activation: relu, derivative: relu_derivative} }
		'sigmoid' { ActivationLayer{activation: sigmoid, derivative: sigmoid_derivative} }
		'tanh' { ActivationLayer{activation: tanh, derivative: tanh_derivative} }
		else { ActivationLayer{activation: relu, derivative: relu_derivative} }
	}
}

// Forward pass through activation layer
pub fn (layer ActivationLayer) forward(input []f64) []f64 {
	return input.map(layer.activation(it))
}

// Backward pass through activation layer
pub fn (layer ActivationLayer) backward(grad_output []f64, input []f64) []f64 {
	mut grad_input := []f64{len: grad_output.len}
	for i in 0 .. grad_output.len {
		grad_input[i] = grad_output[i] * layer.derivative(input[i])
	}
	return grad_input
}

// Batch Normalization Layer
pub fn batch_norm_layer(input_size int) BatchNormLayer {
	return BatchNormLayer{
		gamma: []f64{len: input_size, init: 1.0}
		beta: []f64{len: input_size, init: 0.0}
		mean: []f64{len: input_size, init: 0.0}
		variance: []f64{len: input_size, init: 1.0}
		epsilon: 1e-5
	}
}

// Forward pass through batch norm
pub fn (layer BatchNormLayer) forward(input []f64) []f64 {
	mut output := []f64{len: input.len}
	
	for i in 0 .. input.len {
		// Normalize
		normalized := (input[i] - layer.mean[i]) / math.sqrt(layer.variance[i] + layer.epsilon)
		// Scale and shift
		output[i] = layer.gamma[i] * normalized + layer.beta[i]
	}
	
	return output
}

// Dropout Layer (regularization)
pub fn dropout(input []f64, dropout_rate f64) []f64 {
	if dropout_rate <= 0 {
		return input.clone()
	}
	
	scale := 1.0 / (1.0 - dropout_rate)
	
	return input.map(
		if rand.f64() > dropout_rate { it * scale } else { 0 }
	)
}

// Flatten 2D array to 1D
pub fn flatten(data [][]f64) []f64 {
	mut result := []f64{}
	for row in data {
		result << row
	}
	return result
}

// Reshape 1D array to 2D
pub fn reshape(data []f64, rows int, cols int) [][]f64 {
	assert data.len == rows * cols, "size mismatch for reshape"
	
	mut result := [][]f64{len: rows}
	for i in 0 .. rows {
		result[i] = []f64{len: cols}
		for j in 0 .. cols {
			result[i][j] = data[i * cols + j]
		}
	}
	return result
}

// Convolution 1D (simplified)
pub fn conv1d(input []f64, kernel []f64, stride int) []f64 {
	output_len := (input.len - kernel.len) / stride + 1
	mut output := []f64{len: output_len}
	
	for i in 0 .. output_len {
		mut sum := 0.0
		for j in 0 .. kernel.len {
			sum += input[i * stride + j] * kernel[j]
		}
		output[i] = sum
	}
	
	return output
}

// Max Pooling 1D
pub fn max_pool1d(input []f64, pool_size int, stride int) []f64 {
	output_len := (input.len - pool_size) / stride + 1
	mut output := []f64{len: output_len}
	
	for i in 0 .. output_len {
		mut max_val := input[i * stride]
		for j in 1 .. pool_size {
			if input[i * stride + j] > max_val {
				max_val = input[i * stride + j]
			}
		}
		output[i] = max_val
	}
	
	return output
}

// Average Pooling 1D
pub fn avg_pool1d(input []f64, pool_size int, stride int) []f64 {
	output_len := (input.len - pool_size) / stride + 1
	mut output := []f64{len: output_len}
	
	for i in 0 .. output_len {
		mut sum := 0.0
		for j in 0 .. pool_size {
			sum += input[i * stride + j]
		}
		output[i] = sum / f64(pool_size)
	}
	
	return output
}
