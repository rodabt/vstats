module nn

import math
import rand
import utils
import arrays

pub struct DenseLayer {
pub mut:
	weights     [][]f64  // Stored as [output_size][input_size] for cache efficiency
	bias        []f64
	input_size  int
	output_size int
	// Pre-allocated buffers for performance
	output_buffer []f64
	grad_input_buffer []f64
	// Momentum buffers for accelerated convergence
	weight_velocity [][]f64  // [output_size][input_size]
	bias_velocity   []f64    // [output_size]
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

// dense_layer - Initialize Dense Layer with random weights
// Weights stored as [output_size][input_size] for cache-efficient row-major access
pub fn dense_layer(input_size int, output_size int) DenseLayer {
	// Initialize weights as [output_size][input_size] (transposed for cache efficiency)
	mut weights := [][]f64{len: output_size}
	mut weight_velocity := [][]f64{len: output_size}
	for j in 0 .. output_size {
		weights[j] = []f64{len: input_size}
		weight_velocity[j] = []f64{len: input_size, init: 0}
		for i in 0 .. input_size {
			// Xavier initialization
			weights[j][i] = (rand.f64() - 0.5) / math.sqrt(f64(input_size))
		}
	}

	mut bias := []f64{len: output_size}
	mut bias_velocity := []f64{len: output_size, init: 0}
	for j in 0 .. output_size {
		bias[j] = 0.01
	}

	// Pre-allocate buffers to avoid repeated allocations
	return DenseLayer{
		weights: weights
		bias: bias
		input_size: input_size
		output_size: output_size
		output_buffer: []f64{len: output_size}
		grad_input_buffer: []f64{len: input_size}
		weight_velocity: weight_velocity
		bias_velocity: bias_velocity
	}
}

// forward - Forward pass through dense layer
// Uses row-major weight access pattern for cache efficiency
pub fn (mut layer DenseLayer) forward(input []f64) []f64 {
	assert input.len == layer.input_size, "input size mismatch"

	// Write directly to pre-allocated buffer
	for j in 0 .. layer.output_size {
		mut sum := layer.bias[j]
		// Sequential access: weights[j][i] is row-major (cache-friendly)
		for i in 0 .. layer.input_size {
			sum += input[i] * layer.weights[j][i]
		}
		layer.output_buffer[j] = sum
	}

	return layer.output_buffer
}

// backward - Backward pass through dense layer (returns only gradient, layer updates in-place)
// Optimized with pre-allocated buffer, cache-efficient memory access, and momentum
pub fn (mut layer DenseLayer) backward(grad_output []f64, input []f64, learning_rate f64, momentum f64) []f64 {
	assert grad_output.len == layer.output_size, "output gradient size mismatch"

	// Clear pre-allocated buffer
	for i in 0 .. layer.input_size {
		layer.grad_input_buffer[i] = 0.0
	}

	// Compute input gradient with cache-efficient access
	for j in 0 .. layer.output_size {
		for i in 0 .. layer.input_size {
			layer.grad_input_buffer[i] += grad_output[j] * layer.weights[j][i]
		}
	}

	// Update weights with momentum and cache-efficient sequential access
	for j in 0 .. layer.output_size {
		for i in 0 .. layer.input_size {
			// Compute gradient
			grad := grad_output[j] * input[i]
			// Update velocity: v = momentum * v - learning_rate * grad
			layer.weight_velocity[j][i] = momentum * layer.weight_velocity[j][i] - learning_rate * grad
			// Update weight: w = w + v
			layer.weights[j][i] += layer.weight_velocity[j][i]
		}
	}

	// Update bias with momentum
	for j in 0 .. layer.output_size {
		layer.bias_velocity[j] = momentum * layer.bias_velocity[j] - learning_rate * grad_output[j]
		layer.bias[j] += layer.bias_velocity[j]
	}

	return layer.grad_input_buffer
}

// relu - ReLU Activation Function
pub fn relu(x f64) f64 {
	return if x > 0 { x } else { 0 }
}

// relu_derivative - ReLU Derivative
pub fn relu_derivative(x f64) f64 {
	return if x > 0 { 1 } else { 0 }
}

// sigmoid - Sigmoid Activation Function (delegates to utils)
pub fn sigmoid(x f64) f64 {
	return utils.sigmoid(x)
}

// sigmoid_derivative - Sigmoid Derivative (delegates to utils)
pub fn sigmoid_derivative(x f64) f64 {
	return utils.sigmoid_derivative(x)
}

// tanh - Tanh Activation Function
pub fn tanh(x f64) f64 {
	return math.tanh(x)
}

// tanh_derivative - Tanh Derivative
pub fn tanh_derivative(x f64) f64 {
	t := math.tanh(x)
	return 1 - t * t
}

// softmax - Softmax activation (for output layer)
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

// activation_layer - Create activation layer
pub fn activation_layer(activation_fn string) ActivationLayer {
	return match activation_fn {
		'relu' { ActivationLayer{activation: relu, derivative: relu_derivative} }
		'sigmoid' { ActivationLayer{activation: sigmoid, derivative: sigmoid_derivative} }
		'tanh' { ActivationLayer{activation: tanh, derivative: tanh_derivative} }
		else { ActivationLayer{activation: relu, derivative: relu_derivative} }
	}
}

// forward - Forward pass through activation layer
pub fn (layer ActivationLayer) forward(input []f64) []f64 {
	return input.map(layer.activation(it))
}

// backward - Backward pass through activation layer
pub fn (layer ActivationLayer) backward(grad_output []f64, input []f64) []f64 {
	mut grad_input := []f64{len: grad_output.len}
	for i in 0 .. grad_output.len {
		grad_input[i] = grad_output[i] * layer.derivative(input[i])
	}
	return grad_input
}

// batch_norm_layer - Create Batch Normalization Layer
pub fn batch_norm_layer(input_size int) BatchNormLayer {
	return BatchNormLayer{
		gamma: []f64{len: input_size, init: 1.0}
		beta: []f64{len: input_size, init: 0.0}
		mean: []f64{len: input_size, init: 0.0}
		variance: []f64{len: input_size, init: 1.0}
		epsilon: 1e-5
	}
}

// forward - Forward pass through batch norm
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

// dropout - Apply Dropout regularization
pub fn dropout(input []f64, dropout_rate f64) []f64 {
	if dropout_rate <= 0 {
		return input.clone()
	}
	
	scale := 1.0 / (1.0 - dropout_rate)
	
	return input.map(
		if rand.f64() > dropout_rate { it * scale } else { 0 }
	)
}

// flatten - Flatten 2D array to 1D (use linalg.flatten for generic version)
pub fn flatten(data [][]f64) []f64 {
	mut result := []f64{}
	for row in data {
		result << row
	}
	return result
}

// reshape - Reshape 1D array to 2D
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

// conv1d - 1D Convolution (simplified)
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

// max_pool1d - 1D Max Pooling
pub fn max_pool1d(input []f64, pool_size int, stride int) []f64 {
	output_len := (input.len - pool_size) / stride + 1
	mut output := []f64{len: output_len}
	
	for i in 0 .. output_len {
		window := input[i * stride..i * stride + pool_size]
		max_idx := arrays.idx_max(window) or { 0 }
		output[i] = window[max_idx]
	}
	
	return output
}

// avg_pool1d - 1D Average Pooling
pub fn avg_pool1d(input []f64, pool_size int, stride int) []f64 {
	output_len := (input.len - pool_size) / stride + 1
	mut output := []f64{len: output_len}
	
	for i in 0 .. output_len {
		window := input[i * stride..i * stride + pool_size]
		sum := arrays.sum(window) or { 0.0 }
		output[i] = sum / f64(pool_size)
	}
	
	return output
}
