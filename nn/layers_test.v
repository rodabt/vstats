module nn

import math

fn test__dense_layer_init() {
	layer := dense_layer(3, 2)
	
	assert layer.input_size == 3
	assert layer.output_size == 2
	assert layer.weights.len == 3
	assert layer.weights[0].len == 2
	assert layer.bias.len == 2
}

fn test__dense_layer_forward() {
	layer := dense_layer(2, 2)
	input := [1.0, 2.0]
	
	output := layer.forward(input)
	
	assert output.len == 2
}

fn test__relu_activation() {
	assert relu(0.0) == 0.0
	assert relu(1.0) == 1.0
	assert relu(-1.0) == 0.0
	assert relu(5.0) == 5.0
}

fn test__relu_derivative() {
	assert relu_derivative(1.0) == 1.0
	assert relu_derivative(-1.0) == 0.0
	assert relu_derivative(0.0) == 0.0
}

fn test__sigmoid() {
	// sigmoid(0) = 0.5
	assert math.abs(sigmoid(0.0) - 0.5) < 0.001
	// sigmoid is monotonic increasing
	assert sigmoid(-10.0) < sigmoid(0.0)
	assert sigmoid(0.0) < sigmoid(10.0)
	// sigmoid output is in (0, 1)
	assert sigmoid(10.0) > 0.99
	assert sigmoid(-10.0) < 0.01
}

fn test__sigmoid_derivative() {
	// At 0, derivative of sigmoid is 0.25
	assert math.abs(sigmoid_derivative(0.0) - 0.25) < 0.001
	// Derivative is always positive
	for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
		assert sigmoid_derivative(x) > 0.0
	}
}

fn test__tanh_activation() {
	// tanh(0) = 0
	assert math.abs(tanh(0.0)) < 0.001
	// tanh output is in (-1, 1)
	assert tanh(10.0) > 0.99
	assert tanh(-10.0) < -0.99
}

fn test__tanh_derivative() {
	// At 0, derivative of tanh is 1
	assert math.abs(tanh_derivative(0.0) - 1.0) < 0.001
}

fn test__softmax() {
	x := [1.0, 2.0, 3.0]
	probs := softmax(x)
	
	// Should have same length
	assert probs.len == 3
	// Should sum to 1
	mut sum := 0.0
	for p in probs {
		sum += p
		// Each probability between 0 and 1
		assert p > 0.0 && p < 1.0
	}
	assert math.abs(sum - 1.0) < 0.001
}

fn test__softmax_argmax() {
	x := [1.0, 2.0, 3.0]
	probs := softmax(x)
	
	// Last element should have highest probability
	assert probs[2] > probs[1]
	assert probs[1] > probs[0]
}

fn test__dropout() {
	input := [1.0, 2.0, 3.0, 4.0, 5.0]
	output := dropout(input, 0.5)
	
	assert output.len == input.len
}

fn test__flatten() {
	data := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
	flat := flatten(data)
	
	assert flat.len == 6
	assert flat[0] == 1.0
	assert flat[5] == 6.0
}

fn test__reshape() {
	flat := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	reshaped := reshape(flat, 2, 3)
	
	assert reshaped.len == 2
	assert reshaped[0].len == 3
	assert reshaped[0][0] == 1.0
	assert reshaped[1][2] == 6.0
}

fn test__batch_norm_layer() {
	layer := batch_norm_layer(3)
	
	assert layer.gamma.len == 3
	assert layer.beta.len == 3
	assert layer.mean.len == 3
	assert layer.variance.len == 3
}
