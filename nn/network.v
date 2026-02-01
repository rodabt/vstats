module nn

pub struct NeuralNetwork {
mut:
	dense_layers []DenseLayer
	activation_layers []ActivationLayer
	num_layers int
}

pub struct TrainingConfig {
	learning_rate f64
	epochs int
	batch_size int
	verbose bool
}

// Create a sequential neural network
pub fn sequential(layer_sizes []int, activation_fn string) NeuralNetwork {
	assert layer_sizes.len >= 2, "need at least input and output layers"
	
	mut dense_layers := []DenseLayer{}
	mut activation_layers := []ActivationLayer{}
	
	for i in 0 .. layer_sizes.len - 1 {
		dense_layers << dense_layer(layer_sizes[i], layer_sizes[i + 1])
		if i < layer_sizes.len - 2 {
			activation_layers << activation_layer(activation_fn)
		}
	}
	
	return NeuralNetwork{
		dense_layers: dense_layers
		activation_layers: activation_layers
		num_layers: dense_layers.len
	}
}

// Forward pass through network
pub fn (net NeuralNetwork) forward(input []f64) []f64 {
	mut x := input.clone()
	
	for i in 0 .. net.num_layers {
		x = net.dense_layers[i].forward(x)
		
		if i < net.activation_layers.len {
			x = net.activation_layers[i].forward(x)
		}
	}
	
	return x
}

// Backward pass through network
pub fn (mut net NeuralNetwork) backward(grad_output []f64, input []f64, learning_rate f64) []f64 {
	mut grad := grad_output.clone()
	mut activations := [][]f64{len: net.num_layers}
	
	// Forward pass to store activations
	mut x := input.clone()
	activations[0] = input.clone()
	
	for i in 0 .. net.num_layers {
		x = net.dense_layers[i].forward(x)
		
		if i < net.activation_layers.len {
			x = net.activation_layers[i].forward(x)
			if i + 1 < net.num_layers {
				activations[i + 1] = x
			}
		}
	}
	
	// Backward pass
	for i := net.num_layers - 1; i >= 0; i-- {
		if i < net.activation_layers.len {
			grad = net.activation_layers[i].backward(grad, activations[i])
		}
		
		grad = net.dense_layers[i].backward(grad, activations[i], learning_rate)
	}
	
	return grad
}

// Train the network
pub fn (mut net NeuralNetwork) train(x_train [][]f64, y_train []f64, config TrainingConfig) {
	assert x_train.len == y_train.len, "training data size mismatch"
	
	num_samples := x_train.len
	num_batches := (num_samples + config.batch_size - 1) / config.batch_size
	
	for epoch in 0 .. config.epochs {
		mut total_loss := 0.0
		
		for batch_idx in 0 .. num_batches {
			start := batch_idx * config.batch_size
			end := if start + config.batch_size > num_samples { num_samples } else { start + config.batch_size }
			
			mut batch_loss := 0.0
			
			for i in start .. end {
				// Forward pass
				pred := net.forward(x_train[i])
				
				// Compute loss
				error := pred[0] - y_train[i]
				batch_loss += error * error
				
				// Backward pass
				grad := [2 * error]
				net.backward(grad, x_train[i], config.learning_rate)
			}
			
			total_loss += batch_loss / f64(end - start)
		}
		
		if config.verbose && (epoch + 1) % 10 == 0 {
			avg_loss := total_loss / f64(num_batches)
			println('Epoch ${epoch + 1}/${config.epochs}, Loss: ${avg_loss}')
		}
	}
}

// Predict on new data
pub fn (net NeuralNetwork) predict(x [][]f64) [][]f64 {
	mut predictions := [][]f64{len: x.len}
	
	for i in 0 .. x.len {
		predictions[i] = net.forward(x[i])
	}
	
	return predictions
}

// Get prediction for single sample
pub fn (net NeuralNetwork) predict_single(x []f64) []f64 {
	return net.forward(x)
}

// Evaluate network on test data
pub fn (net NeuralNetwork) evaluate(x_test [][]f64, y_test []f64) f64 {
	assert x_test.len == y_test.len, "test data size mismatch"
	
	mut total_loss := 0.0
	
	for i in 0 .. x_test.len {
		pred := net.forward(x_test[i])
		error := pred[0] - y_test[i]
		total_loss += error * error
	}
	
	return total_loss / f64(x_test.len)
}

// Get network parameters
pub fn (net NeuralNetwork) get_weights() [][][]f64 {
	mut weights := [][][]f64{}
	
	for layer in net.dense_layers {
		weights << layer.weights
	}
	
	return weights
}

// Get network biases
pub fn (net NeuralNetwork) get_biases() [][][]f64 {
	mut biases := [][][]f64{}
	
	for layer in net.dense_layers {
		biases << [layer.bias]
	}
	
	return biases
}

// Set network parameters
pub fn (mut net NeuralNetwork) set_weights(weights [][][]f64) {
	assert weights.len == net.num_layers, "weight count mismatch"
	
	for i in 0 .. net.num_layers {
		net.dense_layers[i].weights = weights[i]
	}
}

// Create default training config
pub fn default_training_config() TrainingConfig {
	return TrainingConfig{
		learning_rate: 0.01
		epochs: 100
		batch_size: 32
		verbose: true
	}
}

// Create custom training config
pub fn training_config(lr f64, epochs int, batch_size int) TrainingConfig {
	return TrainingConfig{
		learning_rate: lr
		epochs: epochs
		batch_size: batch_size
		verbose: false
	}
}
