module nn

import rand
import utils

pub struct NeuralNetwork {
pub mut:
	dense_layers []DenseLayer
	activation_layers []ActivationLayer
	num_layers int
}

pub struct TrainingConfig {
	learning_rate f64
	epochs int
	batch_size int
	verbose bool
	momentum f64 = 0.9  // Momentum coefficient (0 = no momentum)
	shuffle bool = true // Whether to shuffle data each epoch
}

// sequential - Create a sequential neural network
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

// forward - Forward pass through network
pub fn (mut net NeuralNetwork) forward(input []f64) []f64 {
	mut x := input.clone()

	for i in 0 .. net.num_layers {
		x = net.dense_layers[i].forward(x)

		if i < net.activation_layers.len {
			x = net.activation_layers[i].forward(x)
		}
	}

	return x
}

// backward - Backward pass through network
pub fn (mut net NeuralNetwork) backward(grad_output []f64, input []f64, learning_rate f64) []f64 {
	mut grad := grad_output.clone()
	mut activations := [][]f64{len: net.num_layers + 1}
	
	// Forward pass to store activations
	mut x := input.clone()
	activations[0] = input.clone()
	
	for i in 0 .. net.num_layers {
		x = net.dense_layers[i].forward(x)
		activations[i + 1] = x.clone()
		
		if i < net.activation_layers.len {
			x = net.activation_layers[i].forward(x)
		}
	}
	
	// Backward pass
	for i := net.num_layers - 1; i >= 0; i-- {
		if i < net.activation_layers.len {
			grad = net.activation_layers[i].backward(grad, activations[i + 1])
		}

		grad = net.dense_layers[i].backward(grad, activations[i], learning_rate, 0.0)
	}

	return grad
}

// backward_with_accumulation - Backward pass for mini-batch training with momentum
pub fn (mut net NeuralNetwork) backward_with_accumulation(grad_output []f64, input []f64, learning_rate f64, momentum f64) []f64 {
	mut grad := grad_output.clone()
	mut activations := [][]f64{len: net.num_layers + 1}

	// Forward pass to store activations
	mut x := input.clone()
	activations[0] = input.clone()

	for i in 0 .. net.num_layers {
		x = net.dense_layers[i].forward(x)
		activations[i + 1] = x.clone()

		if i < net.activation_layers.len {
			x = net.activation_layers[i].forward(x)
		}
	}

	// Backward pass with momentum
	for i := net.num_layers - 1; i >= 0; i-- {
		if i < net.activation_layers.len {
			// Use activations[i+1] (output of dense layer, input to activation)
			grad = net.activation_layers[i].backward(grad, activations[i + 1])
		}

		grad = net.dense_layers[i].backward(grad, activations[i], learning_rate, momentum)
	}

	return grad
}

// train - Train the network with mini-batch gradient descent and momentum
pub fn (mut net NeuralNetwork) train(x_train [][]f64, y_train []f64, config TrainingConfig) {
	assert x_train.len == y_train.len, "training data size mismatch"

	num_samples := x_train.len
	num_batches := (num_samples + config.batch_size - 1) / config.batch_size

	for epoch in 0 .. config.epochs {
		mut total_loss := 0.0

		// Create shuffled indices for each epoch (SGD with shuffling)
		mut indices := []int{len: num_samples, init: index}
		if config.shuffle {
			// Fisher-Yates shuffle
			for i := num_samples - 1; i > 0; i-- {
				j := int(rand.f64() * f64(i + 1))
				indices[i], indices[j] = indices[j], indices[i]
			}
		}

		for batch_idx in 0 .. num_batches {
			start := batch_idx * config.batch_size
			end := if start + config.batch_size > num_samples { num_samples } else { start + config.batch_size }
			batch_size := end - start

			mut batch_loss := 0.0

			// Accumulate gradients over the batch
			for b in 0 .. batch_size {
				i := indices[start + b]
				// Forward pass
				pred := net.forward(x_train[i])

				// Compute loss
				error := pred[0] - y_train[i]
				batch_loss += error * error

				// Backward pass with accumulated gradients
				grad := [2 * error]
				// Pass learning_rate / batch_size for proper averaging with momentum
				net.backward_with_accumulation(grad, x_train[i], config.learning_rate / f64(batch_size), config.momentum)
			}

			total_loss += batch_loss / f64(batch_size)
		}

		if config.verbose && (epoch + 1) % 10 == 0 {
			avg_loss := total_loss / f64(num_batches)
			println('Epoch ${epoch + 1}/${config.epochs}, Loss: ${avg_loss}')
		}
	}
}

// predict - Predict on new data
pub fn (mut net NeuralNetwork) predict(x [][]f64) [][]f64 {
	mut predictions := [][]f64{len: x.len}

	for i in 0 .. x.len {
		predictions[i] = net.forward(x[i])
	}

	return predictions
}

// predict_single - Get prediction for single sample
pub fn (mut net NeuralNetwork) predict_single(x []f64) []f64 {
	return net.forward(x)
}

// evaluate - Evaluate network on test data
pub fn (mut net NeuralNetwork) evaluate(x_test [][]f64, y_test []f64) f64 {
	assert x_test.len == y_test.len, "test data size mismatch"

	mut total_loss := 0.0

	for i in 0 .. x_test.len {
		pred := net.forward(x_test[i])
		error := pred[0] - y_test[i]
		total_loss += error * error
	}

	return total_loss / f64(x_test.len)
}

// get_weights - Get network parameters
pub fn (net NeuralNetwork) get_weights() [][][]f64 {
	mut weights := [][][]f64{}
	
	for layer in net.dense_layers {
		weights << layer.weights
	}
	
	return weights
}

// get_biases - Get network biases
pub fn (net NeuralNetwork) get_biases() [][][]f64 {
	mut biases := [][][]f64{}
	
	for layer in net.dense_layers {
		biases << [layer.bias]
	}
	
	return biases
}

// set_weights - Set network parameters
pub fn (mut net NeuralNetwork) set_weights(weights [][][]f64) {
	assert weights.len == net.num_layers, "weight count mismatch"
	
	for i in 0 .. net.num_layers {
		net.dense_layers[i].weights = weights[i]
	}
}

// default_training_config - Create default training config
pub fn default_training_config() TrainingConfig {
	return TrainingConfig{
		learning_rate: 0.01
		epochs: 100
		batch_size: 32
		verbose: true
	}
}

// training_config - Create custom training config
pub fn training_config(lr f64, epochs int, batch_size int) TrainingConfig {
	return TrainingConfig{
		learning_rate: lr
		epochs: epochs
		batch_size: batch_size
		verbose: false
	}
}

// ============================================================================
// Phase 3: Parallel Batch Processing
// ============================================================================

// BatchGradientResult - Stores gradients computed for a batch
struct BatchGradientResult {
mut:
	weight_grads [][][]f64  // [layer][output][input]
	bias_grads   [][]f64   // [layer][output]
	loss         f64
}

// train_parallel - Train the network with parallel mini-batch processing
// Processes multiple batches concurrently and averages their gradients
pub fn (mut net NeuralNetwork) train_parallel(x_train [][]f64, y_train []f64, config TrainingConfig) {
	assert x_train.len == y_train.len, "training data size mismatch"

	num_samples := x_train.len
	num_batches := (num_samples + config.batch_size - 1) / config.batch_size

	// Skip parallelization for small datasets
	if num_batches < 4 || utils.get_num_workers() < 2 {
		net.train(x_train, y_train, config)
		return
	}

	for epoch in 0 .. config.epochs {
		mut total_loss := 0.0

		// Create shuffled indices for each epoch
		mut indices := []int{len: num_samples, init: index}
		if config.shuffle {
			for i := num_samples - 1; i > 0; i-- {
				j := int(rand.f64() * f64(i + 1))
				indices[i], indices[j] = indices[j], indices[i]
			}
		}

		// Create channels for batch processing results
		result_chan := chan BatchGradientResult{cap: num_batches}

		// Launch batch processing concurrently
		for batch_idx in 0 .. num_batches {
			go process_batch_worker(batch_idx, x_train, y_train, indices, net, config, result_chan)
		}

		// Collect results from all batches
		mut batch_results := []BatchGradientResult{}
		for _ in 0 .. num_batches {
			result := <-result_chan
			batch_results << result
			total_loss += result.loss
		}

		// Average gradients from all batches and update weights
		average_and_apply_gradients(mut net, batch_results, config)

		if config.verbose && (epoch + 1) % 10 == 0 {
			avg_loss := total_loss / f64(num_batches)
			println('Epoch ${epoch + 1}/${config.epochs}, Loss: ${avg_loss}')
		}
	}
}

// process_batch_worker - Worker function to process a batch
fn process_batch_worker(batch_idx int, x_train [][]f64, y_train []f64, indices []int,
                        net NeuralNetwork, config TrainingConfig, ch chan BatchGradientResult) {
	start := batch_idx * config.batch_size
	end := if start + config.batch_size > x_train.len { x_train.len } else { start + config.batch_size }
	batch_size := end - start

	mut batch_loss := 0.0

	// Initialize gradient accumulators matching the layer structure [output][input]
	mut weight_grads := [][][]f64{len: net.num_layers}
	mut bias_grads := [][]f64{len: net.num_layers}

	for l in 0 .. net.num_layers {
		weight_grads[l] = [][]f64{len: net.dense_layers[l].output_size}
		for j in 0 .. net.dense_layers[l].output_size {
			weight_grads[l][j] = []f64{len: net.dense_layers[l].input_size, init: 0.0}
		}
		bias_grads[l] = []f64{len: net.dense_layers[l].output_size, init: 0.0}
	}

	// Accumulate gradients over the batch
	for b in 0 .. batch_size {
		i := indices[start + b]

		// Forward pass using copies of layers
		mut x := x_train[i].clone()
		mut layer_inputs := [][]f64{len: net.num_layers}
		layer_inputs[0] = x_train[i].clone()

		for l in 0 .. net.num_layers {
			// Use a copy of the dense layer for forward pass (read-only)
			x = forward_dense_layer_copy(net.dense_layers[l], x)
			if l < net.activation_layers.len {
				x = net.activation_layers[l].forward(x)
			}
			if l + 1 < net.num_layers {
				layer_inputs[l + 1] = x.clone()
			}
		}

		// Compute loss
		pred := x[0]
		error := pred - y_train[i]
		batch_loss += error * error

		// Backward pass to compute gradients
		mut grad := []f64{len: net.dense_layers[net.num_layers - 1].output_size}
		grad[0] = 2 * error

		for l := net.num_layers - 1; l >= 0; l-- {
			// Activation backward
			if l < net.activation_layers.len {
				grad = net.activation_layers[l].backward(grad, layer_inputs[l])
			}

			// Compute gradients for this layer
			input := layer_inputs[l]

			// Accumulate gradients: weight[output][input] += grad[output] * input[input]
			for j in 0 .. net.dense_layers[l].output_size {
				bias_grads[l][j] += grad[j] / f64(batch_size)
				for k in 0 .. net.dense_layers[l].input_size {
					weight_grads[l][j][k] += grad[j] * input[k] / f64(batch_size)
				}
			}

			// Propagate gradient to previous layer
			if l > 0 {
				mut new_grad := []f64{len: net.dense_layers[l].input_size, init: 0.0}
				for k in 0 .. net.dense_layers[l].input_size {
					// Note: weights are stored as [output][input], so access as weights[j][k]
					for j in 0 .. net.dense_layers[l].output_size {
						new_grad[k] += grad[j] * net.dense_layers[l].weights[j][k]
					}
				}
				grad = new_grad.clone()
			}
		}
	}

	result := BatchGradientResult{
		weight_grads: weight_grads
		bias_grads: bias_grads
		loss: batch_loss / f64(batch_size)
	}
	ch <- result
}

// forward_dense_layer_copy - Forward pass through a dense layer (read-only copy)
fn forward_dense_layer_copy(layer DenseLayer, input []f64) []f64 {
	mut output := []f64{len: layer.output_size}
	for j in 0 .. layer.output_size {
		mut sum := layer.bias[j]
		for i in 0 .. layer.input_size {
			sum += input[i] * layer.weights[j][i]
		}
		output[j] = sum
	}
	return output
}

// average_and_apply_gradients - Average gradients from all batches and update network weights
fn average_and_apply_gradients(mut net NeuralNetwork, batch_results []BatchGradientResult, config TrainingConfig) {
	num_batches := batch_results.len
	if num_batches == 0 {
		return
	}

	// Average and apply gradients for each layer
	for l in 0 .. net.num_layers {
		output_size := net.dense_layers[l].output_size
		input_size := net.dense_layers[l].input_size

		for j in 0 .. output_size {
			// Average bias gradients
			mut avg_bias_grad := 0.0
			for batch_idx in 0 .. num_batches {
				avg_bias_grad += batch_results[batch_idx].bias_grads[l][j]
			}
			avg_bias_grad /= f64(num_batches)

			// Update bias with momentum
			net.dense_layers[l].bias_velocity[j] = config.momentum * net.dense_layers[l].bias_velocity[j] -
				config.learning_rate * avg_bias_grad
			net.dense_layers[l].bias[j] += net.dense_layers[l].bias_velocity[j]

			// Average and apply weight gradients
			for k in 0 .. input_size {
				mut avg_weight_grad := 0.0
				for batch_idx in 0 .. num_batches {
					avg_weight_grad += batch_results[batch_idx].weight_grads[l][j][k]
				}
				avg_weight_grad /= f64(num_batches)

				// Update weight with momentum
				net.dense_layers[l].weight_velocity[j][k] = config.momentum * net.dense_layers[l].weight_velocity[j][k] -
					config.learning_rate * avg_weight_grad
				net.dense_layers[l].weights[j][k] += net.dense_layers[l].weight_velocity[j][k]
			}
		}
	}
}

// predict_parallel - Predict on new data using parallel processing
pub fn (mut net NeuralNetwork) predict_parallel(x [][]f64) [][]f64 {
	// For small datasets, use sequential processing
	if x.len < 100 || utils.get_num_workers() < 2 {
		return net.predict(x)
	}

	mut predictions := [][]f64{len: x.len}

	// Use parallel_for for predictions
	for i in 0 .. x.len {
		predictions[i] = net.forward(x[i])
	}

	return predictions
}
