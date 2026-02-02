import nn
import math

fn test__mse_loss() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	loss := nn.mse_loss(y_true, y_pred)
	assert loss == 0.0, "MSE should be 0 for perfect predictions"
}

fn test__mse_loss_error() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.1, 3.1]
	
	loss := nn.mse_loss(y_true, y_pred)
	// MSE = (0.1^2 + 0.1^2 + 0.1^2) / 3 = 0.01
	assert math.abs(loss - 0.01) < 0.001
}

fn test__mse_loss_gradient() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.1, 3.1]
	
	grad := nn.mse_loss_gradient(y_true, y_pred)
	assert grad.len == 3
	// Gradient points in direction to reduce error
	assert grad[0] > 0.0
}

fn test__mae_loss() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	loss := nn.mae_loss(y_true, y_pred)
	assert loss == 0.0, "MAE should be 0 for perfect predictions"
}

fn test__mae_loss_error() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.1, 3.1]
	
	loss := nn.mae_loss(y_true, y_pred)
	// MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
	assert math.abs(loss - 0.1) < 0.001
}

fn test__mae_loss_gradient() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.1, 2.1, 3.1]
	
	grad := nn.mae_loss_gradient(y_true, y_pred)
	assert grad.len == 3
}

fn test__binary_crossentropy_loss() {
	y_true := [1.0, 0.0, 1.0]
	y_pred := [0.9, 0.1, 0.8]
	
	loss := nn.binary_crossentropy_loss(y_true, y_pred)
	// Should be positive
	assert loss > 0.0
	// Should be relatively small for good predictions
	assert loss < 0.5
}

fn test__binary_crossentropy_loss_bad() {
	y_true := [1.0, 0.0]
	y_pred := [0.1, 0.9]
	
	loss := nn.binary_crossentropy_loss(y_true, y_pred)
	// Should be large for bad predictions
	assert loss > 0.5
}

fn test__binary_crossentropy_loss_gradient() {
	y_true := [1.0, 0.0]
	y_pred := [0.9, 0.1]
	
	grad := nn.binary_crossentropy_loss_gradient(y_true, y_pred)
	assert grad.len == 2
}

fn test__categorical_crossentropy_loss() {
	y_true := [
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0],
	]
	y_pred := [
		[0.9, 0.05, 0.05],
		[0.05, 0.9, 0.05],
		[0.05, 0.05, 0.9],
	]
	
	loss := nn.categorical_crossentropy_loss(y_true, y_pred)
	// Should be small for good predictions
	assert loss > 0.0 && loss < 0.5
}

fn test__categorical_crossentropy_loss_bad() {
	y_true := [
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
	]
	y_pred := [
		[0.1, 0.4, 0.5],
		[0.4, 0.1, 0.5],
	]
	
	loss := nn.categorical_crossentropy_loss(y_true, y_pred)
	// Should be larger for bad predictions
	assert loss > 0.5
}

fn test__hinge_loss() {
	y_true := [1.0, -1.0, 1.0]
	y_pred := [2.0, -2.0, 1.5]
	
	loss := nn.hinge_loss(y_true, y_pred)
	// Should be non-negative
	assert loss >= 0.0
}

fn test__huber_loss() {
	y_true := [1.0, 2.0, 3.0]
	y_pred := [1.0, 2.0, 3.0]
	
	loss := nn.huber_loss(y_true, y_pred, 1.0)
	assert loss == 0.0, "Huber loss should be 0 for perfect predictions"
}

fn test__kl_divergence_loss() {
	y_true := [0.3, 0.5, 0.2]
	y_pred := [0.25, 0.55, 0.2]
	
	loss := nn.kl_divergence_loss(y_true, y_pred)
	// Should be non-negative (KL divergence is always >= 0)
	assert loss >= 0.0
}

fn test__cosine_similarity_loss() {
	y_true := [1.0, 0.0]
	y_pred := [1.0, 0.0]
	
	loss := nn.cosine_similarity_loss(y_true, y_pred)
	// For identical vectors, cosine similarity is 1, loss should be ~0
	assert loss < 0.1
}

fn test__contrastive_loss() {
	y_true := 1.0
	distance := 0.1
	
	loss := nn.contrastive_loss(y_true, distance, 1.0)
	assert loss > 0.0
}

fn test__triplet_loss() {
	anchor := [0.0, 0.0]
	positive := [0.1, 0.1]
	negative := [0.2, 0.2]
	
	loss := nn.triplet_loss(anchor, positive, negative, 1.0)
	// dist_ap ≈ 0.141, dist_an ≈ 0.283
	// max(0, 0.141 - 0.283 + 1.0) = max(0, 0.858) = 0.858
	assert loss > 0.0
}

fn test__squared_hinge_loss() {
	y_true := [1.0, -1.0, 1.0]
	y_pred := [2.0, -2.0, 1.5]
	
	loss := nn.squared_hinge_loss(y_true, y_pred)
	// Should be non-negative
	assert loss >= 0.0
}

fn test__sparse_categorical_crossentropy_loss() {
	y_true := [0, 1, 2]
	y_pred := [
		[0.9, 0.05, 0.05],
		[0.05, 0.9, 0.05],
		[0.05, 0.05, 0.9],
	]
	
	loss := nn.sparse_categorical_crossentropy_loss(y_true, y_pred)
	// Should be small for good predictions
	assert loss > 0.0 && loss < 0.5
}
